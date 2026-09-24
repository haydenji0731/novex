import pytest
from pydantic import ValidationError

from novex.chains import Chain, GInterval, Strand
from novex.junctions import Junction, JunctionIndex, read_bed


def jn(start, end, strand=Strand.PLUS, chrom="chr1", **kw):
    return Junction(chrom=chrom, strand=strand, intron=GInterval(start, end), **kw)


# --- Junction ----------------------------------------------------------------

def test_junction_rejects_backwards_intron():
    with pytest.raises(ValidationError):
        jn(200, 100)


@pytest.mark.parametrize(
    "strand, donor, acceptor, donor_exon, acceptor_exon",
    [
        # intron 101-200: on '+' transcription enters the intron at 101
        (Strand.PLUS, 101, 200, 100, 201),
        # on '-' it enters at 200, reading right to left
        (Strand.MINUS, 200, 101, 201, 100),
    ],
)
def test_junction_sides(strand, donor, acceptor, donor_exon, acceptor_exon):
    j = jn(101, 200, strand)
    assert j.donor == donor
    assert j.acceptor == acceptor
    assert j.donor_exon_base == donor_exon
    assert j.acceptor_exon_base == acceptor_exon


def test_junction_unknown_strand_has_no_sides():
    j = jn(101, 200, Strand.UNKNOWN)
    with pytest.raises(ValueError):
        j.donor


def test_junction_is_hashable():
    assert len({jn(101, 200), jn(101, 200)}) == 1


# --- read_bed ----------------------------------------------------------------

def write_bed(tmp_path, text):
    p = tmp_path / "juncs.bed"
    p.write_text(text)
    return p


def test_read_bed_converts_to_1_based_inclusive(tmp_path):
    # BED [100, 200) spans intron bases 101-200 in novex coordinates
    p = write_bed(tmp_path, "chr1\t100\t200\tJUNC1\t5\t+\n")
    (j,) = list(read_bed(p))
    assert j.chrom == "chr1"
    assert j.intron == GInterval(101, 200)
    assert j.strand == Strand.PLUS
    assert j.name == "JUNC1"
    assert j.score == 5


def test_read_bed_keeps_the_first_line(tmp_path):
    # regression: the old loaders skipped line 1 as a header
    p = write_bed(tmp_path, "chr1\t100\t200\tA\t1\t+\nchr1\t300\t400\tB\t1\t+\n")
    assert len(list(read_bed(p))) == 2


def test_read_bed_skips_unusable_lines(tmp_path):
    p = write_bed(
        tmp_path,
        "track name=junctions\n"
        "# a comment\n"
        "\n"
        "chr1\t100\t200\tA\t1\t+\n"
        "chr1\t300\t400\tB\t1\t.\n"  # unknown strand -> no reading direction
        "chr2\t500\t600\tC\t1\t-\n",
    )
    juncs = list(read_bed(p))
    assert [(j.chrom, j.strand) for j in juncs] == [
        ("chr1", Strand.PLUS),
        ("chr2", Strand.MINUS),
    ]


def test_read_bed_allows_missing_name_and_score(tmp_path):
    p = write_bed(tmp_path, "chr1\t100\t200\t.\t.\t+\n")
    (j,) = list(read_bed(p))
    assert j.intron == GInterval(101, 200)


# --- JunctionIndex -----------------------------------------------------------

CHAIN = Chain([(10, 19), (30, 39)])


# built lazily: Junction() runs its validator, so these cannot be module-level
def inside():
    """donor_exon_base on '+' is intron.start - 1 -> 15, 19, 30"""
    return [jn(16, 60), jn(20, 60), jn(31, 60)]


def outside():
    return [
        jn(6, 60),                  # 5: before the chain
        jn(26, 60),                 # 25: in the chain's gap
        jn(46, 60),                 # 45: past the chain
        jn(16, 60, Strand.MINUS),   # right position, wrong strand
        jn(16, 60, chrom="chr2"),   # right position, wrong contig
    ]


def test_index_len():
    assert len(JunctionIndex(inside() + outside())) == len(inside()) + len(outside())


def test_index_finds_only_donors_inside_the_chain():
    idx = JunctionIndex(inside() + outside())
    found = idx.donors_in("chr1", Strand.PLUS, CHAIN)
    assert [j.donor_exon_base for j in found] == [15, 19, 30]


def test_index_returns_empty_for_unknown_key():
    idx = JunctionIndex(inside())
    assert idx.donors_in("chrX", Strand.PLUS, CHAIN) == []
    assert idx.donors_in("chr1", Strand.MINUS, CHAIN) == []


def test_index_matches_chain_boundaries():
    idx = JunctionIndex([jn(11, 60), jn(20, 60)])  # donor exon bases 10 and 19
    found = idx.donors_in("chr1", Strand.PLUS, CHAIN)
    assert [j.donor_exon_base for j in found] == [10, 19]


def test_index_minus_strand_uses_the_other_intron_end():
    # on '-', donor_exon_base is intron.end + 1
    chain = Chain([(300, 330)])
    idx = JunctionIndex([jn(100, 299, Strand.MINUS), jn(100, 340, Strand.MINUS)])
    found = idx.donors_in("chr1", Strand.MINUS, chain)
    assert [j.donor_exon_base for j in found] == [300]


def test_index_is_empty_when_built_from_nothing():
    idx = JunctionIndex([])
    assert len(idx) == 0
    assert idx.donors_in("chr1", Strand.PLUS, CHAIN) == []


# --- acceptors_in -------------------------------------------------------------

# acceptor_exon_base on '+' is intron.end + 1; on '-' it is intron.start - 1
def acceptor_inside():
    """introns whose acceptor exon base lands at 10, 19 and 30"""
    return [jn(5, 9), jn(5, 18), jn(5, 29)]


def acceptor_outside():
    return [
        jn(1, 4),                    # 5: before the chain
        jn(5, 24),                   # 25: in the chain's gap
        jn(5, 44),                   # 45: past the chain
        jn(5, 9, Strand.MINUS),      # right position on the wrong strand
        jn(5, 9, chrom="chr2"),      # right position on the wrong contig
    ]


def test_index_finds_only_acceptors_inside_the_chain():
    idx = JunctionIndex(acceptor_inside() + acceptor_outside())
    found = idx.acceptors_in("chr1", Strand.PLUS, CHAIN)
    assert [j.acceptor_exon_base for j in found] == [10, 19, 30]


def test_acceptors_and_donors_are_indexed_independently():
    # one junction: donor exon base 15, acceptor exon base 61
    j = jn(16, 60)
    idx = JunctionIndex([j])
    assert idx.donors_in("chr1", Strand.PLUS, CHAIN) == [j]
    assert idx.acceptors_in("chr1", Strand.PLUS, CHAIN) == []
    assert idx.acceptors_in("chr1", Strand.PLUS, Chain([(55, 70)])) == [j]


def test_acceptors_in_minus_strand_uses_the_other_intron_end():
    # on '-', acceptor_exon_base is intron.start - 1
    chain = Chain([(300, 330)])
    idx = JunctionIndex([jn(331, 400, Strand.MINUS), jn(340, 400, Strand.MINUS)])
    found = idx.acceptors_in("chr1", Strand.MINUS, chain)
    assert [j.acceptor_exon_base for j in found] == [330]


def test_acceptors_in_returns_empty_for_unknown_key():
    idx = JunctionIndex(acceptor_inside())
    assert idx.acceptors_in("chrX", Strand.PLUS, CHAIN) == []
    assert idx.acceptors_in("chr1", Strand.MINUS, CHAIN) == []
