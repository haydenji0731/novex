import pytest

from novex.chains import Chain, GInterval, Strand
from novex.construct import RejectReason, build_all, build_cds, check_orf
from novex.junctions import Junction, JunctionIndex
from novex.transcripts import CdsIndex, RefTranscript, UpstreamChain

COMPLEMENT = str.maketrans("ACGT", "TGCA")


class FakeGenome:
    """One contig of 'A's, with bases placed by transcript position (see test_stop_codons)."""

    def __init__(self, size: int = 200):
        self.bases = ["A"] * (size + 1)  # 1-based; index 0 unused

    def put(self, positions, seq, strand=Strand.PLUS):
        for pos, base in zip(positions, seq, strict=True):
            self.bases[pos] = base if strand == Strand.PLUS else base.translate(COMPLEMENT)
        return self

    def fetch(self, chrom: str, chain: Chain, strand: Strand) -> str:
        seq = "".join("".join(self.bases[x.start : x.end + 1]) for x in chain.intervals)
        return seq.translate(COMPLEMENT)[::-1] if strand == Strand.MINUS else seq


def upstream(cds, strand=Strand.PLUS, uid="u1"):
    return UpstreamChain(id=uid, chrom="chr1", strand=strand, cds=Chain(cds))


def reference(cds, strand=Strand.PLUS, tid="t1"):
    return RefTranscript(id=tid, chrom="chr1", strand=strand, exons=Chain(cds), cds=Chain(cds))


def junction(start, end, strand=Strand.PLUS, name=None):
    return Junction(chrom="chr1", strand=strand, intron=GInterval(start, end), name=name)


# --- check_orf ----------------------------------------------------------------

def test_check_orf_accepts_a_valid_orf():
    assert check_orf("ATGAAAGCCTAA") is None


@pytest.mark.parametrize(
    "seq, reason",
    [
        ("ATGAAAGCCTA", RejectReason.FRAME),        # 11 bases
        ("ATGAAAGCCGCC", RejectReason.NO_STOP),     # no terminal stop
        ("ATGTAAGCCTAA", RejectReason.PTC),         # stop at codon 2
    ],
)
def test_check_orf_rejects(seq, reason):
    assert check_orf(seq) == reason


def test_check_orf_allows_ambiguous_codons():
    # an N-containing codon translates to X, which is not a stop
    assert check_orf("ATGNNNGCCTAA") is None


def test_check_orf_frame_is_checked_before_stop():
    assert check_orf("ATGAAAGCCTAAA") == RejectReason.FRAME


# --- build_cds ----------------------------------------------------------------

def test_build_cds_plus_strand():
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    cds = build_cds(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])


def test_build_cds_minus_strand_swaps_the_roles():
    # on '-', the reference CDS is genomically left and the upstream chain right
    u = upstream([(100, 108)], strand=Strand.MINUS)
    r = reference([(60, 71)], strand=Strand.MINUS)
    cds = build_cds(u, r, junction(72, 99, Strand.MINUS))
    assert cds == Chain([(60, 71), (100, 108)])


def test_build_cds_clips_at_the_junction():
    u = upstream([(11, 30)])          # donor exon base is 19, so 20-30 is discarded
    r = reference([(35, 51)])         # acceptor exon base is 40, so 35-39 is discarded
    cds = build_cds(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])


def test_build_cds_returns_none_when_a_boundary_is_outside_its_chain():
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    assert build_cds(u, r, junction(25, 39)) is None   # donor exon base 24 not in upstream
    assert build_cds(u, r, junction(20, 60)) is None   # acceptor exon base 61 past the reference


def test_build_cds_rejects_mixed_strands():
    u = upstream([(11, 19)], strand=Strand.PLUS)
    r = reference([(40, 51)], strand=Strand.MINUS)
    with pytest.raises(ValueError):
        build_cds(u, r, junction(20, 39))


# --- build_all ----------------------------------------------------------------

def plus_setup(ref_seq="GCCGCCGCCTAA"):
    """Upstream 11-19 (ATG AAA AAA) spliced across intron 20-39 into reference 40-51."""
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), ref_seq)
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    j = junction(20, 39)
    return g, [u], JunctionIndex([j]), CdsIndex([r])


def test_build_all_builds_a_valid_construct():
    g, ups, juncs, refs = plus_setup()
    constructs, rejections = build_all(g.fetch, ups, juncs, refs)

    assert rejections == []
    (c,) = constructs
    assert c.id == "cst_0"
    assert c.upstream_id == "u1"
    assert c.reference.id == "t1"
    assert c.cds == Chain([(11, 19), (40, 51)])
    assert c.chrom == "chr1" and c.strand == Strand.PLUS
    assert g.fetch(c.chrom, c.cds, c.strand) == "ATGAAAAAAGCCGCCGCCTAA"


def test_build_all_reports_the_rejection_reason():
    g, ups, juncs, refs = plus_setup(ref_seq="GCCGCCGCCGCC")  # no terminal stop
    constructs, rejections = build_all(g.fetch, ups, juncs, refs)

    assert constructs == []
    (rej,) = rejections
    assert rej.reason == RejectReason.NO_STOP
    assert (rej.upstream_id, rej.reference_id) == ("u1", "t1")
    assert rej.junction.intron == GInterval(20, 39)


def test_build_all_flags_duplicate_chains():
    # two junctions with the same intron produce the same CDS chain
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    juncs = JunctionIndex([junction(20, 39, name="J1"), junction(20, 39, name="J2")])
    constructs, rejections = build_all(
        g.fetch, [upstream([(11, 19)])], juncs, CdsIndex([reference([(40, 51)])])
    )

    assert len(constructs) == 1
    assert [r.reason for r in rejections] == [RejectReason.DUPLICATE]


def test_build_all_minus_strand():
    g = FakeGenome()
    g.put(range(108, 99, -1), "ATGAAAAAA", Strand.MINUS)     # upstream, read 108 -> 100
    g.put(range(71, 59, -1), "GCCGCCGCCTAA", Strand.MINUS)   # reference, read 71 -> 60

    constructs, rejections = build_all(
        g.fetch,
        [upstream([(100, 108)], strand=Strand.MINUS)],
        JunctionIndex([junction(72, 99, Strand.MINUS)]),
        CdsIndex([reference([(60, 71)], strand=Strand.MINUS)]),
    )

    assert rejections == []
    (c,) = constructs
    assert c.cds == Chain([(60, 71), (100, 108)])
    assert g.fetch(c.chrom, c.cds, c.strand) == "ATGAAAAAAGCCGCCGCCTAA"


def test_build_all_pairs_every_upstream_with_every_reference():
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    refs = CdsIndex([reference([(40, 51)], tid="t1"), reference([(40, 54)], tid="t2")])
    constructs, rejections = build_all(
        g.fetch, [upstream([(11, 19)])], JunctionIndex([junction(20, 39)]), refs
    )

    assert sorted(c.reference.id for c in constructs) + [r.reference_id for r in rejections] == [
        "t1",
        "t2",
    ]


def test_build_all_skips_junctions_outside_the_upstream_chain():
    g, ups, _, refs = plus_setup()
    far = JunctionIndex([junction(80, 99)])   # donor exon base 79 is not in 11-19
    assert build_all(g.fetch, ups, far, refs) == ([], [])


def test_build_all_ids_are_sequential_over_kept_constructs():
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    ups = [upstream([(11, 19)], uid="u1"), upstream([(11, 19)], uid="u2")]
    constructs, rejections = build_all(
        g.fetch, ups, JunctionIndex([junction(20, 39)]), CdsIndex([reference([(40, 51)])])
    )
    # u2 produces the same chain as u1, so it is a duplicate
    assert [c.id for c in constructs] == ["cst_0"]
    assert [r.reason for r in rejections] == [RejectReason.DUPLICATE]
