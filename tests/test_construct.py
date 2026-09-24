import pytest

from novex.chains import Chain, GInterval, Strand
from novex.construct import (
    RejectReason,
    add_utr3,
    build_all_downstream,
    build_all_upstream,
    build_cds_downstream,
    build_cds_upstream,
    build_chains_downstream,
    build_chains_upstream,
    build_exons_downstream,
    check_orf,
    find_stop_offset,
    junction_offset,
    trim_to_stop,
)
from novex.junctions import Junction, JunctionIndex
from novex.transcripts import CdsIndex, RefTranscript, OrfChain

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
    return OrfChain(id=uid, chrom="chr1", strand=strand, cds=Chain(cds))


def reference(cds, strand=Strand.PLUS, tid="t1", exons=((1, 200),)):
    """Reference whose exons span the whole test contig by default, so the query ORF is
    transcribed by it -- build_all_upstream rejects pairs where it is not."""
    return RefTranscript(id=tid, chrom="chr1", strand=strand, exons=Chain(exons), cds=Chain(cds))


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


# --- build_cds_upstream ----------------------------------------------------------------

def test_build_cds_plus_strand():
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    cds = build_cds_upstream(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])


def test_build_cds_minus_strand_swaps_the_roles():
    # on '-', the reference CDS is genomically left and the upstream chain right
    u = upstream([(100, 108)], strand=Strand.MINUS)
    r = reference([(60, 71)], strand=Strand.MINUS)
    cds = build_cds_upstream(u, r, junction(72, 99, Strand.MINUS))
    assert cds == Chain([(60, 71), (100, 108)])


def test_build_cds_clips_at_the_junction():
    u = upstream([(11, 30)])          # donor exon base is 19, so 20-30 is discarded
    r = reference([(35, 51)])         # acceptor exon base is 40, so 35-39 is discarded
    cds = build_cds_upstream(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])


def test_build_cds_returns_none_when_a_boundary_is_outside_its_chain():
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    assert build_cds_upstream(u, r, junction(25, 39)) is None   # donor exon base 24 not in upstream
    assert build_cds_upstream(u, r, junction(20, 60)) is None   # acceptor exon base 61 past the reference


def test_build_cds_rejects_mixed_strands():
    u = upstream([(11, 19)], strand=Strand.PLUS)
    r = reference([(40, 51)], strand=Strand.MINUS)
    with pytest.raises(ValueError):
        build_cds_upstream(u, r, junction(20, 39))


# --- build_all_upstream ----------------------------------------------------------------

def plus_setup(ref_seq="GCCGCCGCCTAA"):
    """Upstream 11-19 (ATG AAA AAA) spliced across intron 20-39 into reference 40-51."""
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), ref_seq)
    u = upstream([(11, 19)])
    r = reference([(40, 51)])
    j = junction(20, 39)
    return g, [u], JunctionIndex([j]), CdsIndex([r])


def test_build_all_builds_a_valid_construct():
    g, ups, juncs, refs = plus_setup()
    constructs, rejections = build_all_upstream(g.fetch, ups, juncs, refs)

    assert rejections == []
    (c,) = constructs
    assert c.id == "ust_0"
    assert c.query_id == "u1"
    assert c.reference.id == "t1"
    assert c.cds == Chain([(11, 19), (40, 51)])
    assert c.chrom == "chr1" and c.strand == Strand.PLUS
    assert g.fetch(c.chrom, c.cds, c.strand) == "ATGAAAAAAGCCGCCGCCTAA"


def test_build_all_reports_the_rejection_reason():
    g, ups, juncs, refs = plus_setup(ref_seq="GCCGCCGCCGCC")  # no terminal stop
    constructs, rejections = build_all_upstream(g.fetch, ups, juncs, refs)

    assert constructs == []
    (rej,) = rejections
    assert rej.reason == RejectReason.NO_STOP
    assert (rej.query_id, rej.reference_id) == ("u1", "t1")
    assert rej.junction.intron == GInterval(20, 39)


def test_build_all_flags_duplicate_chains():
    # two junctions with the same intron produce the same CDS chain
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    juncs = JunctionIndex([junction(20, 39, name="J1"), junction(20, 39, name="J2")])
    constructs, rejections = build_all_upstream(
        g.fetch, [upstream([(11, 19)])], juncs, CdsIndex([reference([(40, 51)])])
    )

    assert len(constructs) == 1
    assert [r.reason for r in rejections] == [RejectReason.DUPLICATE]


def test_build_all_minus_strand():
    g = FakeGenome()
    g.put(range(108, 99, -1), "ATGAAAAAA", Strand.MINUS)     # upstream, read 108 -> 100
    g.put(range(71, 59, -1), "GCCGCCGCCTAA", Strand.MINUS)   # reference, read 71 -> 60

    constructs, rejections = build_all_upstream(
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
    constructs, rejections = build_all_upstream(
        g.fetch, [upstream([(11, 19)])], JunctionIndex([junction(20, 39)]), refs
    )

    assert sorted(c.reference.id for c in constructs) + [r.reference_id for r in rejections] == [
        "t1",
        "t2",
    ]


def test_build_all_skips_junctions_outside_the_upstream_chain():
    g, ups, _, refs = plus_setup()
    far = JunctionIndex([junction(80, 99)])   # donor exon base 79 is not in 11-19
    assert build_all_upstream(g.fetch, ups, far, refs) == ([], [])


def test_build_all_ids_are_sequential_over_kept_constructs():
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    ups = [upstream([(11, 19)], uid="u1"), upstream([(11, 19)], uid="u2")]
    constructs, rejections = build_all_upstream(
        g.fetch, ups, JunctionIndex([junction(20, 39)]), CdsIndex([reference([(40, 51)])])
    )
    # u2 produces the same chain as u1, so it is a duplicate
    assert [c.id for c in constructs] == ["ust_0"]
    assert [r.reason for r in rejections] == [RejectReason.DUPLICATE]


# --- UTRs ---------------------------------------------------------------------

def reference_with_utrs(cds, exons, strand=Strand.PLUS, tid="t1"):
    return RefTranscript(id=tid, chrom="chr1", strand=strand, exons=Chain(exons), cds=Chain(cds))


def test_build_chains_carries_the_reference_3prime_utr():
    u = upstream([(11, 19)])
    r = reference_with_utrs(cds=[(40, 51)], exons=[(1, 60)])
    cds, exons = build_chains_upstream(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])
    assert exons == Chain([(1, 19), (40, 60)])   # 52-60 is the 3' UTR, 1-10 the 5' UTR


def test_build_chains_carries_the_reference_5prime_utr():
    # the ORF sits inside the reference's own 5' UTR, so 1-10 comes across
    u = upstream([(11, 19)])
    r = reference_with_utrs(cds=[(40, 51)], exons=[(1, 60)])
    cds, exons = build_chains_upstream(u, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 51)])
    assert exons == Chain([(1, 19), (40, 60)])    # utr5 merged with the ORF's first exon


def test_build_chains_keeps_a_spliced_5prime_utr_separate():
    u = upstream([(11, 19)])
    r = reference_with_utrs(cds=[(40, 51)], exons=[(1, 5), (11, 60)])
    _, exons = build_chains_upstream(u, r, junction(20, 39))
    assert exons == Chain([(1, 5), (11, 19), (40, 60)])


@pytest.mark.parametrize(
    "exons, why",
    [
        ([(30, 60)], "reference starts downstream of the ORF"),
        ([(1, 5), (30, 60)], "the ORF is intronic for this isoform"),
    ],
)
def test_build_all_rejects_references_that_do_not_transcribe_the_orf(exons, why):
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    r = reference_with_utrs(cds=[(40, 51)], exons=exons)
    constructs, rejections = build_all_upstream(
        g.fetch, [upstream([(11, 19)])], JunctionIndex([junction(20, 39)]), CdsIndex([r])
    )
    assert constructs == [], why
    assert [x.reason for x in rejections] == [RejectReason.NOT_TRANSCRIBED], why


def test_build_chains_minus_strand_utrs():
    # transcription runs 120 -> 50; 5' UTR is genomically above the ORF
    u = upstream([(100, 108)], strand=Strand.MINUS)
    r = reference_with_utrs(cds=[(60, 71)], exons=[(50, 120)], strand=Strand.MINUS)
    cds, exons = build_chains_upstream(u, r, junction(72, 99, Strand.MINUS))
    assert cds == Chain([(60, 71), (100, 108)])
    assert exons == Chain([(50, 71), (100, 120)])  # 50-59 is 3' UTR, 109-120 is 5' UTR


def test_build_all_stores_the_exon_chain():
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 52), "GCCGCCGCCTAA")
    r = reference_with_utrs(cds=[(40, 51)], exons=[(1, 60)])
    constructs, _ = build_all_upstream(
        g.fetch, [upstream([(11, 19)])], JunctionIndex([junction(20, 39)]), CdsIndex([r])
    )
    (c,) = constructs
    assert c.cds == Chain([(11, 19), (40, 51)])
    assert c.exons == Chain([(1, 19), (40, 60)])


def test_build_chains_omits_the_5prime_utr_when_the_orf_starts_at_the_reference_tss():
    u = upstream([(11, 19)])
    r = reference_with_utrs(cds=[(40, 51)], exons=[(11, 60)])   # transcript starts at the ORF
    _, exons = build_chains_upstream(u, r, junction(20, 39))
    assert exons == Chain([(11, 19), (40, 60)])


# =============================================================================
# DOWNSTREAM (cli -d down): reference ATG ---> junction ---> query ORF
# =============================================================================

def dref(cds, exons, strand=Strand.PLUS, tid="t1"):
    """A reference whose exons are given explicitly (it must contain the query ORF)."""
    return RefTranscript(id=tid, chrom="chr1", strand=strand, exons=Chain(exons), cds=Chain(cds))


# --- build_cds_downstream / build_exons_downstream ----------------------------

def test_build_cds_downstream_plus_strand():
    # reference 11-19 is genomically left, query 40-60 right
    q = upstream([(40, 60)], uid="q1")
    r = dref(cds=[(11, 19)], exons=[(1, 90)])
    assert build_cds_downstream(q, r, junction(20, 39)) == Chain([(11, 19), (40, 60)])


def test_build_cds_downstream_minus_strand_swaps_the_roles():
    # on '-', transcription runs right to left: the reference is genomically right
    q = upstream([(40, 60)], strand=Strand.MINUS, uid="q1")
    r = dref(cds=[(100, 108)], exons=[(1, 200)], strand=Strand.MINUS)
    assert build_cds_downstream(q, r, junction(61, 99, Strand.MINUS)) == Chain([(40, 60), (100, 108)])


def test_build_cds_downstream_clips_at_the_junction():
    q = upstream([(30, 60)], uid="q1")     # acceptor exon base is 40, so 30-39 is dropped
    r = dref(cds=[(11, 25)], exons=[(1, 90)])  # donor exon base is 19, so 20-25 is dropped
    assert build_cds_downstream(q, r, junction(20, 39)) == Chain([(11, 19), (40, 60)])


def test_build_exons_downstream_carries_the_reference_5prime_utr():
    # the 5' UTR arrives through the splice: reference.exons is the left chain, unclipped at its 5' end
    q = upstream([(40, 60)], uid="q1")
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 90)])
    assert build_exons_downstream(q, r, junction(20, 39)) == Chain([(1, 19), (40, 60)])


def test_build_chains_downstream_returns_candidate_and_exons():
    q = upstream([(40, 60)], uid="q1")
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 90)])
    cds, exons = build_chains_downstream(q, r, junction(20, 39))
    assert cds == Chain([(11, 19), (40, 60)])     # candidate: runs to the end of the query chain
    assert exons == Chain([(1, 19), (40, 90)])    # 61-90 transferred from the reference


# --- junction_offset ----------------------------------------------------------

def test_junction_offset_counts_reference_bases():
    r = dref(cds=[(11, 19)], exons=[(1, 90)])
    assert junction_offset(r, junction(20, 39)) == 9


def test_junction_offset_ignores_reference_cds_past_the_donor():
    r = dref(cds=[(11, 25)], exons=[(1, 90)])     # only 11-19 is upstream of the donor
    assert junction_offset(r, junction(20, 39)) == 9


def test_junction_offset_minus_strand():
    r = dref(cds=[(100, 108)], exons=[(1, 200)], strand=Strand.MINUS)
    assert junction_offset(r, junction(61, 99, Strand.MINUS)) == 9


# --- find_stop_offset ---------------------------------------------------------

@pytest.mark.parametrize(
    "seq, expected",
    [
        ("ATGAAATAA", 9),        # stop is the third codon
        ("TAAAAAGCC", 3),        # stop first
        ("ATGAAAGCC", None),     # no stop at all
        ("ATGAAAGC", None),      # trailing partial codon cannot be a stop
        ("", None),
    ],
)
def test_find_stop_offset(seq, expected):
    assert find_stop_offset(seq) == expected


def test_find_stop_offset_takes_the_first_stop():
    assert find_stop_offset("ATGTAAGCCTGA") == 6


# --- trim_to_stop -------------------------------------------------------------

CANDIDATE = Chain([(11, 19), (40, 60)])   # 9 reference bases + 21 query bases


def test_trim_to_stop_cuts_the_chain_at_the_stop():
    seq = "ATG" + "AAA" * 2 + "GCC" * 2 + "TAA" + "GGG" * 2   # stop at offset 18
    assert trim_to_stop(CANDIDATE, Strand.PLUS, seq, 9) == Chain([(11, 19), (40, 48)])


def test_trim_to_stop_reports_no_stop():
    assert trim_to_stop(CANDIDATE, Strand.PLUS, "GCC" * 10, 9) == RejectReason.NO_STOP


def test_trim_to_stop_reports_a_stop_inside_the_reference_part():
    # an in-frame stop before the junction means the REFERENCE's own CDS carries one
    seq = "ATGTAA" + "GCC" * 8
    assert trim_to_stop(CANDIDATE, Strand.PLUS, seq, 9) == RejectReason.PTC


def test_trim_to_stop_at_the_junction_boundary_is_ptc():
    # the stop ends exactly where the reference part does
    seq = "ATGAAATAA" + "GCC" * 7
    assert trim_to_stop(CANDIDATE, Strand.PLUS, seq, 9) == RejectReason.PTC


def test_trim_to_stop_keeps_the_whole_chain_when_the_stop_is_last():
    seq = "GCC" * 9 + "TAA"
    assert trim_to_stop(CANDIDATE, Strand.PLUS, seq, 9) == CANDIDATE


# --- add_utr3 -----------------------------------------------------------------

def test_add_utr3_transfers_the_reference_remainder():
    cds = Chain([(11, 19), (40, 60)])
    exons = Chain([(1, 19), (40, 60)])
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 90)])
    assert add_utr3(exons, cds, r) == Chain([(1, 19), (40, 90)])


def test_add_utr3_keeps_a_spliced_reference_3prime_utr_separate():
    cds = Chain([(11, 19), (40, 60)])
    exons = Chain([(1, 19), (40, 60)])
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 60), (70, 90)])
    assert add_utr3(exons, cds, r) == Chain([(1, 19), (40, 60), (70, 90)])


def test_add_utr3_adds_nothing_when_the_reference_ends_there():
    cds = Chain([(11, 19), (40, 60)])
    exons = Chain([(1, 19), (40, 60)])
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 60)])
    assert add_utr3(exons, cds, r) == exons


def test_add_utr3_minus_strand():
    # on '-' the 3' UTR is genomically to the LEFT of the query chain
    cds = Chain([(40, 60), (100, 108)])
    exons = Chain([(40, 60), (100, 120)])
    r = dref(cds=[(100, 108)], exons=[(10, 60), (100, 120)], strand=Strand.MINUS)
    assert add_utr3(exons, cds, r) == Chain([(10, 60), (100, 120)])


# --- build_all_downstream -----------------------------------------------------

def down_setup(query_seq="GCCGCCTAAGGGGGGGGGGGG", ref_exons=((1, 19), (40, 90))):
    """Reference ATG AAA AAA at 11-19, intron 20-39, query ORF at 40-60."""
    g = FakeGenome().put(range(11, 20), "ATGAAAAAA").put(range(40, 61), query_seq)
    q = upstream([(40, 60)], uid="q1")
    r = dref(cds=[(11, 19)], exons=list(ref_exons))
    return g, [q], JunctionIndex([junction(20, 39)]), CdsIndex([r])


def test_build_all_downstream_builds_a_valid_construct():
    g, qs, juncs, refs = down_setup()
    constructs, rejections = build_all_downstream(g.fetch, qs, juncs, refs)

    assert rejections == []
    (c,) = constructs
    assert c.id == "dst_0"
    assert c.query_id == "q1"
    assert c.reference.id == "t1"
    assert c.cds == Chain([(11, 19), (40, 48)])      # trimmed at the stop
    assert c.exons == Chain([(1, 19), (40, 90)])     # both UTRs present
    assert g.fetch(c.chrom, c.cds, c.strand) == "ATGAAAAAAGCCGCCTAA"


def test_build_all_downstream_reports_no_stop():
    g, qs, juncs, refs = down_setup(query_seq="GCC" * 7)
    constructs, rejections = build_all_downstream(g.fetch, qs, juncs, refs)
    assert constructs == []
    assert [r.reason for r in rejections] == [RejectReason.NO_STOP]


def test_build_all_downstream_rejects_a_reference_that_does_not_transcribe_the_orf():
    g, qs, juncs, _ = down_setup()
    r = dref(cds=[(11, 19)], exons=[(1, 19), (40, 50)])   # stops short of the query chain
    constructs, rejections = build_all_downstream(g.fetch, qs, juncs, CdsIndex([r]))
    assert constructs == []
    assert [x.reason for x in rejections] == [RejectReason.NOT_TRANSCRIBED]


def test_build_all_downstream_flags_duplicate_chains():
    g, qs, _, refs = down_setup()
    juncs = JunctionIndex([junction(20, 39, name="J1"), junction(20, 39, name="J2")])
    constructs, rejections = build_all_downstream(g.fetch, qs, juncs, refs)
    assert [c.id for c in constructs] == ["dst_0"]
    assert [r.reason for r in rejections] == [RejectReason.DUPLICATE]


def test_build_all_downstream_minus_strand():
    # transcription runs 108 -> 100, then 60 -> 40
    g = FakeGenome()
    g.put(range(108, 99, -1), "ATGAAAAAA", Strand.MINUS)
    g.put(range(60, 39, -1), "GCCGCCTAAGGGGGGGGGGGG", Strand.MINUS)

    q = upstream([(40, 60)], strand=Strand.MINUS, uid="q1")
    r = dref(cds=[(100, 108)], exons=[(10, 60), (100, 120)], strand=Strand.MINUS)
    constructs, rejections = build_all_downstream(
        g.fetch, [q], JunctionIndex([junction(61, 99, Strand.MINUS)]), CdsIndex([r])
    )

    assert rejections == []
    (c,) = constructs
    assert c.cds == Chain([(52, 60), (100, 108)])     # trimmed at the stop, reading leftwards
    assert g.fetch(c.chrom, c.cds, c.strand) == "ATGAAAAAAGCCGCCTAA"
