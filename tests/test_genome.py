import pathlib

import pytest

from novex.chains import Chain, GInterval, Strand
from novex.genome import (
    CANONICAL_MOTIFS,
    fetch_chain_seq,
    filter_juncs_by_motif,
    get_junc_dinucs,
    motif_ok,
    parse_motifs,
    parse_starts,
    first_codon,
    start_ok,
    filter_upstream_by_start,
    load_genome,
)
from novex.junctions import Junction
from novex.transcripts import OrfChain

COMPLEMENT = str.maketrans("ACGT", "TGCA")

# 1-based positions:      1234567890
CONTIG = (
    "ACGTACGTAC"      #  1-10
    "GTAAAAAAAG"      # 11-20  intron 11-20: starts GT, ends AG
    "TTTTTTTTTT"      # 21-30
    "GCCCCCCCAG"      # 31-40  intron 31-40: starts GC, ends AG
    "AAAAAAAAAA"      # 41-50
)


@pytest.fixture
def fa(tmp_path):
    p = tmp_path / "t.fa"
    p.write_text(">chr1\n" + "\n".join(CONTIG[i : i + 20] for i in range(0, len(CONTIG), 20)) + "\n")
    return load_genome(p)


def jn(start, end, strand=Strand.PLUS):
    return Junction(chrom="chr1", strand=strand, intron=GInterval(start, end))


# --- fetch_chain_seq -------------------------------------------------------------

def test_fetch_chain_seq_is_1_based_inclusive(fa):
    assert fetch_chain_seq(fa, "chr1", Chain([(1, 4)]), Strand.PLUS) == "ACGT"
    assert fetch_chain_seq(fa, "chr1", Chain([(11, 12)]), Strand.PLUS) == "GT"


def test_fetch_chain_seq_splices_intervals(fa):
    assert fetch_chain_seq(fa, "chr1", Chain([(1, 4), (11, 12)]), Strand.PLUS) == "ACGTGT"


def test_fetch_chain_seq_reverse_complements_on_minus(fa):
    plus = fetch_chain_seq(fa, "chr1", Chain([(1, 4), (11, 12)]), Strand.PLUS)
    minus = fetch_chain_seq(fa, "chr1", Chain([(1, 4), (11, 12)]), Strand.MINUS)
    assert minus == plus.translate(COMPLEMENT)[::-1]


def test_fetch_chain_seq_uppercases(tmp_path):
    p = tmp_path / "soft.fa"
    p.write_text(">chr1\nacgtacgtac\n")
    assert fetch_chain_seq(load_genome(p), "chr1", Chain([(1, 4)]), Strand.PLUS) == "ACGT"


def test_fetch_chain_seq_rejects_unknown_strand(fa):
    with pytest.raises(ValueError):
        fetch_chain_seq(fa, "chr1", Chain([(1, 4)]), Strand.UNKNOWN)


def test_fetch_chain_seq_matches_stop_codons_fetch_signature(fa):
    from functools import partial

    fetch = partial(fetch_chain_seq, fa)
    assert fetch("chr1", Chain([(1, 4)]), Strand.PLUS) == "ACGT"


# --- get_junc_dinucs ---------------------------------------------------------

def test_get_junc_dinucs_plus(fa):
    assert get_junc_dinucs(fa, jn(11, 20)) == ("GT", "AG")


def test_get_junc_dinucs_minus_reads_the_other_end(fa):
    # on '-', the donor side is the intron's genomic end; revcomp of "AG" is "CT"
    assert get_junc_dinucs(fa, jn(11, 20, Strand.MINUS)) == ("CT", "AC")


def test_get_junc_dinucs_gc_ag(fa):
    assert get_junc_dinucs(fa, jn(31, 40)) == ("GC", "AG")


# --- motif_ok / filter_juncs_by_motif ----------------------------------------------

def test_motif_ok_accepts_gt_ag_and_gc_ag(fa):
    assert motif_ok(fa, jn(11, 20), CANONICAL_MOTIFS)
    assert motif_ok(fa, jn(31, 40), CANONICAL_MOTIFS)


def test_motif_ok_rejects_noncanonical(fa):
    assert not motif_ok(fa, jn(21, 30), CANONICAL_MOTIFS)          # TT..TT
    assert not motif_ok(fa, jn(11, 20, Strand.MINUS), CANONICAL_MOTIFS)


def test_motif_ok_honors_a_custom_allowed_set(fa):
    assert not motif_ok(fa, jn(31, 40), allowed=frozenset({("GT", "AG")}))


def test_filter_juncs_by_motif_keeps_only_canonical(fa):
    juncs = [jn(11, 20), jn(21, 30), jn(31, 40)]
    assert [j.intron.start for j in filter_juncs_by_motif(fa, juncs)] == [11, 31]


def test_canonical_motifs_contents():
    assert CANONICAL_MOTIFS == frozenset({("GT", "AG"), ("GC", "AG")})


# --- against real data, when present ------------------------------------------

CHR20 = pathlib.Path(__file__).parent.parent / "data" / "chr20.fa"


@pytest.mark.skipif(not CHR20.exists(), reason="data/chr20.fa not present")
def test_fetch_chain_seq_on_real_contig():
    fa = load_genome(CHR20)
    chain = Chain([(1_000_001, 1_000_005), (1_000_011, 1_000_015)])
    spliced = fetch_chain_seq(fa, "chr20", chain, Strand.PLUS)

    parts = "".join(
        fetch_chain_seq(fa, "chr20", Chain([(x.start, x.end)]), Strand.PLUS) for x in chain.intervals
    )
    assert spliced == parts
    assert len(spliced) == len(chain) == 10
    assert fetch_chain_seq(fa, "chr20", chain, Strand.MINUS) == spliced.translate(COMPLEMENT)[::-1]


# --- parse_motifs -------------------------------------------------------------

@pytest.mark.parametrize(
    "spec, expected",
    [
        ("canonical", CANONICAL_MOTIFS),
        ("CANONICAL", CANONICAL_MOTIFS),
        ("gtag", frozenset({("GT", "AG")})),
        ("u12", CANONICAL_MOTIFS | frozenset({("AT", "AC")})),
        ("any", None),
        ("GT-AG", frozenset({("GT", "AG")})),
        ("gt-ag,at-ac", frozenset({("GT", "AG"), ("AT", "AC")})),
    ],
)
def test_parse_motifs(spec, expected):
    assert parse_motifs(spec) == expected


@pytest.mark.parametrize("spec", ["GT_AG", "GTT-AG", "", "nonsense"])
def test_parse_motifs_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        parse_motifs(spec)


def test_motifs_any_skips_screening(fa):
    juncs = [jn(11, 20), jn(21, 30), jn(31, 40)]
    assert filter_juncs_by_motif(fa, juncs, allowed=None) == juncs
    assert motif_ok(fa, jn(21, 30), allowed=None)


# --- start codon screening ----------------------------------------------------

def up(cds, strand=Strand.PLUS, uid="u1"):
    return OrfChain(id=uid, chrom="chr1", strand=strand, cds=Chain(cds))


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("atg", frozenset({"ATG"})),
        ("ATG", frozenset({"ATG"})),
        ("any", None),
        ("near-cognate", frozenset({"ATG", "CTG", "GTG", "TTG", "ACG"})),
        ("atg,ctg", frozenset({"ATG", "CTG"})),
    ],
)
def test_parse_starts(spec, expected):
    assert parse_starts(spec) == expected


@pytest.mark.parametrize("spec", ["AT", "ATGG", "AXG", "nonsense"])
def test_parse_starts_rejects_bad_input(spec):
    with pytest.raises(ValueError):
        parse_starts(spec)


def test_first_codon_reads_in_transcript_direction(fa):
    # CONTIG starts ACGTACGTAC...; on '-' the first codon of (1,3) is revcomp("ACG")
    assert first_codon(fa, "chr1", Chain([(1, 3)]), Strand.PLUS) == "ACG"
    assert first_codon(fa, "chr1", Chain([(1, 3)]), Strand.MINUS) == "CGT"


def test_start_ok(fa):
    # bases 31-33 are "GCC"; 11-13 are "GTA"
    assert start_ok(fa, up([(31, 33)]), frozenset({"GCC"}))
    assert not start_ok(fa, up([(31, 33)]), frozenset({"ATG"}))
    assert start_ok(fa, up([(31, 33)]), None)  # --starts any


def test_filter_upstream_by_start_counts(fa):
    chains = [up([(31, 33)], uid="keep"), up([(11, 13)], uid="drop")]
    kept = filter_upstream_by_start(fa, chains, frozenset({"GCC"}))
    assert [u.id for u in kept] == ["keep"]
    assert filter_upstream_by_start(fa, chains, None) == chains
