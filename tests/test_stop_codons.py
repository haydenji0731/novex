import pytest

from novex.chains import Chain, Strand
from novex.stop_codons import detect_stop_convention, extend_cds, last_codon, normalize, normalize_all
from novex.transcripts import RefTranscript, StopConvention

COMPLEMENT = str.maketrans("ACGT", "TGCA")


class FakeGenome:
    """A single contig of 'A's, with bases placed by transcript position."""

    def __init__(self, size: int = 100):
        self.bases = ["A"] * (size + 1)  # 1-based; index 0 unused

    def put(self, positions, seq, strand=Strand.PLUS):
        """Write `seq` (5'->3') at `positions`, given in transcript order."""
        for pos, base in zip(positions, seq, strict=True):
            self.bases[pos] = base if strand == Strand.PLUS else base.translate(COMPLEMENT)
        return self

    def fetch(self, chrom: str, chain: Chain, strand: Strand) -> str:
        seq = "".join("".join(self.bases[iv.start : iv.end + 1]) for iv in chain.intervals)
        if strand == Strand.MINUS:
            seq = seq.translate(COMPLEMENT)[::-1]
        return seq


def tx(cds, exons=None, strand=Strand.PLUS, tid="t1"):
    return RefTranscript(
        id=tid,
        chrom="chr1",
        strand=strand,
        exons=Chain(exons or cds),
        cds=Chain(cds),
    )


# --- last_codon --------------------------------------------------------------

def test_last_codon_plus():
    g = FakeGenome().put([17, 18, 19], "TAA")
    assert last_codon(g.fetch, tx([(11, 19)])) == "TAA"


def test_last_codon_minus_is_read_in_transcript_direction():
    # on '-', the last coding bases are the genomically first ones
    g = FakeGenome().put([13, 12, 11], "TAG", strand=Strand.MINUS)
    assert last_codon(g.fetch, tx([(11, 19)], strand=Strand.MINUS)) == "TAG"


def test_last_codon_uppercases_soft_masked_sequence():
    g = FakeGenome().put([17, 18, 19], "tga")
    assert last_codon(g.fetch, tx([(11, 19)])) == "TGA"


def test_last_codon_rejects_short_cds():
    g = FakeGenome()
    with pytest.raises(ValueError):
        last_codon(g.fetch, tx([(11, 12)]))


# --- extend_cds ------------------------------------------------------------

def test_extended_cds_within_one_exon():
    t = tx([(11, 19)], exons=[(11, 40)])
    assert extend_cds(t) == Chain([(11, 22)])


def test_extended_cds_crosses_an_intron():
    # CDS ends at 18; the next 3 transcript bases are 19, 31, 32
    t = tx([(11, 18)], exons=[(11, 19), (31, 40)])
    assert extend_cds(t) == Chain([(11, 19), (31, 32)])


def test_extended_cds_starts_a_new_interval_at_an_exon_boundary():
    t = tx([(11, 19)], exons=[(11, 19), (31, 40)])
    assert extend_cds(t) == Chain([(11, 19), (31, 33)])


def test_extended_cds_minus_strand_crosses_an_intron():
    # on '-', transcription runs 40->31 then 19->11; CDS ends at 32
    t = tx([(32, 40)], exons=[(11, 19), (31, 40)], strand=Strand.MINUS)
    assert extend_cds(t) == Chain([(18, 19), (31, 40)])


def test_extended_cds_returns_none_when_exons_run_out():
    t = tx([(11, 19)], exons=[(11, 20)])  # only 1 base past the CDS
    assert extend_cds(t) is None


# --- detect ------------------------------------------------------------------

def test_detect_included():
    g = FakeGenome().put([17, 18, 19], "TAA")
    assert detect_stop_convention(g.fetch, tx([(11, 19)], exons=[(11, 40)])) == StopConvention.INCLUDED


def test_detect_excluded():
    g = FakeGenome().put([17, 18, 19], "AAA").put([20, 21, 22], "TGA")
    assert detect_stop_convention(g.fetch, tx([(11, 19)], exons=[(11, 40)])) == StopConvention.EXCLUDED


def test_detect_excluded_across_an_intron():
    # this is what the old `cds.end + 3` got wrong: base 20 is intronic
    g = FakeGenome().put([17, 18, 19], "AAA").put([31, 32, 33], "TAG").put([20, 21, 22], "TAA")
    t = tx([(11, 19)], exons=[(11, 19), (31, 40)])
    assert detect_stop_convention(g.fetch, t) == StopConvention.EXCLUDED


def test_detect_missing_when_no_stop_either_way():
    g = FakeGenome().put([17, 18, 19], "AAA").put([20, 21, 22], "CCC")
    assert detect_stop_convention(g.fetch, tx([(11, 19)], exons=[(11, 40)])) == StopConvention.MISSING


def test_detect_missing_when_exons_run_out():
    g = FakeGenome().put([17, 18, 19], "AAA")
    assert detect_stop_convention(g.fetch, tx([(11, 19)], exons=[(11, 19)])) == StopConvention.MISSING


def test_detect_missing_for_short_cds():
    g = FakeGenome()
    assert detect_stop_convention(g.fetch, tx([(11, 12)], exons=[(11, 40)])) == StopConvention.MISSING


def test_detect_minus_strand_excluded():
    g = FakeGenome().put([13, 12, 11], "AAA", strand=Strand.MINUS)
    g.put([10, 9, 8], "TGA", strand=Strand.MINUS)
    t = tx([(11, 19)], exons=[(5, 19)], strand=Strand.MINUS)
    assert detect_stop_convention(g.fetch, t) == StopConvention.EXCLUDED


# --- normalize ---------------------------------------------------------------

def test_normalize_leaves_included_cds_alone():
    g = FakeGenome().put([17, 18, 19], "TAA")
    out = normalize(g.fetch, tx([(11, 19)], exons=[(11, 40)]))
    assert out.cds == Chain([(11, 19)])
    assert out.stop == StopConvention.INCLUDED


def test_normalize_extends_excluded_cds():
    g = FakeGenome().put([17, 18, 19], "AAA").put([20, 21, 22], "TGA")
    out = normalize(g.fetch, tx([(11, 19)], exons=[(11, 40)]))
    assert out.cds == Chain([(11, 22)])
    assert out.stop == StopConvention.EXCLUDED
    assert last_codon(g.fetch, out) == "TGA"


def test_normalize_keeps_missing_cds_unchanged():
    g = FakeGenome().put([17, 18, 19], "AAA").put([20, 21, 22], "CCC")
    out = normalize(g.fetch, tx([(11, 19)], exons=[(11, 40)]))
    assert out.cds == Chain([(11, 19)])
    assert out.stop == StopConvention.MISSING


def test_normalize_preserves_other_fields():
    g = FakeGenome().put([17, 18, 19], "TAA")
    t = tx([(11, 19)], exons=[(11, 40)]).model_copy(update={"gene_id": "g1"})
    out = normalize(g.fetch, t)
    assert (out.id, out.chrom, out.strand, out.gene_id) == ("t1", "chr1", Strand.PLUS, "g1")
    assert out.exons == Chain([(11, 40)])


def test_normalize_all_keeps_order():
    g = FakeGenome().put([17, 18, 19], "TAA")
    txs = [tx([(11, 19)], exons=[(11, 40)], tid=f"t{i}") for i in range(3)]
    assert [t.id for t in normalize_all(g.fetch, txs)] == ["t0", "t1", "t2"]
