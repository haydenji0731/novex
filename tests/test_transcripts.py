import pytest

from novex.chains import Chain, Strand
from novex.transcripts import (
    CdsIndex,
    RefTranscript,
    StopConvention,
    OrfChain,
    read_references,
    read_queries,
)


def gtf_row(feature, start, end, tid, chrom="chr1", strand="+", gene="g1"):
    attrs = f'gene_id "{gene}"; transcript_id "{tid}";'
    return f"{chrom}\tnovex\t{feature}\t{start}\t{end}\t.\t{strand}\t0\t{attrs}\n"


def write(tmp_path, *rows, name="in.gtf"):
    p = tmp_path / name
    p.write_text("".join(rows))
    return p


def ref(tid, cds, chrom="chr1", strand=Strand.PLUS, exons=None):
    return RefTranscript(
        id=tid,
        chrom=chrom,
        strand=strand,
        exons=Chain(exons or cds),
        cds=Chain(cds),
    )


# --- read_queries -----------------------------------------------------------

def test_read_queries_builds_chain_from_cds_rows(tmp_path):
    p = write(
        tmp_path,
        gtf_row("transcript", 100, 300, "u1"),
        gtf_row("CDS", 100, 150, "u1"),
        gtf_row("CDS", 251, 300, "u1"),
    )
    (u,) = read_queries(p)
    assert u.id == "u1"
    assert u.chrom == "chr1"
    assert u.strand == Strand.PLUS
    assert u.cds == Chain([(100, 150), (251, 300)])


def test_read_queries_ignores_exon_and_other_rows(tmp_path):
    p = write(
        tmp_path,
        "# a comment\n",
        gtf_row("exon", 50, 300, "u1"),       # wider than the CDS: must not be used
        gtf_row("CDS", 100, 150, "u1"),
        gtf_row("start_codon", 100, 102, "u1"),
    )
    (u,) = read_queries(p)
    assert u.cds == Chain([(100, 150)])


def test_read_queries_sorts_rows_by_position(tmp_path):
    p = write(
        tmp_path,
        gtf_row("CDS", 251, 300, "u1"),
        gtf_row("CDS", 100, 150, "u1"),
    )
    (u,) = read_queries(p)
    assert u.cds == Chain([(100, 150), (251, 300)])


def test_read_queries_separates_transcripts(tmp_path):
    p = write(
        tmp_path,
        gtf_row("CDS", 100, 150, "u1"),
        gtf_row("CDS", 100, 150, "u2", chrom="chr2", strand="-"),
    )
    by_id = {u.id: u for u in read_queries(p)}
    assert set(by_id) == {"u1", "u2"}
    assert by_id["u2"].chrom == "chr2"
    assert by_id["u2"].strand == Strand.MINUS


def test_read_queries_rejects_inconsistent_rows(tmp_path):
    p = write(
        tmp_path,
        gtf_row("CDS", 100, 150, "u1"),
        gtf_row("CDS", 200, 250, "u1", strand="-"),
    )
    with pytest.raises(ValueError):
        read_queries(p)


# --- read_references ---------------------------------------------------------

def test_read_references_keeps_exons_and_cds(tmp_path):
    p = write(
        tmp_path,
        gtf_row("exon", 100, 200, "t1"),
        gtf_row("exon", 301, 400, "t1"),
        gtf_row("CDS", 150, 200, "t1"),
        gtf_row("CDS", 301, 350, "t1"),
    )
    (t,) = read_references(p)
    assert t.id == "t1"
    assert t.gene_id == "g1"
    assert t.exons == Chain([(100, 200), (301, 400)])
    assert t.cds == Chain([(150, 200), (301, 350)])
    assert t.stop == StopConvention.UNDETECTED


def test_read_references_skips_transcripts_without_cds(tmp_path):
    p = write(
        tmp_path,
        gtf_row("exon", 100, 200, "nc1"),          # ncRNA: exons only
        gtf_row("exon", 100, 200, "t1"),
        gtf_row("CDS", 150, 200, "t1"),
    )
    assert [t.id for t in read_references(p)] == ["t1"]


def test_read_references_falls_back_to_cds_for_exons(tmp_path):
    p = write(tmp_path, gtf_row("CDS", 150, 200, "t1"))
    (t,) = read_references(p)
    assert t.exons == Chain([(150, 200)])


def test_read_references_reads_minus_strand(tmp_path):
    p = write(
        tmp_path,
        gtf_row("exon", 100, 200, "t1", strand="-"),
        gtf_row("CDS", 150, 200, "t1", strand="-"),
    )
    (t,) = read_references(p)
    assert t.strand == Strand.MINUS


# --- CdsIndex ----------------------------------------------------------------

def test_index_len_counts_transcripts():
    idx = CdsIndex([ref("t1", [(100, 200), (301, 400)]), ref("t2", [(100, 200)])])
    assert len(idx) == 2


@pytest.mark.parametrize(
    "pos, expected",
    [
        (100, ["t1"]),   # first base
        (200, ["t1"]),   # last base of an interval: inclusive ends
        (201, []),       # first intronic base
        (300, []),       # last intronic base
        (301, ["t1"]),   # first base of the next interval
        (400, ["t1"]),   # last base of the CDS
        (401, []),       # past the end
        (99, []),        # before the start
    ],
)
def test_index_containing_is_end_inclusive(pos, expected):
    idx = CdsIndex([ref("t1", [(100, 200), (301, 400)])])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, pos)] == expected


def test_index_returns_all_overlapping_isoforms():
    idx = CdsIndex([
        ref("t1", [(100, 300)]),
        ref("t2", [(200, 400)]),   # overlaps t1
        ref("t3", [(500, 600)]),
    ])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, 250)] == ["t1", "t2"]


def test_index_lists_each_transcript_once():
    # several intervals of the same transcript are in the tree; output must not repeat it
    idx = CdsIndex([ref("t1", [(100, 200), (201, 300)])])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, 150)] == ["t1"]


def test_index_separates_chrom_and_strand():
    idx = CdsIndex([
        ref("t1", [(100, 200)]),
        ref("t2", [(100, 200)], strand=Strand.MINUS),
        ref("t3", [(100, 200)], chrom="chr2"),
    ])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, 150)] == ["t1"]
    assert [t.id for t in idx.containing("chr1", Strand.MINUS, 150)] == ["t2"]
    assert [t.id for t in idx.containing("chr2", Strand.PLUS, 150)] == ["t3"]


def test_index_empty():
    idx = CdsIndex([])
    assert len(idx) == 0
    assert idx.containing("chr1", Strand.PLUS, 150) == []
