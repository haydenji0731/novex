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


# --- reference priority (MANE Select, then shorter) --------------------------

def tagged_row(feature, start, end, tid, tags):
    row = gtf_row(feature, start, end, tid).rstrip("\n")
    return row + "".join(f' tag "{t}";' for t in tags) + "\n"


def test_read_references_detects_mane_among_repeated_tags(tmp_path):
    # GENCODE repeats `tag`; MANE_Select must be found even when it is not the last one
    p = write(
        tmp_path,
        tagged_row("exon", 100, 300, "t1", ["basic", "MANE_Select", "CCDS"]),
        tagged_row("CDS", 150, 250, "t1", ["basic", "MANE_Select", "CCDS"]),
        tagged_row("exon", 100, 300, "t2", ["basic", "CCDS"]),
        tagged_row("CDS", 150, 250, "t2", ["basic", "CCDS"]),
    )
    refs = {r.id: r for r in read_references(p)}
    assert refs["t1"].mane_select is True
    assert refs["t2"].mane_select is False


def test_read_references_detects_mane_in_gff(tmp_path):
    attrs = "ID=x;Parent=t1;transcript_id=t1;gene_id=g1;tag=basic,MANE_Select,CCDS"
    rows = [f"chr1\tnovex\t{ft}\t100\t300\t.\t+\t0\t{attrs}\n" for ft in ("exon", "CDS")]
    (r,) = read_references(write(tmp_path, *rows, name="in.gff"), fmt="gff")
    assert r.mane_select is True


def test_index_prefers_mane_select_over_shorter():
    long_mane = RefTranscript(
        id="t9", chrom="chr1", strand=Strand.PLUS,
        exons=Chain([(10, 50), (100, 300)]), cds=Chain([(150, 250)]), mane_select=True,
    )
    short = ref("t1", [(150, 250)], exons=[(100, 300)])
    idx = CdsIndex([short, long_mane])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, 200)] == ["t9", "t1"]


def test_index_prefers_shorter_transcript_then_id():
    # same CDS, differ only in 5' UTR: shorter exon chain wins regardless of id
    longer = ref("t1", [(150, 250)], exons=[(10, 50), (100, 300)])
    shorter = ref("t2", [(150, 250)], exons=[(100, 300)])
    tie = ref("t0", [(150, 250)], exons=[(100, 300)])
    idx = CdsIndex([longer, shorter, tie])
    assert [t.id for t in idx.containing("chr1", Strand.PLUS, 200)] == ["t0", "t2", "t1"]


@pytest.mark.parametrize("tag,expected", [
    ("MANE_Select", True),          # GENCODE / Ensembl
    ("MANE Select", True),          # RefSeq
    ("mane-select", True),
    ("MANE Plus Clinical", False),  # not the MANE Select transcript
    ("RefSeq Select", False),
])
def test_read_references_mane_tag_conventions(tmp_path, tag, expected):
    p = write(
        tmp_path,
        tagged_row("exon", 100, 300, "t1", ["basic", tag]),
        tagged_row("CDS", 150, 250, "t1", ["basic", tag]),
    )
    (r,) = read_references(p)
    assert r.mane_select is expected
