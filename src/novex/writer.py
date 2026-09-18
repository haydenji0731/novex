"""Write constructs as GTF, and rejected candidates as TSV.
"""

from collections.abc import Iterable
from pathlib import Path

from novex.chains import Chain, GInterval, Strand
from novex.construct import Construct, Rejection

SOURCE = "novex"

REJECTION_HEADER = (
    "upstream_id",
    "reference_id",
    "chrom",
    "strand",
    "intron_start",
    "intron_end",
    "junction_name",
    "reason",
)


def construct_attributes(construct: Construct) -> dict[str, str]:
    """GTF attributes carried on every row of a construct.
    """
    j = construct.junction
    return {
        "transcript_id": construct.id,
        "gene_id": construct.reference.gene_id or construct.reference.id,
        "upstream_id": construct.upstream_id,
        "reference_id": construct.reference.id,
        "junction": f"{j.chrom}:{j.intron.start}-{j.intron.end}",
    }


def format_attributes(attrs: dict[str, str]) -> str:
    """GTF attribute column: key "value"; pairs, space separated."""
    return " ".join(f'{key} "{value}";' for key, value in attrs.items())


def gtf_row(
    chrom: str,
    feature: str,
    interval: GInterval,
    strand: Strand,
    attrs: dict[str, str],
    frame: str = ".",
) -> str:
    """One GTF line: 1-based inclusive, tab separated, newline terminated."""
    fields = (
        chrom,
        SOURCE,
        feature,
        str(interval.start),
        str(interval.end),
        ".",
        str(strand),
        frame,
        format_attributes(attrs),
    )
    return "\t".join(fields) + "\n"


def compute_cds_frames(cds: Chain, strand: Strand) -> list[str]:
    """GTF frame values for each CDS interval.
    """
    frames: dict[int, str] = {}
    ptr = 0
    for x in cds._tx_order(strand):
        frames[x.start] = str((3 - ptr % 3) % 3)
        ptr += x.length
    return [frames[x.start] for x in cds.intervals]


def construct_rows(construct: Construct) -> list[str]:
    """transcript, exon and CDS rows for one construct.
    """
    attrs = construct_attributes(construct)
    chrom, strand = construct.chrom, construct.strand
    span = GInterval(construct.exons.intervals[0].start, construct.exons.intervals[-1].end)

    rows = [gtf_row(chrom, "transcript", span, strand, attrs)]
    rows += [gtf_row(chrom, "exon", x, strand, attrs) for x in construct.exons.intervals]
    rows += [
        gtf_row(chrom, "CDS", x, strand, attrs, frame)
        for x, frame in zip(construct.cds.intervals, compute_cds_frames(construct.cds, strand), strict=True)
    ]
    return rows


def write_gtf(path: Path, constructs: Iterable[Construct]) -> int:
    """Write constructs as GTF. Returns how many were written."""
    written = 0
    with open(path, "w") as fh:
        for construct in constructs:
            fh.writelines(construct_rows(construct))
            written += 1
    return written


def write_rejections(path: Path, rejections: Iterable[Rejection]) -> int:
    """Write rejected candidates as TSV, one row each. Returns the row count."""
    written = 0
    with open(path, "w") as fh:
        fh.write("\t".join(REJECTION_HEADER) + "\n")
        for r in rejections:
            fh.write(
                "\t".join(
                    (
                        r.upstream_id,
                        r.reference_id,
                        r.junction.chrom,
                        str(r.junction.strand),
                        str(r.junction.intron.start),
                        str(r.junction.intron.end),
                        r.junction.name or ".",
                        str(r.reason),
                    )
                )
                + "\n"
            )
            written += 1
    return written
