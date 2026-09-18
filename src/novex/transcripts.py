"""Load exon / CDS chains
"""

from collections.abc import Iterable, Iterator
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from intervaltree import IntervalTree
from pydantic import BaseModel

from mjol.utils import load_attributes
from novex.chains import Chain, Strand, GInterval

fileFmt = Literal["gtf", "gff"]


class StopConvention(StrEnum):
    """Whether a reference CDS includes its stop codon.

    UNDETECTED is the state before stops.py inspects the sequence.
    """
    INCLUDED = "included"
    EXCLUDED = "excluded"
    MISSING = "missing"      # partial CDS: no stop codon either way
    UNDETECTED = "undetected"


class UpstreamChain(BaseModel, frozen=True):
    """Open reading frame chain upstream of reference CDS.
    """
    id: str
    chrom: str
    strand: Strand
    cds: Chain


class RefTranscript(BaseModel, frozen=True):
    """A reference transcript with a CDS, the acceptor side of a construct."""
    id: str
    chrom: str
    strand: Strand
    exons: Chain
    cds: Chain
    gene_id: str | None = None
    stop: StopConvention = StopConvention.UNDETECTED

    def with_cds(self, cds: Chain, stop: StopConvention) -> "RefTranscript":
        """Copy with a normalized CDS chain and the detected convention.

        Used by stops.py, since this model is frozen. model_copy(update=...) does
        the work.
        """
        
        raise NotImplementedError


@dataclass
class _TxGroup:
    """Rows collected for one transcript id while reading a file."""
    chrom: str
    strand: Strand
    cds: list[GInterval] = field(default_factory=list)
    exons: list[GInterval] = field(default_factory=list)
    gene_id: str | None = None


def _iter_rows(
    path: str | Path,
    id_attr_key: str,
    fmt: fileFmt,
    keep: tuple[str, ...],
) -> Iterator[tuple[str, str, str, Strand, GInterval, dict[str, str]]]:
    """Yield (tx_id, feature_type, chrom, strand, interval, attributes) for rows in `keep`.
    """
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            if len(f) != 9:
                raise ValueError(f"{path}:{lineno}: expected 9 fields, got {len(f)}")
            chrom, _, ftype, start, end, _, strand = f[:7]
            if ftype not in keep:
                continue
            attrs = load_attributes(f[8], kv_sep=" " if fmt == "gtf" else "=")
            tx_id = attrs.get(id_attr_key)
            if tx_id is None:
                raise ValueError(f"{path}:{lineno}: no {id_attr_key} attribute")
            # TODO: consider adding this to mjol parsing logic
            tx_id = tx_id.strip('"')
            yield (
                tx_id,
                ftype,
                chrom,
                Strand(strand),
                GInterval(start=int(start), end=int(end)),
                attrs,
            )


def _group_by_tx(
    path: str | Path,
    id_attr_key: str,
    fmt: fileFmt,
    keep: tuple[str, ...],
    gene_attr_key: str | None = None,
) -> dict[str, _TxGroup]:
    """Group rows by transcript id.
    """
    data: dict[str, _TxGroup] = {}
    for tx_id, ftype, chrom, strand, interval, attrs in _iter_rows(path, id_attr_key, fmt, keep):
        grp = data.get(tx_id)

        if grp is None:
            gene_id = attrs.get(gene_attr_key) if gene_attr_key else None
            grp = data[tx_id] = _TxGroup(chrom, strand, gene_id=gene_id.strip('"') if gene_id else None)

        elif (grp.chrom, grp.strand) != (chrom, strand):
            raise ValueError(f"{path}: rows for {tx_id} disagree on contig or strand")
        (grp.cds if ftype == "CDS" else grp.exons).append(interval)
    return data


def read_upstream(
    path: str | Path,
    id_attr_key: str = "transcript_id",
    fmt: fileFmt = "gtf"
) -> list[UpstreamChain]:
    """Load upstream ORFs from a GTF/GFF file.
    """
    data = _group_by_tx(path, id_attr_key, fmt, keep=("CDS",))
    return [
        UpstreamChain(id=tx_id, chrom=acc.chrom, strand=acc.strand, cds=Chain(sorted(acc.cds)))
        for tx_id, acc in data.items()
    ]


def read_references(
    path: str | Path,
    id_attr_key: str = "transcript_id",
    gene_attr_key: str = "gene_id",
    fmt: fileFmt = "gtf",
) -> list[RefTranscript]:
    """Load reference transcripts with a CDS.
    """
    data = _group_by_tx(path, id_attr_key, fmt, keep=("CDS", "exon"), gene_attr_key=gene_attr_key)
    return [
        RefTranscript(
            id=tx_id,
            chrom=g.chrom,
            strand=g.strand,
            exons=Chain(sorted(g.exons or g.cds)),
            cds=Chain(sorted(g.cds)),
            gene_id=g.gene_id,
        )
        for tx_id, g in data.items()
        if g.cds
    ]


class CdsIndex:
    """Supports searching transcripts with a CDS covering a specific position.
    """

    def __init__(self, transcripts: Iterable[RefTranscript]):
        """Construct interval trees from CDS intervals.
        """
        self._transcripts = list(transcripts)
        self._trees: dict[tuple[str, Strand], IntervalTree] = defaultdict(IntervalTree)
        for tx in self._transcripts:
            tree = self._trees[(tx.chrom, tx.strand)]
            for x in tx.cds.intervals:
                tree.addi(x.start, x.end + 1, tx)

    def __len__(self) -> int:
        """Number of transcripts.
        """
        return len(self._transcripts)

    def containing(self, chrom: str, strand: Strand, pos: int) -> list[RefTranscript]:
        """Transcripts on (chrom, strand) whose CDS covers `pos`.
        """
        tree = self._trees.get((chrom, strand))
        if tree is None:
            return []
        return sorted({x.data for x in tree.at(pos)}, key=lambda t: t.id)
