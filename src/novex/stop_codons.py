"""Detect whether a reference CDS includes its stop codon, and normalize when it does not.
"""

from collections.abc import Callable, Iterable

from novex.chains import Chain, GInterval, Strand
from novex.transcripts import RefTranscript, StopConvention
from novex.codons import STOP_CODONS

Fetch = Callable[[str, Chain, Strand], str]


def last_codon(fetch: Fetch, tx: RefTranscript) -> str:
    """The final 3 bases (upper-cased) of tx.cds, read 5'->3'.
    """
    if len(tx.cds) < 3:
        raise ValueError(f"{tx.id}: CDS is {len(tx.cds)} bp, too short to hold a codon")

    return fetch(tx.chrom, tx.cds, tx.strand)[-3:].upper()


def extend_cds(tx: RefTranscript, n: int = 3) -> Chain | None:
    """tx.cds grown by n (default: 3) more transcript bases, following tx.exons.
    """
    cds_end = tx.cds.tx_to_genomic(tx.strand, len(tx.cds) - 1)   # last coding base, 5'->3'
    offset = tx.exons.genomic_to_tx(tx.strand, cds_end)

    extra: list[GInterval] = []
    for k in range(1, n + 1):
        try:
            pos = tx.exons.tx_to_genomic(tx.strand, offset + k)
        except IndexError:
            return None
        extra.append(GInterval(pos, pos))

    return _merge(list(tx.cds.intervals) + extra)


def _merge(intervals: list[GInterval]) -> Chain:
    """Sort intervals and join the ones that touch or overlap."""
    merged: list[GInterval] = []
    for x in sorted(intervals):
        if merged and x.start <= merged[-1].end + 1:
            merged[-1] = GInterval(merged[-1].start, max(merged[-1].end, x.end))
        else:
            merged.append(x)
    return Chain(merged)


def detect_stop_convention(fetch: Fetch, tx: RefTranscript) -> StopConvention:
    """Which convention this transcript's CDS follows.

    INCLUDED : last_codon(tx) is a stop codon
    EXCLUDED : it is not, but the next 3 transcript bases are
    MISSING : neither = a partial CDS
    """
    if len(tx.cds) < 3:
        return StopConvention.MISSING

    if last_codon(fetch, tx) in STOP_CODONS:
        return StopConvention.INCLUDED

    extended = extend_cds(tx)
    if extended is not None:
        codon = fetch(tx.chrom, extended, tx.strand)[-3:].upper()
        if codon in STOP_CODONS:
            return StopConvention.EXCLUDED

    return StopConvention.MISSING


def normalize(fetch: Fetch, tx: RefTranscript) -> RefTranscript:
    """Return tx with its CDS ending in its stop codon, and `stop` recording the input's convention.

    INCLUDED -> unchanged apart from `stop`
    EXCLUDED -> cds replaced by extended_cds(tx)
    MISSING  -> unchanged; callers decide whether to keep such transcripts

    Uses RefTranscript.with_cds, since the model is frozen.
    """
    stop_convention = detect_stop_convention(fetch, tx)
    if stop_convention != StopConvention.EXCLUDED: # included or missing
        return tx.copy_with_cds(tx.cds, stop_convention) # nothing done

    extended = extend_cds(tx)
    if extended is None:
        raise RuntimeError(f"{tx.id}: CDS excludes its stop codon but cannot be extended")

    return tx.copy_with_cds(extended, stop_convention)


def normalize_all(fetch: Fetch, txs: Iterable[RefTranscript]) -> list[RefTranscript]:
    """Run normalize() over all transcripts.
    """
    return [normalize(fetch, tx) for tx in txs]


def drop_partial(txs: Iterable[RefTranscript]) -> list[RefTranscript]:
    """Discard reference transcripts whose stop codon was found.
    """
    return [tx for tx in txs if tx.stop != StopConvention.MISSING]
