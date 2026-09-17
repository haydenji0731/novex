"""Splice junctions and lookup by donor position.
"""

from collections.abc import Iterable, Iterator
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, field_validator

from bisect import bisect_right, bisect_left

from novex.chains import Chain, GInterval, Strand


class Junction(BaseModel, frozen=True):
    """One splice junction: an intron on a stranded contig.

    `name` and `score` are whatever the BED file carried (columns 4 and 5), kept
    only for reporting.
    """
    chrom: str
    strand: Strand
    intron: GInterval
    name: str | None = None
    score: float | None = None

    @field_validator("intron")
    @classmethod
    def _validate(cls, v: GInterval) -> GInterval:
        """Reject introns that are empty or backwards (start > end).
        """
        if v.length < 1:
            raise ValueError(f"invalid intron interval: {v}")
        return v

    @property
    def _sign(self) -> int:
        match self.strand:
            case Strand.PLUS: return 1
            case Strand.MINUS: return -1
            case _: raise ValueError("unknown strand")

    @property
    def donor(self) -> int:
        """First intronic base.
        """
        return self.intron.start if self._sign > 0 else self.intron.end

    @property
    def acceptor(self) -> int:
        """Last intronic base.
        """
        return self.intron.end if self._sign > 0 else self.intron.start

    @property
    def donor_exon_base(self) -> int:
        """Exonic base upstream of donor.
        """
        return self.donor - self._sign
        

    @property
    def acceptor_exon_base(self) -> int:
        """Exonic base downstream of acceptor.
        """
        return self.acceptor + self._sign


def read_bed(path: str | Path) -> Iterator[Junction]:
    """Yield Junctions from a BED file of introns.
    """
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            f = line.split("\t")
            if len(f) < 6:
                raise ValueError(f"{path}:{lineno}: expected at least 6 fields, got {len(f)}")
            chrom, start, end, name, score, strand = f[:6]

            if strand == Strand.UNKNOWN: # skip unknown strand junctions
                continue

            yield Junction(
                chrom=chrom,
                strand=Strand(strand),
                intron=GInterval(int(start) + 1, int(end)), # convert from 0-based, half-open to 1-based, fully-closed
                name=None if name == "." else name,
                score=None if score == "." else float(score)
            )

class _JGroup(NamedTuple):
    positions: list[int]
    junctions: list[Junction]

class JunctionIndex:
    """Junctions grouped by (chrom, strand) and sorted by donor_exon_base.
    """

    def __init__(self, junctions: Iterable[Junction]):
        """Group junctions by (chrom, strand), then sort each group by donor_exon_base.
        """
        jdata: dict[tuple[str, Strand], list[Junction]] = defaultdict(list)
        for j in junctions:
            jdata[(j.chrom, j.strand)].append(j)

        self._groups: dict[tuple[str, Strand], _JGroup] = {}
        for k, js in jdata.items():
            js.sort(key=lambda j: j.donor_exon_base)
            self._groups[k] = _JGroup(
                positions = [j.donor_exon_base for j in js],
                junctions = js
            )

    def __len__(self) -> int:
        """Total number of junctions."""
        return sum(len(g.junctions) for g in self._groups.values())

    def donors_in(self, chrom: str, strand: Strand, chain: Chain) -> list[Junction]:
        """Junctions on (chrom, strand) whose donor_exon_base falls inside `chain`.
        """
        g = self._groups.get((chrom, strand))
        if g is None:
            return []
        out: list[Junction] = []

        for x in chain.intervals:
            lo = bisect_left(g.positions, x.start)
            hi = bisect_right(g.positions, x.end)
            out.extend(g.junctions[lo:hi])

        return out
    