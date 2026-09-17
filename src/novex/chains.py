"""Operations on exon / CDS chains
"""

from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import NamedTuple
from pydantic import BaseModel, field_validator
from bisect import bisect_right

class GInterval(NamedTuple): # ends inclusive
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

class Strand(StrEnum):
    PLUS = "+"
    MINUS = "-"
    UNKNOWN = "."

class Chain(BaseModel, frozen=True):
    intervals: tuple[GInterval, ...]

    def __init__(self, intervals: Iterable[tuple[int, int]]):
        super().__init__(intervals=intervals)

    def __len__(self) -> int:
        return sum(x.length for x in self.intervals)

    def __contains__(self, pos: int) -> bool:
        return self._find(pos) is not None
    
    def __add__(self, other: "Chain") -> "Chain":
        return Chain(self.intervals + other.intervals)

    def _find(self, pos: int) -> int | None:
        i = bisect_right(self.intervals, pos, key=lambda x: x.start) - 1
        if i >= 0 and pos <= self.intervals[i].end:
            return i
        return None

    def _tx_order(self, strand: Strand) -> Iterator[GInterval]:
        match strand:
            case Strand.PLUS:
                return iter(self.intervals)
            case Strand.MINUS:
                return reversed(self.intervals)
            case _:
                raise ValueError("cannot order intervals on unknown strand")

    @field_validator("intervals")
    @classmethod
    def _validate(cls, v):
        if not v:
            raise ValueError("chain must contain at least one interval")
        prev_end = None
        for x in v:
            if x.start > x.end:
                raise ValueError(f"interval start > end: {x}")
            if prev_end is not None and x.start <= prev_end:
                raise ValueError(f"intervals must be sorted and non-overlapping: {x.start} follows {prev_end}")
            prev_end = x.end
        return v

    def clip_end(self, pos: int) -> "Chain":
        """Keep only the bases at or before `pos`; the interval containing `pos` is truncated to end there.
        """
        i = self._find(pos)
        if i is None:
            raise ValueError(f"{pos} is not in chain {self.intervals}")

        last = GInterval(self.intervals[i].start, pos)
        return Chain(self.intervals[:i] + (last,))

    def clip_start(self, pos: int) -> "Chain":
        """Keep only the bases at or after `pos`; the interval containing `pos` is truncated to start there.
        """
        i = self._find(pos)
        if i is None:
            raise ValueError(f"{pos} is not in chain {self.intervals}")

        first = GInterval(pos, self.intervals[i].end)
        return Chain((first,) + self.intervals[i + 1:])

    def genomic_to_tx(self, strand: Strand, pos: int) -> int:
        """0-based offset of genomic `pos` from the 5' end of the chain, reading in `strand` direction.
        """
        if strand == Strand.UNKNOWN:
            raise ValueError("unknown strand")
        
        i = self._find(pos)
        if i is None:
            raise ValueError(f"{pos} is not in chain {self.intervals}")

        if strand == Strand.PLUS:
            return sum(x.length for x in self.intervals[:i]) + (pos - self.intervals[i].start)
        elif strand == Strand.MINUS:
            return sum(x.length for x in self.intervals[i+1:]) + (self.intervals[i].end - pos)

    def tx_to_genomic(self, strand: Strand, offset: int) -> int:
        """0-based offset of transcript `pos` from the 5' end of the chain, reading in `strand` direction.
        """
        if strand == Strand.UNKNOWN:
            raise ValueError("unknown strand")
        if offset < 0:
            raise IndexError(f"negative offset: {offset}")

        ctr = offset
        for x in self._tx_order(strand):
            if ctr < x.length:
                return x.start + ctr if strand == Strand.PLUS else x.end - ctr
            ctr -= x.length

        raise IndexError(f"offset {offset} is past the end of a {len(self)} bp chain")


def splice(left: Chain, right: Chain, intron: GInterval) -> Chain | None:
    """Join two chains using an intron.
    """
    # (donor, acceptor) in '+' else (acceptor, donor)
    left_end, right_start = intron.start - 1, intron.end + 1
    if left_end not in left or right_start not in right:
        return None

    return left.clip_end(left_end) + right.clip_start(right_start)
