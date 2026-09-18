"""Access genome sequences & check dinucleotide
"""

from collections.abc import Iterable

import pyfastx

from novex.chains import Chain, Strand
from novex.junctions import Junction

Motifs = frozenset[tuple[str, str]]

CANONICAL_MOTIFS: Motifs = frozenset({("GT", "AG"), ("GC", "AG")})

MOTIF_SETS: dict[str, Motifs | None] = {
    "canonical": CANONICAL_MOTIFS,
    "gtag": frozenset({("GT", "AG")}),
    "u12": CANONICAL_MOTIFS | frozenset({("AT", "AC")}),
    "any": None,
}


def parse_motifs(pattern: str) -> Motifs | None:
    """Turn a --motifs value into an allowed set, or None for no screening.
    """
    k = pattern.strip().lower()
    if k in MOTIF_SETS:
        return MOTIF_SETS[k]

    motifs = []
    for x in pattern.split(","):
        donor, sep, acceptor = x.strip().upper().partition("-")
        if not sep or len(donor) != 2 or len(acceptor) != 2:
            raise ValueError(
                f"bad motif {x.strip()!r}: expected a name ({', '.join(MOTIF_SETS)}) "
                "or dinucleotide pairs like 'GT-AG'"
            )
        motifs.append((donor, acceptor))
    return frozenset(motifs)


def load_genome(path) -> pyfastx.Fasta:
    return pyfastx.Fasta(str(path))


def fetch_chain_seq(fa: pyfastx.Fasta, chrom: str, chain: Chain, strand: Strand) -> str:
    """A chain's spliced sequence, 5'->3', upper-cased.
    """
    if strand == Strand.UNKNOWN:
        raise ValueError("unknown strand")

    intervals = [(x.start, x.end) for x in chain.intervals]

    # 1-based, inclusive; different from slicing Sequence obj directly
    return fa.fetch(chrom, intervals, strand=str(strand)).upper()


def junction_dinucs(fa: pyfastx.Fasta, junction: Junction) -> tuple[str, str]:
    """(donor, acceptor) dinucleotides of `junction`, read 5'->3'.
    """
    step = junction.strand.sign
    donor = sorted((junction.donor, junction.donor + step))
    acceptor = sorted((junction.acceptor - step, junction.acceptor))
    return (
        fetch_chain_seq(fa, junction.chrom, Chain([tuple(donor)]), junction.strand),
        fetch_chain_seq(fa, junction.chrom, Chain([tuple(acceptor)]), junction.strand),
    )


def motif_ok(
    fa: pyfastx.Fasta,
    junction: Junction,
    allowed: Motifs | None,
) -> bool:
    """Whether the junction's dinucleotides are one of `allowed`.
    """
    if allowed is None: # no screening
        return True
    return junction_dinucs(fa, junction) in allowed


def filter_by_motif(
    fa: pyfastx.Fasta,
    junctions: Iterable[Junction],
    allowed: Motifs | None = CANONICAL_MOTIFS,
) -> list[Junction]:
    """Keep only junctions whose motifs are allowed.
    """
    if allowed is None:
        return list(junctions)
    return [j for j in junctions if motif_ok(fa, j, allowed)]
