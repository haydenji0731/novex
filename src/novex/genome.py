"""Access genome sequences & check dinucleotide
"""

from collections.abc import Iterable

import pyfastx

from novex.chains import Chain, Strand
from novex.codons import START_CODON
from novex.junctions import Junction
from novex.transcripts import UpstreamChain

Motifs = frozenset[tuple[str, str]]
Starts = frozenset[str]

START_SETS: dict[str, Starts | None] = {
    "atg": frozenset({START_CODON}),
    "near-cognate": frozenset({START_CODON, "CTG", "GTG", "TTG", "ACG"}),
    "any": None,
}

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


def get_junc_dinucs(fa: pyfastx.Fasta, junction: Junction) -> tuple[str, str]:
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
    return get_junc_dinucs(fa, junction) in allowed


def filter_juncs_by_motif(
    fa: pyfastx.Fasta,
    junctions: Iterable[Junction],
    allowed: Motifs | None = CANONICAL_MOTIFS,
) -> list[Junction]:
    """Keep only junctions whose motifs are allowed.
    """
    if allowed is None:
        return list(junctions)
    return [j for j in junctions if motif_ok(fa, j, allowed)]


def parse_starts(pattern: str) -> Starts | None:
    """Turn a --starts value into an allowed set of start codons, or None for no screening.
    """
    k = pattern.strip().lower()
    if k in START_SETS:
        return START_SETS[k]

    starts = []
    for x in pattern.split(","):
        codon = x.strip().upper()
        if len(codon) != 3 or set(codon) - set("ACGT"):
            raise ValueError(
                f"bad start codon {x.strip()!r}: expected a name ({', '.join(START_SETS)}) "
                "or 3-base codons like 'ATG,CTG'"
            )
        starts.append(codon)
    return frozenset(starts)


def first_codon(fa: pyfastx.Fasta, chrom: str, chain: Chain, strand: Strand) -> str:
    """The first 3 bases of `chain`, read 5'->3'."""
    return fetch_chain_seq(fa, chrom, chain, strand)[:3]


def start_ok(fa: pyfastx.Fasta, upstream: UpstreamChain, allowed: Starts | None) -> bool:
    """Whether the upstream chain begins with an allowed start codon.
    """
    if allowed is None:
        return True
    return first_codon(fa, upstream.chrom, upstream.cds, upstream.strand) in allowed


def filter_upstream_by_start(
    fa: pyfastx.Fasta,
    upstreams: Iterable[UpstreamChain],
    allowed: Starts | None,
) -> list[UpstreamChain]:
    """Keep only upstream chains starting with an allowed codon."""
    if allowed is None:
        return list(upstreams)
    return [u for u in upstreams if start_ok(fa, u, allowed)]
