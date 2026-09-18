"""Build constructs: upstream chain -> junction -> reference CDS, and check the ORF.
"""

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from novex.chains import Chain, Strand, splice
from novex.codons import STOP_CODONS, translate
from novex.junctions import Junction, JunctionIndex
from novex.stop_codons import Fetch
from novex.transcripts import CdsIndex, RefTranscript, UpstreamChain


class RejectReason(StrEnum):
    """reason for rejecting a construct"""
    FRAME = "frame" # indivisible by 3
    NO_STOP = "no_stop"
    PTC = "ptc" # premature stop
    DUPLICATE = "duplicate" # same CDS chain as a construct already built

class Rejection(BaseModel, frozen=True):
    upstream_id: str
    reference_id: str
    junction: Junction
    reason: RejectReason

class Construct(BaseModel, frozen=True):
    id: str
    upstream_id: str
    reference: RefTranscript
    junction: Junction
    cds: Chain

    @property
    def chrom(self) -> str:
        return self.reference.chrom

    @property
    def strand(self) -> Strand:
        return self.reference.strand


def check_orf(seq: str) -> RejectReason | None:
    """Check a construct's spliced CDS sequence; None means it is a valid ORF.
    """
    if len(seq) == 0:
        raise ValueError(f'empty seq')
    
    if len(seq) % 3 != 0:
        return RejectReason.FRAME

    if seq[-3:] not in STOP_CODONS:
        return RejectReason.NO_STOP

    if "*" in translate(seq)[:-1]: # ambiguous codons are okay
        return RejectReason.PTC

    return None


def build_cds(upstream: UpstreamChain, reference: RefTranscript, junction: Junction) -> Chain | None:
    """Splice the upstream chain into the reference CDS across `junction`.
    """

    if upstream.strand != reference.strand:
        raise ValueError(
            f"cannot splice {upstream.id} ({upstream.strand}) into "
            f"{reference.id} ({reference.strand}): different strands"
        )

    if reference.strand.sign > 0:
        return splice(upstream.cds, reference.cds, junction.intron)
    return splice(reference.cds, upstream.cds, junction.intron)

def build_all(
    fetch: Fetch,
    upstreams: Iterable[UpstreamChain],
    junctions: JunctionIndex,
    references: CdsIndex,
) -> tuple[list[Construct], list[Rejection]]:
    """Returns the constructs that passed check_orf, and one Rejection per candidate
    that did not (includes duplicate CDS chains).
    """

    ctr = 0
    constructs: list[Construct] = []
    rejections: list[Rejection] = []
    seen_chains: dict[tuple[str, Strand, Chain], str] = dict()

    def reject(u_id: str, r_id: str, j: Junction, reason: RejectReason) -> None:
        rejections.append(Rejection(upstream_id=u_id, reference_id=r_id, junction=j, reason=reason))

    for u in upstreams:
        for j in junctions.donors_in(u.chrom, u.strand, u.cds):
            for r in references.containing(j.chrom, j.strand, j.acceptor_exon_base):

                cds = build_cds(u, r, j)
                if cds is None:
                    # technically unreachable; donors_in() and containing() guarantee 
                    raise AssertionError(
                        f"splice failed for {u.id} + {r.id} across {j.chrom}:{j.intron}"
                    )

                if (r.chrom, r.strand, cds) in seen_chains:
                    reject(u.id, r.id, j, RejectReason.DUPLICATE)
                    continue

                seq = fetch(r.chrom, cds, r.strand)
                reason = check_orf(seq)
                if reason is not None:
                    reject(u.id, r.id, j, reason)
                    continue

                construct_id = f"cst_{ctr}"
                ctr += 1
                constructs.append(
                    Construct(
                        id=construct_id, upstream_id=u.id,
                        reference=r, junction=j, cds=cds
                    )
                )
                seen_chains[(r.chrom, r.strand, cds)] = construct_id

    return constructs, rejections
