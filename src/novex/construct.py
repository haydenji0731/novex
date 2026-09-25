"""Build constructs: upstream chain -> junction -> reference CDS, and check the ORF.
"""

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from novex.chains import Chain, Strand, merge, splice, head
from novex.codons import STOP_CODONS, translate
from novex.junctions import Junction, JunctionIndex
from novex.stop_codons import Fetch
from novex.transcripts import CdsIndex, RefTranscript, OrfChain


class RejectReason(StrEnum):
    """reason for rejecting a construct"""
    FRAME = "frame" # indivisible by 3
    NO_STOP = "no_stop"
    PTC = "ptc" # premature stop
    DUPLICATE = "duplicate" # same CDS chain as a construct already built
    NOT_TRANSCRIBED = "not_transcribed"
    LENGTH = "length" # construct CDS / reference CDS below the threshold

# TODO: add a `direction` column so rejections from -d up and -d down runs can be told
# apart once merged -- the reasons mean different things (downstream PTC = an in-frame stop
# in the REFERENCE's own CDS, before the junction)
class Rejection(BaseModel, frozen=True):
    query_id: str
    reference_id: str
    junction: Junction
    reason: RejectReason

class Construct(BaseModel, frozen=True):
    id: str
    query_id: str
    reference: RefTranscript
    junction: Junction
    cds: Chain
    exons: Chain   # cds plus annotated UTRs; what GTF exon rows are written from

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


# UPSTREAM ONLY
def build_cds_upstream(upstream: OrfChain, reference: RefTranscript, junction: Junction) -> Chain | None:
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

# UPSTREAM ONLY
def build_exons_upstream(upstream: OrfChain, reference: RefTranscript, junction: Junction) -> Chain | None:
    """Splice the upstream chain into the reference exons across `junction`.
    """
    if reference.strand.sign > 0:
        return splice(upstream.cds, reference.exons, junction.intron)
    return splice(reference.exons, upstream.cds, junction.intron)

# UPSTREAM ONLY
def add_utr5(exons: Chain, cds: Chain, reference: RefTranscript) -> Chain:
    """Prepend the reference's 5' UTR, i.e. its exons before the construct's first coding base.
    """
    strand = reference.strand
    first_coding = cds.tx_to_genomic(strand, 0)

    # TODO: add an example for illustration later
    offset = reference.exons.genomic_to_tx(strand, first_coding)
    if offset == 0:
        return exons # no 5' utr added
    
    boundary = reference.exons.tx_to_genomic(strand, offset - 1)

    utr5 = (
        reference.exons.clip_end(boundary) if strand.sign > 0
        else reference.exons.clip_start(boundary)
    )

    return merge(list(utr5.intervals) + list(exons.intervals))

# UPSTREAM ONLY
def build_chains_upstream(
    upstream: OrfChain, reference: RefTranscript, junction: Junction
) -> tuple[Chain, Chain] | None:
    """(cds, exons) for one construct consisting of upstream & reference, 
    or None if the junction does not apply.
    """
    cds = build_cds_upstream(upstream, reference, junction)
    exons = build_exons_upstream(upstream, reference, junction)
    if cds is None or exons is None:
        return None
    return cds, add_utr5(exons, cds, reference)

def transcribes(chain: Chain, reference: RefTranscript) -> bool:
    """Whether every base of `chain` lies within this reference's exons."""
    for x in chain.intervals:
        i = reference.exons._find(x.start)
        if i is None or x.end > reference.exons.intervals[i].end:
            return False
    return True


# UPSTREAM ONLY
def build_all_upstream(
    fetch: Fetch,
    queries: Iterable[OrfChain],
    junctions: JunctionIndex,
    references: CdsIndex,
    min_cds_ratio: float = 0.0,
) -> tuple[list[Construct], list[Rejection]]:
    """Returns the constructs that passed check_orf, and one Rejection per candidate
    that did not (includes duplicate CDS chains).
    """

    ctr = 0
    constructs: list[Construct] = []
    rejections: list[Rejection] = []
    seen_chains: dict[tuple[str, Strand, Chain], str] = dict()

    def reject(u_id: str, r_id: str, j: Junction, reason: RejectReason) -> None:
        rejections.append(Rejection(query_id=u_id, reference_id=r_id, junction=j, reason=reason))

    for u in queries:
        for j in junctions.donors_in(u.chrom, u.strand, u.cds):
            for r in references.containing(j.chrom, j.strand, j.acceptor_exon_base):
                
                pos = j.donor_exon_base
                retained = u.cds.clip_end(pos) if u.strand.sign > 0 else u.cds.clip_start(pos)
                if not transcribes(retained, r):
                    reject(u.id, r.id, j, RejectReason.NOT_TRANSCRIBED)
                    continue
                
                chains = build_chains_upstream(u, r, j)
                if chains is None:
                    # technically unreachable; donors_in() and containing() guarantee 
                    raise AssertionError(
                        f"splice failed for {u.id} + {r.id} across {j.chrom}:{j.intron}"
                    )
                cds, exons = chains

                if len(cds) / len(r.cds) < min_cds_ratio:
                    reject(u.id, r.id, j, RejectReason.LENGTH)
                    continue

                if (r.chrom, r.strand, cds) in seen_chains:
                    reject(u.id, r.id, j, RejectReason.DUPLICATE)
                    continue

                seq = fetch(r.chrom, cds, r.strand)
                reason = check_orf(seq)
                if reason is not None:
                    reject(u.id, r.id, j, reason)
                    continue

                construct_id = f"ust_{ctr}"
                ctr += 1
                constructs.append(
                    Construct(
                        id=construct_id, query_id=u.id,
                        reference=r, junction=j, cds=cds, exons=exons
                    )
                )
                seen_chains[(r.chrom, r.strand, cds)] = construct_id

    return constructs, rejections

def build_cds_downstream(downstream: OrfChain, reference: RefTranscript, junction: Junction) -> Chain | None:
    """Splice the reference CDS into the downstream chain across `junction`.
    """
    if downstream.strand != reference.strand:
        raise ValueError(
            f"cannot splice {downstream.id} ({downstream.strand}) into "
            f"{reference.id} ({reference.strand}): different strands"
        )

    if reference.strand.sign > 0:
        return splice(reference.cds, downstream.cds, junction.intron)
    return splice(downstream.cds, reference.cds, junction.intron)


def build_exons_downstream(downstream: OrfChain, reference: RefTranscript, junction: Junction) -> Chain | None:
    """Exon chain for a downstream construct: reference exons up to the junction, then the query chain.

    Splicing through reference.exons carries the reference 5' UTR side; whatever of the
    query chain lies beyond the trimmed stop codon becomes 3' UTR.
    """
    if reference.strand.sign > 0:
        return splice(reference.exons, downstream.cds, junction.intron)
    return splice(downstream.cds, reference.exons, junction.intron)


def junction_offset(reference: RefTranscript, junction: Junction) -> int:
    """How many CDS bases the reference contributes, i.e. where the query part starts."""
    pos = junction.donor_exon_base
    prefix = (
        reference.cds.clip_end(pos) if reference.strand.sign > 0
        else reference.cds.clip_start(pos)
    )
    return len(prefix)


def find_stop_offset(seq: str) -> int | None:
    """0-based offset just past the first in-frame stop codon, or None if there is none.
    """
    aa = translate(seq)
    k = aa.find("*")
    return None if k < 0 else 3 * (k + 1)


def trim_to_stop(cds: Chain, strand: Strand, seq: str, junc_offset: int) -> Chain | RejectReason:
    """Cut `cds` back to its first in-frame stop, or say why it cannot be used.

    `junc_offset` is how many CDS bases come from the reference, i.e. where the query
    part starts.
    """
    n = find_stop_offset(seq)
    if n is None:
        return RejectReason.NO_STOP # ORF runs off the end of the query chain
    if n <= junc_offset:
        return RejectReason.PTC # stop lies in the reference part, before the junction

    return head(cds, strand, n)


def add_utr3(exons: Chain, cds: Chain, reference: RefTranscript) -> Chain:
    """Append the reference's 3' UTR, i.e. its exons beyond the construct's last coding base.
    """
    strand = reference.strand
    last_coding = cds.tx_to_genomic(strand, len(cds) - 1)

    offset = reference.exons.genomic_to_tx(strand, last_coding)
    if offset == len(reference.exons) - 1:
        return exons # no 3' utr added
    
    boundary = reference.exons.tx_to_genomic(strand, offset + 1)

    utr3 = (
        reference.exons.clip_start(boundary) if strand.sign > 0
        else reference.exons.clip_end(boundary)
    )

    return merge(list(exons.intervals) + list(utr3.intervals))


def build_chains_downstream(
    downstream: OrfChain, reference: RefTranscript, junction: Junction
) -> tuple[Chain, Chain] | None:
    """(cds, exons) for one construct consisting of reference & downstream, 
    or None if the junction does not apply.
    """
    cds = build_cds_downstream(downstream, reference, junction)
    exons = build_exons_downstream(downstream, reference, junction)
    if cds is None or exons is None:
        return None
    return cds, add_utr3(exons, cds, reference)


def build_all_downstream(
    fetch: Fetch,
    queries: Iterable[OrfChain],
    junctions: JunctionIndex,
    references: CdsIndex,
    min_cds_ratio: float = 0.0,
) -> tuple[list[Construct], list[Rejection]]:
    """Same contract as build_all_upstream, for -d down.

    Unlike upstream, the CDS is not known until the sequence is read: the candidate runs
    to the end of the query chain and trim_to_stop cuts it back to the first in-frame
    stop. So the duplicate check happens after the fetch, not before it.

    TODO(decide): Construct has no field recording which direction built it, so the GTF
    attributes and rejections.tsv cannot distinguish upstream from downstream results.
    """
    ctr = 0
    constructs: list[Construct] = []
    rejections: list[Rejection] = []
    seen_chains: dict[tuple[str, Strand, Chain], str] = dict()

    def reject(q_id: str, r_id: str, j: Junction, reason: RejectReason) -> None:
        rejections.append(Rejection(query_id=q_id, reference_id=r_id, junction=j, reason=reason))

    for d in queries:
        for j in junctions.acceptors_in(d.chrom, d.strand, d.cds):
            for r in references.containing(j.chrom, j.strand, j.donor_exon_base):
                
                pos = j.acceptor_exon_base
                retained = d.cds.clip_start(pos) if d.strand.sign > 0 else d.cds.clip_end(pos)
                if not transcribes(retained, r):
                    reject(d.id, r.id, j, RejectReason.NOT_TRANSCRIBED)
                    continue

                chains = build_chains_downstream(d, r, j)
                if chains is None:
                    # unreachable: acceptors_in and containing guarantee both boundary bases
                    raise AssertionError(
                        f"splice failed for {r.id} + {d.id} across {j.chrom}:{j.intron}"
                    )
                candidate, exons = chains

                seq = fetch(r.chrom, candidate, r.strand)
                result = trim_to_stop(candidate, r.strand, seq, junction_offset(r, j))
                if isinstance(result, RejectReason):
                    reject(d.id, r.id, j, result)
                    continue

                cds = result
                if len(cds) / len(r.cds) < min_cds_ratio:
                    reject(d.id, r.id, j, RejectReason.LENGTH)
                    continue

                key = (r.chrom, r.strand, cds)
                if key in seen_chains:
                    reject(d.id, r.id, j, RejectReason.DUPLICATE)
                    continue

                construct_id = f"dst_{ctr}"
                ctr += 1
                constructs.append(
                    Construct(
                        id=construct_id, query_id=d.id,
                        reference=r, junction=j,
                        cds=cds, exons=exons
                    )
                )
                seen_chains[key] = construct_id

    return constructs, rejections
