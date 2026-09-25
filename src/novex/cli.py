import argparse
import sys
from functools import partial
from pathlib import Path

from novex.construct import RejectReason, build_all_upstream, build_all_downstream
from novex.genome import (
    fetch_chain_seq,
    filter_juncs_by_motif,
    filter_upstream_by_start,
    filter_reference_by_start,
    load_genome,
    parse_motifs,
    parse_starts,
)
from novex.codons import START_CODON
from novex.junctions import JunctionIndex, read_bed
from novex.stop_codons import drop_partial, normalize_all
from novex.transcripts import CdsIndex, read_references, read_queries
from novex.writer import write_gtf, write_rejections


GTF_NAME = "constructs.gtf"
REJECTIONS_NAME = "rejections.tsv"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no such file: {value}")
    return path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="novex",
        description="Build novel coding transcripts from query ORFs, "
                    "a reference annotation, and splice junctions.",
    )
    parser.add_argument("-q", "--query", required=True, type=existing_file, help="GTF/GFF of query ORFs (CDS rows)")
    parser.add_argument("-r", "--reference", required=True, type=existing_file, help="reference annotation (GTF/GFF)")
    parser.add_argument("-j", "--junctions", required=True, type=existing_file, help="splice junctions (BED)")
    parser.add_argument("-g", "--genome", required=True, type=existing_file, help="genome FASTA")
    parser.add_argument("-o", "--out-dir", required=True, type=Path, help="directory for output files")
    parser.add_argument(
        "-d", "--direction", choices=["up", "down"], default="up",
        help="where the query ORF sits relative to the reference CDS: "
             "up = upstream (query provides the 5' end), down = downstream (query provides the 3' end)",
    )
    parser.add_argument("--fmt", choices=["gtf", "gff"], default="gtf", help="annotation format")
    parser.add_argument(
        "--motifs", type=parse_motifs, default="canonical",
        help="canonical | gtag | u12 | any | explicit pairs, e.g. GT-AG,AT-AC",
    )
    parser.add_argument(
        "--min-cds-ratio", type=float, default=0.0,
        help="reject constructs whose CDS is shorter than this fraction of the "
             "reference transcript's CDS (0.0 = no filter)",
    )
    parser.add_argument(
        "--starts", type=parse_starts, default="atg",
        help="atg | near-cognate | any | explicit codons, e.g. ATG,CTG",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fa = load_genome(args.genome)
    fetch = partial(fetch_chain_seq, fa)

    prelim_queries = read_queries(args.query, fmt=args.fmt)

    if args.direction == "up":
        queries = filter_upstream_by_start(fa, prelim_queries, args.starts)
        log(f"queries | {len(queries)}/{len(prelim_queries)} with a valid start codon")
    else:
        queries = prelim_queries
        log(f"queries | {len(queries)} (start codons come from the reference)")

    prelim_juncs = list(read_bed(args.junctions))
    juncs = filter_juncs_by_motif(fa, prelim_juncs, args.motifs)
    log(f"junctions | {len(juncs)}/{len(prelim_juncs)} with an allowed splice motif")

    prelim_references = read_references(args.reference, fmt=args.fmt)

    if args.direction == "up":
        prelim_references = normalize_all(fetch, prelim_references)
        references = drop_partial(prelim_references)
        log(f"references | {len(references)}/{len(prelim_references)} with a stop codon")
    else:
        references = filter_reference_by_start(fa, prelim_references, frozenset({START_CODON}))
        log(f"references | {len(references)}/{len(prelim_references)} with a start codon")

    if args.direction == "up":
        constructs, rejections = build_all_upstream(
            fetch, queries, JunctionIndex(juncs), CdsIndex(references), args.min_cds_ratio
        )
    else:
        constructs, rejections = build_all_downstream(
            fetch, queries, JunctionIndex(juncs), CdsIndex(references), args.min_cds_ratio
        )

    counts = {reason: 0 for reason in RejectReason}
    for r in rejections:
        counts[r.reason] += 1
    log(f"constructs | {len(constructs)} built from {len(constructs) + len(rejections)} candidates")
    for reason, n in counts.items():
        if n:
            log(f"  rejected, {reason}: {n}")

    gtf_path = args.out_dir / GTF_NAME
    write_gtf(gtf_path, constructs)
    write_rejections(args.out_dir / REJECTIONS_NAME, rejections)
    log(f"wrote {gtf_path} and {args.out_dir / REJECTIONS_NAME}")



if __name__ == "__main__":
    main()
