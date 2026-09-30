#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ATTR = re.compile(r'(\w+) "([^"]*)"')
JUNC = re.compile(r"^(.+):(\d+)-(\d+)$")


def parse_attrs(field: str) -> dict[str, str]:
    return dict(ATTR.findall(field))


def anchor_base(intron: tuple[int, int], strand: str, direction: str) -> int:
    """The junction base the novel piece splice in/out
    """
    start, end = intron
    if (direction == "up") == (strand == "+"):
        return start - 1
    return end + 1


def get_novel_pieces(
    cds: list[tuple[int, int]], intron: tuple[int, int], strand: str, direction: str
) -> list[tuple[int, int]]:
    base = anchor_base(intron, strand, direction)

    if (direction == "up") == (strand == "+"):
        return [x for x in cds if x[1] <= base]
    return [x for x in cds if x[0] >= base]


def trim_stop_codon(
    pieces: list[tuple[int, int]], strand: str, direction: str
) -> list[tuple[int, int]]:
    """Drop the 3 terminal bases of a downstream piece, i.e. its stop codon.
    """
    if direction != "down":
        return pieces

    out = list(pieces)
    left = 3
    while left and out:
        # the query's 3' end: highest coordinate on +, lowest on -
        i = -1 if strand == "+" else 0
        start, end = out[i]
        take = min(left, end - start + 1)
        if take == end - start + 1:
            out.pop(i)
        elif strand == "+":
            out[i] = (start, end - take)
        else:
            out[i] = (start + take, end)
        left -= take

    return out


def read_constructs(path: Path):
    """Yield (transcript_id, query_id, chrom, strand, cds_intervals, junction, score, direction)."""
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    meta: dict[str, tuple[str, str, str, str, str]] = {}

    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "CDS":
                continue
            a = parse_attrs(f[8])
            tid = a["transcript_id"]
            cds[tid].append((int(f[3]), int(f[4])))
            meta[tid] = (
                a["query_id"], f[0], f[6], a["junction"],
                a.get("junction_score", "."),
                # written by novex since Direction was added; "" for older files
                a.get("direction", ""),
            )

    for tid, intervals in cds.items():
        query_id, chrom, strand, junction, score, direction = meta[tid]
        m = JUNC.match(junction)
        if m is None:
            raise ValueError(f"{path}: {tid} has unparsable junction info {junction!r}")
        intron = (int(m.group(2)), int(m.group(3)))
        yield tid, query_id, chrom, strand, sorted(intervals), intron, score, direction


def main() -> None:
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gtfs", nargs="+", type=Path, help="one or more novex-generated constructs.gtf files")
    parser.add_argument("-o", "--prefix", default="candidates", help="output prefix [candidates]")
    parser.add_argument(
        "--no-stop-codon", action="store_true", help=""
    )
    args = parser.parse_args()

    spans: dict[tuple, tuple[str, list[tuple[int, int]]]] = {}
    junctions: dict[tuple, dict[tuple[int, int], float | None]] = defaultdict(dict)
    tracked: dict[tuple, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    stop_only = 0

    for n, path in enumerate(args.gtfs):
        for tid, query_id, chrom, strand, cds, intron, score, attr in read_constructs(path):
            if attr in ("up", "down"):
                direction = attr
            elif tid.startswith("dst"):
                direction = "down"
            elif tid.startswith("ust"):
                direction = "up"
            else:
                raise ValueError(
                    f"{path}: cannot tell which direction built {tid!r}: "
                    f'direction attribute is {attr!r} and the id is neither ust_* nor dst_*'
                )

            pieces = get_novel_pieces(cds, intron, strand, direction)

            if not pieces:
                raise ValueError(f"{path}: {tid} has no novel pieces?")

            if args.no_stop_codon:
                pieces = trim_stop_codon(pieces, strand, direction)
                if not pieces:
                    # the whole novel piece was the stop codon, or part of one
                    stop_only += 1
                    continue

            key = (chrom, strand, tuple(pieces))
            spans[key] = (query_id, pieces)
            value = None if score == "." else float(score)
            prev = junctions[key].get(intron)
            if prev is None or (value is not None and value > prev):
                junctions[key][intron] = value
            tracked[key][n].append(tid)

    by_query: dict[str, list[tuple]] = defaultdict(list)
    for key in spans:
        by_query[spans[key][0]].append(key)

    names: dict[tuple, str] = {}
    for query_id, keys in by_query.items():
        strand = keys[0][1]
        keys.sort(key=lambda k: k[2][0][0], reverse=(strand == "-"))
        for i, key in enumerate(keys):
            names[key] = f"{query_id}_c{i}"

    bed = Path(f"{args.prefix}.bed")
    tracking = Path(f"{args.prefix}.tracking")
    ordered = sorted(spans, key=lambda k: (k[0], k[2][0][0]))

    # strictly BED6: IGV parses columns 7-9 as thickStart/thickEnd/itemRgb, so extra
    # columns there make it drop the feature. The metrics live in the tracking file.
    with open(bed, "w") as fh:
        fh.write("#chrom\tstart\tend\tcandidate\tscore\tstrand\n")
        for key in ordered:
            chrom, strand, piece = key
            for start, end in piece:
                fh.write(f"{chrom}\t{start - 1}\t{end}\t{names[key]}\t.\t{strand}\n")

    with open(tracking, "w") as fh:
        fh.write("# candidates -> constructs\n")
        for n, path in enumerate(args.gtfs, 1):
            fh.write(f"# file_{n} = {path}\n")
        cols = "\t".join(f"file_{n}" for n in range(1, len(args.gtfs) + 1))
        fh.write(
            "#candidate\tchrom\tstart\tend\tstrand\t"
            f"n_junctions\tmax_junction_score\tsum_junction_score\t{cols}\n"
        )
        for key in ordered:
            chrom, strand, piece = key
            found = [v for v in junctions[key].values() if v is not None]
            best = max(found) if found else None
            total = sum(found) if found else None
            cells = [
                ",".join(tracked[key][n]) if tracked[key].get(n) else "."
                for n in range(len(args.gtfs))
            ]
            fh.write(
                f"{names[key]}\t{chrom}\t{piece[0][0] - 1}\t{piece[-1][1]}\t{strand}\t"
                f"{len(junctions[key])}\t{'.' if best is None else f'{best:g}'}\t"
                f"{'.' if total is None else f'{total:g}'}\t"
                + "\t".join(cells)
                + "\n"
            )

    note = f" | {stop_only} construct(s) dropped, stop codon was all they contributed" if stop_only else ""
    print(f"wrote {bed} and {tracking} | {len(spans)} candidates from {len(args.gtfs)} file(s){note}")


if __name__ == "__main__":
    main()
