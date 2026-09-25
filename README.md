# novex

novex constructs novel protein-coding transcripts by splicing upstream or downstream ORFs into
reference coding sequences (CDS), using splice junctions supplied by the user.

## Output

`--out-dir` receives two files:

**`constructs.gtf`** — one `transcript` row per construct, then its `exon` and `CDS` rows.
CDS rows **include the stop codon**, so no separate `stop_codon` feature is written. This
differs from GENCODE/Ensembl GTF, where CDS rows exclude it. Attributes:

| attribute | meaning |
|---|---|
| `transcript_id` | construct id; `ust_*` from `-d up`, `dst_*` from `-d down`, so runs in both directions can be concatenated |
| `gene_id` | the reference transcript's gene |
| `query_id` | the query ORF it was built from |
| `reference_id` | the reference transcript it splices into |
| `junction` | the intron, `chrom:start-end` |
| `junction_name`, `junction_score` | columns 4 and 5 of the junction BED, `.` if absent |
| `cds_len`, `ref_cds_len`, `cds_ratio` | construct CDS length, reference CDS length, and their ratio — what `--min-cds-ratio` thresholds on |

**`rejections.tsv`** — one row per candidate that was not built, with the reason:
`not_transcribed` (no isoform of the reference transcribes the query ORF), `frame`,
`no_stop`, `ptc` (premature stop), `length` (below `--min-cds-ratio`), `duplicate`
(same CDS chain as a construct already built).

To get sequences, run [gffread](https://github.com/gpertea/gffread) on the GTF:

    gffread -w constructs.nt.fa -g <genome> constructs.gtf
    gffread -S -y constructs.pt.fa -g <genome> constructs.gtf

## Coordinate convention

novex reports genomic positions as **1-based with inclusive ends**, matching GTF/GFF.
Junction BED input is 0-based half-open, so the `junction` attribute does not read the
same as the BED line it came from: a junction at `chr11 802396 804093` is written
`chr11:802397-804093`. Both describe the same intron. Use `junction_name` to match a
construct back to its BED record.
