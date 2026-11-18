import argparse
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description="Map strand-aware HG002 CpG labels to GRCh38 RefSeq contigs.")
parser.add_argument("label_raw", help="Headerless prefiltered 8-column CpG TSV (label in column 8)")
parser.add_argument("chr_map_path", help="Headerless chromosome-to-RefSeq TSV")
parser.add_argument("ref_fasta", help="Matching GRCh38 reference FASTA")
parser.add_argument("output_dir", type=Path, help="New output directory")
args = parser.parse_args()
label_raw, chr_map_path, ref_fasta = args.label_raw, args.chr_map_path, args.ref_fasta
args.output_dir.mkdir()
out_path = args.output_dir / "CpG.labels.tsv"
summary_path = args.output_dir / "CpG.summary.txt"

# chr1 -> NC_000001.11
chr_to_ref = {}
with open(chr_map_path) as f:
    for line in f:
        if not line.strip():
            continue
        chrom, refseq = line.rstrip("\n").split("\t")[:2]
        chr_to_ref[chrom] = refseq

wanted = set(chr_to_ref.values())

print(f"[1/3] Loading primary reference chromosomes: {len(wanted)} contigs")

seqs = {}
current = None
buf = []

with open(ref_fasta) as f:
    for line in f:
        if line.startswith(">"):
            if current is not None:
                seqs[current] = "".join(buf).upper()
                print(f"  loaded {current}: {len(seqs[current]):,} bp")
            name = line[1:].split()[0]
            if name in wanted:
                current = name
                buf = []
            else:
                current = None
                buf = []
        else:
            if current is not None:
                buf.append(line.strip())

if current is not None:
    seqs[current] = "".join(buf).upper()
    print(f"  loaded {current}: {len(seqs[current]):,} bp")

missing = wanted - set(seqs)
if missing:
    raise RuntimeError(f"Missing reference contigs: {sorted(missing)}")

print("[2/3] Inferring CpG strand from reference sequence")

stats = Counter()
bad_examples = []

with (
    open(label_raw) as inp,
    open(out_path, "x") as out,
    (args.output_dir / "CG.pos.tsv").open("x") as pos_out,
    (args.output_dir / "CG.neg.tsv").open("x") as neg_out,
):
    for line in inp:
        if not line.strip():
            continue

        parts = line.rstrip("\n").split("\t")
        chrom = parts[0]
        start = int(parts[1])
        label = parts[7]

        stats["raw_rows"] += 1

        ref = chr_to_ref.get(chrom)
        if ref is None:
            stats["skip_no_chr_map"] += 1
            continue

        seq = seqs[ref]
        if start < 0 or start >= len(seq):
            stats["skip_out_of_range"] += 1
            continue

        base = seq[start]
        prev_base = seq[start - 1] if start > 0 else "N"
        next_base = seq[start + 1] if start + 1 < len(seq) else "N"

        # Forward-strand CpG cytosine: C followed by G
        if base == "C" and next_base == "G":
            strand = "+"
            pos = start
            stats["forward_CpG"] += 1

        # Reverse-strand CpG cytosine: reference base is G preceded by C
        elif base == "G" and prev_base == "C":
            strand = "-"
            pos = start
            stats["reverse_CpG"] += 1

        else:
            stats["skip_non_CpG_context"] += 1
            if len(bad_examples) < 20:
                bad_examples.append((chrom, ref, start, prev_base + base + next_base, label))
            continue

        row = f"{ref}\t{pos}\t{strand}\t{label}\n"
        out.write(row)
        if label == "1":
            pos_out.write(row)
        elif label == "0":
            neg_out.write(row)
        stats["emitted"] += 1
        stats[f"label_{label}"] += 1

print("[3/3] Writing summary")

with open(summary_path, "w") as s:
    for k in sorted(stats):
        s.write(f"{k}\t{stats[k]}\n")

    if bad_examples:
        s.write("\n# bad_examples: chrom refseq start prev_base+base+next_base label\n")
        for x in bad_examples:
            s.write("\t".join(map(str, x)) + "\n")

print("Summary:")
for k in sorted(stats):
    print(f"{k}\t{stats[k]}")

if bad_examples:
    print("Bad examples:")
    for x in bad_examples[:10]:
        print(x)

print(f"Output: {out_path}")
print(f"Summary: {summary_path}")
