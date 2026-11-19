"""Select strand-aware labels directly from zero-based Bismark CX reports."""
import argparse
from pathlib import Path


def call_sites(path, context, keep_contig):
    positive, negative = set(), set()
    with path.open() as f:
        for line in f:
            chrom, pos, strand, methylated, unmethylated, motif, *_ = line.split()
            if motif != context or not keep_contig(chrom):
                continue
            methylated, unmethylated = int(methylated), int(unmethylated)
            coverage = methylated + unmethylated
            if coverage == 0:
                continue
            site = chrom, int(pos), strand
            # A later covered CX row replaces an earlier call for the same key.
            positive.discard(site)
            negative.discard(site)
            if coverage >= 5 and methylated == 0:
                negative.add(site)
            elif coverage >= 5 and methylated / coverage >= 0.9:
                positive.add(site)
    return positive, negative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("arabidopsis", "rice"))
    parser.add_argument("--cx", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arab = args.dataset == "arabidopsis"
    if len(args.cx) != (3 if arab else 1):
        parser.error("Arabidopsis requires three CX reports; rice requires one pooled CX report")
    if len({p.resolve() for p in args.cx}) != len(args.cx):
        parser.error("CX reports must be distinct")
    for path in args.cx:
        if not path.is_file():
            parser.error(f"Missing CX report: {path}")
    rice_contigs = {str(i) for i in range(1, 13)}

    def keep(chrom):
        return chrom.startswith("NC_003") if arab else chrom in rice_contigs

    args.output_dir.mkdir()
    for context in ("CG", "CHG", "CHH"):
        pos = neg = None
        for path in args.cx:
            cur_pos, cur_neg = call_sites(path, context, keep)
            if pos is None:
                pos, neg = cur_pos, cur_neg
            else:
                if context == "CG":
                    pos.intersection_update(cur_pos)
                else:
                    pos.update(cur_pos)
                neg.intersection_update(cur_neg)
        for suffix, sites in (("pos", pos), ("neg", neg)):
            with (args.output_dir / f"{context}.{suffix}.tsv").open("x") as f:
                for chrom, pos0, strand in sorted(sites):
                    print(chrom, pos0, strand, sep="\t", file=f)
            print(context, suffix, len(sites), sep="\t")


if __name__ == "__main__":
    main()
