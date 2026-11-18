"""Create the benchmark's fixed-seed 8:1:1 record split from a feature TSV.gz."""
import argparse
import gzip
from itertools import islice
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_gzip", type=Path)
    parser.add_argument("output_dir", type=Path, help="New directory for the three TSVs")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("The output directory must not already exist")
    rows = []
    with gzip.open(args.feature_gzip, "rt") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) >= 12 and fields[-2] in ("0", "1"):
                rows.append(line)
    random.Random(42).shuffle(rows)
    cuts = (0, int(len(rows) * 0.8), int(len(rows) * 0.9), len(rows))
    args.output_dir.mkdir()
    for name, start, end in zip(("train", "val", "test"), cuts, cuts[1:]):
        with (args.output_dir / f"{name}_811.tsv").open("x") as f:
            f.writelines(islice(rows, start, end))
        print(name, end - start, sep="\t")


if __name__ == "__main__":
    main()
