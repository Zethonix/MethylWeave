"""Exercise the data-processing CLIs with synthetic CX, CpG and eventalign rows."""
import argparse
import csv
import gzip
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def write_fixture(folder, contig):
    reference = ["A"] * 700
    cx, raw = [], []
    events = ["contig\tposition\treference_kmer\tread_name\tevent_index\tsamples"]
    for i in range(22):
        position = 12 + 30 * i
        strand = "+" if i % 2 == 0 else "-"
        if strand == "+":
            reference[position:position+2] = "CG"
        else:
            reference[position-1:position+1] = "CG"
        if i < 20:
            label = int(i < 10)
            met, unm = 20 * label, 20 * (1 - label)
            cx.append(f"{contig}\t{position}\t{strand}\t{met}\t{unm}\tCG\tCGA")
            raw.append(f"chr1\t{position}\t{position+1}\t{100*label}\t{met}\t{unm}\t20\t{label}")
        for k in range(13):
            p = position - 6 + k
            idx = k if strand == "+" else 12 - k
            kmer = "".join(reference[p:p+5])
            events.append(f"{contig}\t{p}\t{kmer}\tsynthetic_{i:02}\t{idx}\t{k},{k+1}")
    cx_file = folder / "synthetic.CX_report.txt"
    cx_file.write_text("\n".join(cx) + "\n")
    ref_file = folder / "synthetic.fasta"
    ref_file.write_text(f">{contig}\n{''.join(reference)}\n")
    input_cpg = folder / "synthetic_cpg.tsv"
    input_cpg.write_text("\n".join(raw) + "\n")
    event_file = folder / "synthetic_events.tsv"
    event_file.write_text("\n".join(events) + "\n")
    chrom_map = folder / "synthetic_chrom_map.tsv"
    chrom_map.write_text(f"chr1\t{contig}\n")
    return cx_file, ref_file, input_cpg, event_file, chrom_map


def count_and_check(file):
    with gzip.open(file, "rt") as handle:
        rows = [line.rstrip("\n").split("\t") for line in handle]
    for r in rows:
        assert len(r) == 13 and len(r[6]) == 13 and r[6][6] == "C"
        assert all(len(r[i].split(",")) == 13 for i in (7, 8, 9))
        raw = r[10].split(";")
        assert len(raw) == 13 and all(len(x.split(",")) == 16 for x in raw)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="New directory for fixtures and results")
    args = parser.parse_args()
    args.output_dir.mkdir()
    summary = []
    for dataset, chrom in (("arabidopsis", "NC_003_SYNTHETIC"), ("rice", "1"), ("hg002", "SYNTHETIC_REFSEQ")):
        folder = args.output_dir / dataset
        folder.mkdir()
        cx, ref, cpg, events, chrom_map = write_fixture(folder, chrom)
        labels = folder / "labels"
        if dataset == "hg002":
            run("bsseq/build_hg002_labels.py", cpg, chrom_map, ref, labels)
        else:
            inputs = [cx]
            if dataset == "arabidopsis":
                for i in (2, 3):
                    rep = folder / f"synthetic_rep{i}.CX_report.txt"
                    rep.write_bytes(cx.read_bytes())
                    inputs.append(rep)
            run("bsseq/build_plant_labels.py", dataset, "--cx", *inputs, "--output-dir", labels)
        label_opts = []
        for ctx in ("cg", "chg", "chh"):
            for y in ("pos", "neg"):
                path = labels / f"{ctx.upper()}.{y}.tsv"
                label_opts.extend((f"--{ctx}_{y}", str(path if path.exists() else "/dev/null")))
        output = folder / "labeled.tsv.gz"
        unlabeled = folder / "unlabeled.tsv.gz"
        script = "extract_hg002_features.py" if dataset == "hg002" else "extract_plant_features.py"
        options = ["--out_unlabelled", str(unlabeled)] if dataset != "hg002" else []
        with events.open() as handle:
            subprocess.run([sys.executable, str(ROOT / "nanopore" / script), *label_opts,
                            "--out", str(output), "--p", "1", *options], stdin=handle, check=True)
        rows = count_and_check(output)
        assert len(rows) == 20 and sum(row[11] == "1" for row in rows) == 10
        assert {row[2] for row in rows} == {"+", "-"}
        other = count_and_check(unlabeled) if unlabeled.exists() else []
        assert len(other) == (0 if dataset == "hg002" else 2)
        assert all(row[11] == "-1" for row in other)
        split = folder / "split"
        run("dataset/split_811.py", output, split)
        sizes = [len((split / f"{s}_811.tsv").read_text().splitlines()) for s in ("train", "val", "test")]
        assert sizes == [16, 2, 2]
        summary.append([dataset, "synthetic", len(rows), 10, len(other), *sizes])
    with (args.output_dir / "summary.tsv").open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["workflow", "input_origin", "labeled", "positive", "unlabeled", "train", "val", "test"])
        writer.writerows(summary)
    print(f"Synthetic example passed: {args.output_dir / 'summary.tsv'}")


if __name__ == "__main__":
    main()
