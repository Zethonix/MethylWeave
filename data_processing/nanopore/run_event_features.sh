#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash run_event_features.sh <arabidopsis|rice|hg002> <indexed.blow5> <indexed.fastq> <aligned.bam> <reference.fasta> <labels_dir> <new_output_dir> [plant_read_ids.txt]"
    echo "Supply the frozen plant read IDs in their original order. HG002 takes its prepared primary BAM and unsplit FASTQ."
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ $# != 7 && $# != 8 ]]; then usage >&2; exit 2; fi
dataset=$1
blow5=$2
fastq=$3
bam=$4
ref=$5
labels=$6
out=$7
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
require_file() {
    [[ -r $1 && -s $1 && -f $1 ]] || { echo "Missing input file: $1" >&2; exit 2; }
}
require_tool() {
    command -v "$1" >/dev/null || { echo "Missing tool: $1" >&2; exit 2; }
}
case "$dataset" in
    arabidopsis|rice) [[ $# == 8 ]] || { usage >&2; exit 2; }; require_file "$8"; require_tool samtools ;;
    hg002) [[ $# == 7 ]] || { usage >&2; exit 2; } ;;
    *) usage >&2; exit 2 ;;
esac
for input in "$blow5" "$fastq" "$bam" "$ref"; do require_file "$input"; done
require_tool f5c
require_tool python3
for input in "$labels/CG.pos.tsv" "$labels/CG.neg.tsv"; do
    [[ -r $input && -f $input ]] || { echo "Missing label file: $input" >&2; exit 2; }
done
cg=(--cg_pos "$labels/CG.pos.tsv" --cg_neg "$labels/CG.neg.tsv")
flags=(--samples --scale-events --print-read-names --disable-cuda=yes)
if [[ $dataset == hg002 ]]; then
    mkdir -- "$out"
    f5c eventalign --slow5 "$blow5" -r "$fastq" -b "$bam" -g "$ref" "${flags[@]}" -K 2M -t 24 \
        2> "$out/eventalign.log" |
        python3 "$here/extract_hg002_features.py" "${cg[@]}" \
            --chg_pos /dev/null --chg_neg /dev/null --chh_pos /dev/null --chh_neg /dev/null \
            --out "$out/labeled.tsv.gz" --p 4 2> "$out/extraction.log"
    exit
fi

for name in CHG.pos CHG.neg CHH.pos CHH.neg; do
    [[ -r $labels/$name.tsv && -f $labels/$name.tsv ]] || {
        echo "Missing label file: $labels/$name.tsv" >&2; exit 2;
    }
done
mkdir -- "$out"
mkdir -- "$out/batches"
split -l 5000 -d -a 4 -- "$8" "$out/batches/ids_"
shopt -s nullglob
batch_ids=("$out"/batches/ids_*)
if [[ ${#batch_ids[@]} == 0 ]]; then echo "No plant read IDs" >&2; exit 2; fi
labeled=()
unlabeled=()
for ids in "${batch_ids[@]}"; do
    prefix=${ids%/*}/batch_${ids##*_}
    samtools view -b -N "$ids" -o "$prefix.bam" "$bam"
    samtools index "$prefix.bam"
    f5c eventalign --slow5 "$blow5" -r "$fastq" -b "$prefix.bam" -g "$ref" --pore r9 \
        "${flags[@]}" -K 512K -t 16 2> "$prefix.eventalign.log" |
        python3 "$here/extract_plant_features.py" "${cg[@]}" \
            --chg_pos "$labels/CHG.pos.tsv" --chg_neg "$labels/CHG.neg.tsv" \
            --chh_pos "$labels/CHH.pos.tsv" --chh_neg "$labels/CHH.neg.tsv" \
            --out "$prefix.labeled.tsv.gz" --out_unlabelled "$prefix.unlabeled.tsv.gz" \
            --p 4 2> "$prefix.extraction.log"
    labeled+=("$prefix.labeled.tsv.gz")
    unlabeled+=("$prefix.unlabeled.tsv.gz")
    rm -- "$prefix.bam" "$prefix.bam.bai"
done
cat -- "${labeled[@]}" > "$out/labeled.tsv.gz"
cat -- "${unlabeled[@]}" > "$out/unlabeled.tsv.gz"
