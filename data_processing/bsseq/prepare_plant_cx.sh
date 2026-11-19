#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: bash prepare_plant_cx.sh <arabidopsis|rice> <R1.fq> <R2.fq> <prepared_bismark_genome_dir> <new_output_dir>"
    echo "For Arabidopsis replicate 2, set BISMARK_PARALLEL=40. Rice takes pooled mates."
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
if [[ $# != 5 ]]; then usage >&2; exit 2; fi
dataset=$1
[[ -f $2 && -f $3 && -d $4 && -d $4/Bisulfite_Genome ]] || {
    echo "Provide both mates and a prepared Bismark genome directory" >&2; exit 2;
}
r1=$(realpath -- "$2")
r2=$(realpath -- "$3")
ref=$(realpath -- "$4")
for tool in bismark samtools picard bismark_methylation_extractor; do
    command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 2; }
done
case "$dataset" in
    arabidopsis)
        qc=cutadapt; jobs=${BISMARK_PARALLEL:-20}; sort_threads=$jobs; bme_jobs=22; buffer=40G
        sort_memory=()
        ;;
    rice)
        qc=fastp; jobs=10; sort_threads=6; bme_jobs=6; buffer=10G
        sort_memory=(-m 1500M)
        ;;
    *) usage >&2; exit 2 ;;
esac
command -v "$qc" >/dev/null || { echo "Missing tool: $qc" >&2; exit 2; }
mkdir -- "$5"
cd -- "$5"
if [[ $qc == cutadapt ]]; then
    cutadapt -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC \
        -A AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT --max-n 0.1 -m 60 --trim-n \
        -q 15,15 -j 20 -o R1.cut.fastq -p R2.cut.fastq "$r1" "$r2" > cutadapt.log 2>&1
else
    fastp -i "$r1" -I "$r2" -o R1.cut.fastq -O R2.cut.fastq -q 15 -l 60 \
        --detect_adapter_for_pe --trim_poly_g --thread 16 -h fastp.html -j fastp.json > fastp.log 2>&1
fi
bismark --parallel "$jobs" -o . --temp_dir . --bowtie2 -N 1 -L 30 -most_valid_alignments 3 \
    --genome "$ref" -1 R1.cut.fastq -2 R2.cut.fastq > bismark.log 2>&1
samtools sort -@ "$sort_threads" "${sort_memory[@]}" -o coordinate.bam R1.cut_bismark_bt2_pe.bam
samtools index -@ "$sort_threads" coordinate.bam
picard MarkDuplicates I=coordinate.bam O=deduplicated.bam M=duplicate_metrics.txt \
    REMOVE_DUPLICATES=true > picard.log 2>&1
if [[ $dataset == arabidopsis ]]; then sort_threads=20; fi
samtools sort -n -@ "$sort_threads" "${sort_memory[@]}" -o target.bam deduplicated.bam
bismark_methylation_extractor -p --no_overlap --ignore 4 --ignore_r2 4 --comprehensive \
    --parallel "$bme_jobs" --bedGraph --cytosine_report --zero_based --CX \
    --buffer_size "$buffer" --genome_folder "$ref" target.bam > methylation_extractor.log 2>&1
test -s target.CX_report.txt
