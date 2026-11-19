# MethylWeave data processing

This directory builds labeled Nanopore target-site tables from reference methylation calls and event-aligned current signals. Commands below run from `data_processing/` and write to new output directories.

| Dataset | Reference | Signal cohort |
|---|---|---|
| *Arabidopsis thaliana* | TAIR10.1 (GCF_000001735.4) | R9.4, 20,000 selected read IDs |
| *Oryza sativa* | IRGSP-1.0 | R9.4, 20,000 selected read IDs |
| HG002 | GRCh38 (GCF_000001405.26) | R10.4.1, 200,000 reads selected by raw signal length from the 240,000-read candidate POD5 pool |

## Installation

On Linux with Bash and GNU coreutils, create the Python environment:

```bash
conda env create -f environment.yml
conda activate methylweave-data
python3 -c 'import sys, numpy; print(sys.version.split()[0], numpy.__version__)'
```

The environment pins Python 3.10.19 and NumPy 2.2.6. It is sufficient for the synthetic example and for working with prepared CX and eventalign inputs. For the upstream workflows, install the following tools separately and expose them on `PATH`:

| Stage | Tool versions |
|---|---|
| Arabidopsis BS-seq | Cutadapt 5.2, Bismark 0.25.1 with Bowtie2 2.5.4, SAMtools 1.20, Picard 2.18.29 (Java runtime) |
| Rice BS-seq | fastp 1.1.0, Bismark 0.25.1 with Bowtie2 2.5.5, SAMtools 1.22.1, Picard 2.18.29 (Java runtime) |
| Signal preprocessing | f5c 1.4 and SAMtools; the alignment inputs were prepared with minimap2 2.24-r1122 and SAMtools 1.13 |
| Basecalling, if needed | Guppy 6.5.7 for the plant cohorts; Dorado 0.5.3 for HG002 |

Bismark 0.25.1 also requires Perl. Check tool availability in the active shell before processing. Only the stages you run require their corresponding executables.

For example, add [f5c 1.4](https://bioconda.github.io/recipes/f5c/README.html) to the active environment before signal processing:

```bash
conda install --strict-channel-priority -c conda-forge -c bioconda f5c=1.4
f5c --version
```

## Reference labels

For each Arabidopsis Col-0 replicate, pass the paired FASTQs and a prepared Bismark genome directory to the CX workflow. For rice, pool run mates by read direction before processing and pass a single paired input. The workflow uses Cutadapt 5.2 for Arabidopsis and fastp 1.1.0 for rice, followed by Bismark v0.25.1 alignment, duplicate removal, name sorting, and zero-based CX reporting.

```bash
bash bsseq/prepare_plant_cx.sh arabidopsis col0_r1_R1.fastq.gz col0_r1_R2.fastq.gz TAIR10_bismark arab_r1_cx
BISMARK_PARALLEL=40 bash bsseq/prepare_plant_cx.sh arabidopsis col0_r2_R1.fastq.gz col0_r2_R2.fastq.gz TAIR10_bismark arab_r2_cx
bash bsseq/prepare_plant_cx.sh arabidopsis col0_r3_R1.fastq.gz col0_r3_R2.fastq.gz TAIR10_bismark arab_r3_cx
bash bsseq/prepare_plant_cx.sh rice rice_R1.fastq.gz rice_R2.fastq.gz IRGSP_bismark rice_cx

python3 bsseq/build_plant_labels.py arabidopsis --output-dir arab_labels \
  --cx arab_r1_cx/target.CX_report.txt arab_r2_cx/target.CX_report.txt arab_r3_cx/target.CX_report.txt
python3 bsseq/build_plant_labels.py rice --output-dir rice_labels --cx rice_cx/target.CX_report.txt
```

Plant calls require coverage of at least five. Positive sites have a methylated fraction of at least 0.9 and negative sites have a fraction of zero. For Arabidopsis, positive CG calls and all negative calls are intersected across replicates; positive CHG and CHH calls are united. Rice calls use the pooled CX report. Output files are named `{CG,CHG,CHH}.{pos,neg}.tsv` and keyed by contig, zero-based coordinate, and strand. Plant label selection follows the rules used by [plant_5mC_analysis](https://github.com/PengNi/plant_5mC_analysis).

For HG002, supply a headerless, prefiltered CpG table with columns `chrom, start, end, methylation_percent, methylated, unmethylated, coverage, label`. The high-confidence input uses coverage ≥20, positive fraction ≥0.99 and negative fraction ≤0.02. Supply a two-column chromosome-to-RefSeq map and the matching GRCh38 FASTA. The script maps coordinates, infers the strand from the CpG sequence, and writes separate positive and negative lists.

```bash
python3 bsseq/build_hg002_labels.py hg002_highconf.tsv chr_to_refseq.tsv GRCh38.fasta hg002_labels
```

## Signal processing and features

The plant cohorts use Guppy 6.5.7 with the R9.4.1 HAC configuration, and `minimap2 -ax map-ont` against the matching reference. Supply the selected cohort BLOW5, original ordered read-ID list, basecalled FASTQ, and coordinate-sorted BAM. For HG002, use the selected POD5 cohort with Dorado 0.5.3 and `dna_r10.4.1_e8.2_400bps_hac@v4.3.0`. Supply its converted BLOW5, unsplit FASTQ and primary coordinate-sorted BAM against GRCh38. Input signal files, cohort manifests and reference labels are not bundled.

On local working copies, index each FASTQ with its associated BLOW5. Then run CPU event alignment and feature extraction. The plant driver uses 5,000-ID batches and writes separate labeled and unlabeled tables; HG002 uses CpG labels and writes only the labeled table.

```bash
f5c index --slow5 arab_20k.blow5 arab.fastq
f5c index --slow5 rice_20k.blow5 rice.fastq
f5c index -t 16 --slow5 hg002_200k.blow5 hg002_unsplit.fastq

bash nanopore/run_event_features.sh arabidopsis arab_20k.blow5 arab.fastq arab.bam \
  TAIR10.fasta arab_labels arab_features arab_20k_ids.txt
bash nanopore/run_event_features.sh rice rice_20k.blow5 rice.fastq rice.bam \
  IRGSP.fasta rice_labels rice_features rice_20k_ids.txt
bash nanopore/run_event_features.sh hg002 hg002_200k.blow5 hg002_unsplit.fastq hg002_primary.bam \
  GRCh38.fasta hg002_labels hg002_features
```

The extractor joins labels by contig, coordinate and strand. It constructs a strand-oriented, cytosine-centered 13-mer and normalizes current using the read median and scaled median absolute deviation. At each position, it concatenates event samples, records the pre-cap sample count, and uses up to 2,000 samples to compute mean and standard deviation. It subsamples to 16 values in original order or center-pads with zeros.

The Python extractors also accept prepared eventalign TSV on standard input. The header provides `contig`, `position`, `reference_kmer`, `read_name`, `event_index` and `samples`; rows for each read must be contiguous. The synthetic example below uses this interface.

## Format and splits

Each headerless row has 13 tab-separated fields:

```text
contig  position  strand  position_copy  read_id  t  sequence_13mer  mean  std  dwell  samples  label  context
```

The mean, standard deviation and dwell vectors form a `[13,3]` input. The current vector contains 13 semicolon-separated blocks of 16 comma-separated values. Labels are 0 or 1 in the benchmark; unlabeled plant rows use −1. For 5-mer tokenization, the model input loader pads the stored 13-mer with two synthetic C bases on both ends to obtain 13 overlapping tokens.

Run the fixed-seed splitter on the labeled feature gzip for each dataset. It shuffles **records** with seed 42, then partitions them into `train_811.tsv`, `val_811.tsv`, and `test_811.tsv` at an approximate 8:1:1 ratio. It does not group by site or read.

The splitter holds all eligible decompressed rows in memory to preserve the shuffle. Budget RAM above the uncompressed TSV size (about 23 GiB for Arabidopsis, 58 GiB for rice, and 3.2 GiB for HG002, before Python and allocator overhead). Do not split chunks independently.

```bash
python3 dataset/split_811.py arab_features/labeled.tsv.gz arab_split
python3 dataset/split_811.py rice_features/labeled.tsv.gz rice_split
python3 dataset/split_811.py hg002_features/labeled.tsv.gz hg002_split
```

## Synthetic example

Run the label builders, feature extractors and splitter on generated CX, CpG and eventalign rows. No raw sequencing files or external aligners are needed.

```bash
python3 examples/run_synthetic_example.py synthetic_run
diff -u examples/expected_summary.tsv synthetic_run/summary.tsv
```

Each workflow produces 20 labeled rows with a 16/2/2 split. The output directory contains the synthetic inputs, label lists, feature gzip files and split TSVs. This example checks the data interfaces, not raw-sequencing preprocessing.
