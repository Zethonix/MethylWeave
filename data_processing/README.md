# Data Processing

This guide follows the data flow from public sequencing resources to labeled model inputs, based on manuscript Section 2.1 and Supplementary Note 1. It records the reported method; the actual processing code, file interfaces and execution parameters remain subject to server-side verification.

## Sources and Reference Genomes

Plant nanopore and BS-seq data come from Ni et al. (2021), *Genome-wide detection of cytosine methylations in plant from Nanopore data using deep learning*, under [BioProject PRJNA764549](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA764549) and [SRA SRP337810](https://www.ncbi.nlm.nih.gov/sra?term=SRP337810). Nanopore sequencing used R9.4; reference labels cover CpG, CHG and CHH contexts.

Human HG002 (GM24385) uses two distinct ONT resources:

| Role | Release | Official record |
| --- | --- | --- |
| R10.4.1 nanopore input | `giab_2023.05`, Zenodo v1 | *Sequencing Genome in a Bottle samples*: [DOI](https://doi.org/10.5281/zenodo.8363974), [Zenodo](https://zenodo.org/records/8363974) |
| BS-seq CpG reference frequencies | `gm24385_mod_2021.09`, Zenodo v1; EPI2ME `gm24385-5mc` | *Matched Oxford Nanopore Technologies and Bisulfite Sequencing of the GM24385 Cell Line*: [DOI](https://doi.org/10.5281/zenodo.6372330), [Zenodo](https://zenodo.org/records/6372330) |

The BS-seq resource also contains nanopore data, but the manuscript identifies `giab_2023.05` as MethylWeave's human nanopore input. These releases should remain distinct in data provenance.

| Dataset | Reported reference | Official source |
| --- | --- | --- |
| *Arabidopsis thaliana* | TAIR10.1 (GCF_000001735.4) | [NCBI](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001735.4/) |
| *Oryza sativa* | IRGSP-1.0 | [RAP-DB](https://rapdb.dna.naro.go.jp/download/irgsp1.html) |
| Human *Homo sapiens* HG002 | GRCh38 (GCF_000001405.26) | [NCBI](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/) |

## Reference Label Construction

Plant paired-end BS-seq reads were trimmed with Cutadapt, aligned with Bismark/Bowtie2, deduplicated with Picard and processed with the Bismark methylation extractor. HG002 uses the published BS-seq methylation frequencies from the resource above.

| Dataset | Contexts | Minimum coverage | Methylated | Unmethylated |
| --- | --- | ---: | --- | --- |
| Both plants | CpG, CHG, CHH | 5 | Frequency ≥ 0.9 | Frequency = 0 |
| HG002 | CpG | 20 | Frequency ≥ 0.99 | Frequency ≤ 0.02 |

Sites outside these coverage and frequency criteria are excluded from the high-confidence label sets. For three *A. thaliana* BS-seq replicates, CpG methylated and unmethylated sets each use the intersection. For CHG and CHH, methylated sets use the union and unmethylated sets the intersection. The same preprocessing and confidence criteria are reported for *O. sativa*, without a separately specified rice replicate-integration rule.

## Nanopore Processing

| Stage | Manuscript-reported operation |
| --- | --- |
| Basecalling | Guppy high-accuracy mode for plants, with `dna_r9.4.1_450bps_hac_prom.cfg`; Dorado v0.5.3 for HG002 |
| Reference alignment | minimap2 alignment to the species-specific reference above |
| Event alignment | f5c v1.4 maps current events to reference positions; the plant workflow explicitly includes FAST5-to-BLOW5 conversion |
| Normalization | Median-based current normalization with 1.4826 × median absolute deviation (MAD) as the robust scale |
| Current feature extraction | Event-current mean, standard deviation and duration, plus 16 retained current samples per position |

For events longer than 16 samples, random subsampling preserves the original sample order; shorter events are zero-padded. The retained samples do not imply a uniform physical sampling interval. Negative-strand sequences are reverse-complemented and corresponding signal features reordered into a common target-relative orientation.

The manuscript does not fully specify HG002 intermediate-container handling, normalization scope or zero-scale behavior. This guide therefore does not establish those implementation choices.

## Model Input Representation

Each record describes a target cytosine within a nanopore read. Six positions on either side of the target give 13 represented positions, from −6 to +6. Two additional reference nucleotides on each side extend the sequence span to 17 nt (−8 to +8), allowing one centered 5-mer token at each represented position.

| Component | Per-record representation |
| --- | --- |
| Sequence | 13 overlapping 5-mer tokens derived from the 17-nt reference span |
| Event statistics | 13 × 3: current mean, standard deviation and duration |
| Raw current | 13 × 16: retained normalized samples, with padding for short events |
| Label | Binary reference methylation state of the target cytosine |

Sequence and signal features share the same 13-position axis; the additional sequence flanks do not add signal positions. Labels are assigned by matching **chromosome, genomic coordinate and strand** to high-confidence BS-seq sites. This transfers site-level reference evidence to nanopore records, rather than measuring the methylation state of each nanopore molecule directly.

These dimensions define the scientific interface consumed by the [model](../model/README.md). They do not specify file extensions, column names, token IDs, numeric class encoding or serialization.

## Dataset Construction and Handoff

For random-benchmark construction, the manuscript reports sampling 20,000 nanopore reads per plant species and 200,000 HG002 reads. A read can yield several target-site records, and multiple reads can cover the same genomic site.

| Dataset | Labeled records | Positive records (%) |
| --- | ---: | ---: |
| *A. thaliana* | 15,158,210 | 1.29 |
| *O. sativa* | 37,936,498 | 12.09 |
| HG002 | 2,071,663 | 33.82 |

These are proportions in the constructed, confidence-filtered datasets, not genome-wide methylation frequencies. The study preserves the original class proportions for benchmark comparisons. Random and chromosome-held-out datasets were constructed separately; [Reproducibility](../docs/reproducibility.md#dataset-partitions) contains the split rules, chromosome assignments and complete partition statistics.
