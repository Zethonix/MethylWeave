# MethylWeave

**MethylWeave: nanopore 5mC detection with scarce positive samples**

## Project Overview

MethylWeave is a nanopore sequencing framework for detecting 5-methylcytosine (5mC) at target cytosines when methylation-positive records are scarce. It combines local sequence context with aligned ionic-current evidence, with particular emphasis on cross-position sequence–signal interaction and class-dependent contrastive learning.

The repository is being assembled as a research reproducibility project. The **Data Processing** module is provided with environment metadata, processing scripts and a synthetic interface example. The complete model training and inference implementation has not yet been released.

## Method Overview

MethylWeave represents each target cytosine with 13 target-relative positions, overlapping 5-mer sequence context, event statistics and retained raw-current samples. Its reported model uses multiscale sequence encoding, signal integration, bidirectional cross-position interaction and feature fusion. Classification is trained jointly with the existing supervised minority contrastive learning (SupMin) strategy of Mildenberger et al. (2025), rather than presenting SupMin as a newly proposed theory.

The scientific workflow is:

```text
Public sequencing datasets
        ↓
Data Processing
        ↓
Target-site feature construction
        ↓
Model-ready datasets
        ↓
MethylWeave training and inference
        ↓
Evaluation
```

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── CITATION.cff
├── data_processing/
│   ├── README.md
│   ├── environment.yml
│   ├── bsseq/
│   ├── nanopore/
│   ├── dataset/
│   └── examples/
├── model/
│   └── README.md
└── docs/
    └── reproducibility.md
```

`data_processing/` contains the released data-processing code and its synthetic example. `model/` documents the reported model design and learning procedure; its complete training and inference implementation is not included in this release. `docs/reproducibility.md` records the study's data partitions, parameters, metrics and evaluation protocols.

## Data Sources

The study covers *Arabidopsis thaliana*, *Oryza sativa* and human *Homo sapiens* HG002. Plant nanopore and BS-seq data are associated with [NCBI BioProject PRJNA764549 / SRA SRP337810](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA764549). The HG002 nanopore input is ONT [giab_2023.05](https://doi.org/10.5281/zenodo.8363974); reference CpG methylation frequencies use the separate [gm24385_mod_2021.09](https://doi.org/10.5281/zenodo.6372330) resource. Detailed source provenance and processing interfaces are maintained in [Data Processing](data_processing/README.md).

## Getting Started

- [Data Processing README](data_processing/README.md): environment, scripts, input interfaces and the synthetic example.
- [Model README](model/README.md): model structure, joint training and inference scope.
- [Reproducibility](docs/reproducibility.md): data splits, training settings, metrics and evaluation protocols.

The Data Processing README is the entry point for the released processing scripts. This root README intentionally does not duplicate their command lines. The presence of these scripts does not imply that the complete model pipeline has been validated locally or that model training and inference code are already public.

## Citation

Tong Zheng, Jinmiao Song, Shengwei Tian, Yazhang Xue and Long Yu. **MethylWeave: nanopore 5mC detection with scarce positive samples.** Manuscript.

Tong Zheng and Jinmiao Song contributed equally. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata. No publication DOI or formal publication details are assigned here.
