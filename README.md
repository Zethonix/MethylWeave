# MethylWeave

**MethylWeave: nanopore 5mC detection with scarce positive samples**

## Overview

MethylWeave detects 5-methylcytosine (5mC) at target cytosines within nanopore reads, with a focus on datasets containing scarce methylation-positive records. It integrates local sequence context and ionic-current evidence to distinguish methylated from unmethylated records.

This package currently contains research documentation. Executable code and model weights are not included; the server-side implementation has not yet been verified for release.

## Method

MethylWeave combines multiscale sequence encoding, event-statistic and raw-current integration, and bidirectional cross-position sequence–signal interaction. Classification is trained jointly with the existing supervised minority contrastive learning (SupMin) strategy of Mildenberger et al. (2025). The [model guide](model/README.md) describes the architecture, training objective and inference behavior, with the original SupMin reference.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── .gitignore
├── data_processing/
│   └── README.md
├── model/
│   └── README.md
└── docs/
    └── reproducibility.md
```

Data Processing constructs model inputs from public sequencing data. The model module consumes those inputs for training and prediction; experimental protocols are documented separately.

## Data Sources

| Dataset | Nanopore input | Reference methylation labels |
| --- | --- | --- |
| *Arabidopsis thaliana* and *Oryza sativa* | R9.4 data from [PRJNA764549 / SRP337810](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA764549) | Plant BS-seq data from the same public project |
| Human *Homo sapiens* HG002 | ONT R10.4.1 [giab_2023.05](https://doi.org/10.5281/zenodo.8363974) | Separate GM24385 BS-seq resource, [gm24385_mod_2021.09](https://doi.org/10.5281/zenodo.6372330) |

## Getting Started

| Task | Entry point |
| --- | --- |
| Trace source data, labeling and feature construction | [Data Processing](data_processing/README.md) |
| Understand the model, joint training and inference | [Model](model/README.md) |
| Check splits, parameters, metrics and evaluation protocols | [Reproducibility](docs/reproducibility.md) |

These guides describe the manuscript-reported workflow. They do not provide verified installation or execution instructions.

## Citation

Tong Zheng, Jinmiao Song, Shengwei Tian, Yazhang Xue and Long Yu. **MethylWeave: nanopore 5mC detection with scarce positive samples.** Manuscript.

Tong Zheng and Jinmiao Song contributed equally. See [CITATION.cff](CITATION.cff) for machine-readable metadata. No publication DOI or formal publication details are assigned.
