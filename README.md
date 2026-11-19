# MethylWeave

**MethylWeave: nanopore 5mC detection with scarce positive samples**

## Project overview

MethylWeave detects 5-methylcytosine (5mC) from nanopore target-site records when methylation-positive examples are scarce. This repository contains the main model, training and prediction entry points, and the data-processing tools needed to construct its sequence and current inputs.

## Method overview

MethylWeave combines multiscale sequence encoding with event statistics and raw-current features. Bidirectional cross-position interactions connect sequence and signal evidence, followed by four-way feature fusion and a bidirectional GRU classifier. Training combines classification with the existing supervised minority contrastive learning (SupMin) strategy using two masked views and EMA loss scaling. See the [model guide](model/README.md) for architecture and usage.

## Repository structure

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
└── model/
    ├── README.md
    ├── methylweave.py
    ├── losses.py
    ├── dataset.py
    ├── train.py
    ├── predict.py
    ├── config.yaml
    ├── requirements.txt
    └── THIRD_PARTY_NOTICES.txt
```

## Installation / Getting started

1. Follow the [data-processing guide](data_processing/README.md) to prepare labeled target-site tables.
2. Install model dependencies and follow the [model guide](model/README.md) to train a model and predict 5mC probabilities.

Use separate environments for data processing and model training: their dependency pins differ. Commands in the model guide run from the repository root; data-processing commands run from `data_processing/`. Raw sequencing data and pretrained weights are not bundled. Keep generated data, checkpoints, logs and predictions outside version control.

Public inputs include plant nanopore and BS-seq data for *Arabidopsis thaliana* and *Oryza sativa* from [PRJNA764549 / SRP337810](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA764549), human HG002 nanopore data from [giab_2023.05](https://doi.org/10.5281/zenodo.8363974), and HG002 reference methylation data from [gm24385_mod_2021.09](https://doi.org/10.5281/zenodo.6372330).

## Citation

Tong Zheng, Jinmiao Song, Shengwei Tian, Yazhang Xue and Long Yu. **MethylWeave: nanopore 5mC detection with scarce positive samples.** Manuscript.

Tong Zheng and Jinmiao Song contributed equally. See [CITATION.cff](CITATION.cff) for citation metadata and [third-party notices](model/THIRD_PARTY_NOTICES.txt) for SupMin attribution and applicable third-party licenses.
