# Reproducibility

This guide collects the manuscript-reported experimental settings and evaluation protocols. Data provenance and feature construction are described in [Data Processing](../data_processing/README.md); the learning objective and inference behavior are described in [Model](../model/README.md). These protocols have not been independently reproduced from the server implementation.

## Dataset Partitions

Evaluation uses individual target-site records. Counts below are retained from **Supplementary Table S1**; context counts include both methylated and unmethylated records. Positive (%) is 100 × methylated / records. Preservation of the original class proportions does not imply class-stratified splitting or identical fractions in every partition.

### Random Split

Records were randomly divided **8:1:1** into training, validation and test sets. A random record split does not establish read-level or genomic-site independence across partitions.

| Dataset | Partition | Records | Methylated | Unmethylated | Positive (%) | CpG | CHG | CHH |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *A. thaliana* | Train | 12,126,568 | 156,042 | 11,970,526 | 1.29 | 1,441,628 | 1,790,389 | 8,894,551 |
| *A. thaliana* | Validation | 1,515,821 | 19,384 | 1,496,437 | 1.28 | 179,844 | 224,441 | 1,111,536 |
| *A. thaliana* | Test | 1,515,821 | 19,524 | 1,496,297 | 1.29 | 179,499 | 223,738 | 1,112,584 |
| *O. sativa* | Train | 30,349,198 | 3,670,239 | 26,678,959 | 12.09 | 6,452,301 | 4,460,195 | 19,436,702 |
| *O. sativa* | Validation | 3,793,650 | 457,465 | 3,336,185 | 12.06 | 805,988 | 556,703 | 2,430,959 |
| *O. sativa* | Test | 3,793,650 | 458,707 | 3,334,943 | 12.09 | 806,708 | 556,733 | 2,430,209 |
| *H. sapiens* (HG002) | Train | 1,657,330 | 560,625 | 1,096,705 | 33.83 | 1,657,330 | Not evaluated | Not evaluated |
| *H. sapiens* (HG002) | Validation | 207,166 | 70,344 | 136,822 | 33.96 | 207,166 | Not evaluated | Not evaluated |
| *H. sapiens* (HG002) | Test | 207,167 | 69,700 | 137,467 | 33.64 | 207,167 | Not evaluated | Not evaluated |

### Chromosome-held-out Split

Supplementary Note 1 reports mutually exclusive chromosome assignments and cross-partition read filtering:

| Dataset | Training chromosomes | Validation chromosomes | Test chromosomes |
| --- | --- | --- | --- |
| *A. thaliana* | Chr1, Chr2, Chr4 | Chr3 | Chr5 |
| *O. sativa* | Chr1–3, Chr6–12 | Chr4 | Chr5 |
| *H. sapiens* (HG002) | chr1–4, chr6–8, chr10–12, chr14, chr16–17, chr19–21, chrX | chr5, chr18, chr22, chrY | chr9, chr13, chr15 |

These datasets were constructed separately from the random partitions. The manuscript reports no shared chromosomes or genomic sites between the chromosome-held-out partitions; the actual read-filtering implementation remains unverified.

| Dataset | Partition | Records | Methylated | Unmethylated | Positive (%) | CpG | CHG | CHH |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *A. thaliana* | Train | 9,108,138 | 115,403 | 8,992,735 | 1.27 | 1,075,912 | 1,341,254 | 6,690,972 |
| *A. thaliana* | Validation | 3,008,873 | 41,372 | 2,967,501 | 1.37 | 362,035 | 446,095 | 2,200,743 |
| *A. thaliana* | Test | 2,945,804 | 36,680 | 2,909,124 | 1.25 | 351,969 | 437,143 | 2,156,692 |
| *O. sativa* | Train | 31,357,799 | 3,764,980 | 27,592,819 | 12.01 | 6,636,861 | 4,609,044 | 20,111,894 |
| *O. sativa* | Validation | 3,402,502 | 428,691 | 2,973,811 | 12.60 | 746,791 | 495,406 | 2,160,305 |
| *O. sativa* | Test | 2,881,111 | 351,257 | 2,529,854 | 12.19 | 616,692 | 426,772 | 1,837,647 |
| *H. sapiens* (HG002) | Train | 1,655,883 | 559,987 | 1,095,896 | 33.82 | 1,655,883 | Not evaluated | Not evaluated |
| *H. sapiens* (HG002) | Validation | 205,111 | 69,383 | 135,728 | 33.83 | 205,111 | Not evaluated | Not evaluated |
| *H. sapiens* (HG002) | Test | 210,669 | 71,299 | 139,370 | 33.84 | 210,669 | Not evaluated | Not evaluated |

## Training and Model Selection

The following settings are reported in manuscript Section 2.6 and describe the experimental context, not an executable environment specification.

| Item | Reported setting |
| --- | --- |
| Framework and hardware | PyTorch; two NVIDIA RTX 3090 GPUs |
| Optimizer | AdamW; initial learning rate 2 × 10⁻⁴; weight decay 0.01 |
| Schedule and precision | Cosine annealing; FP16 mixed precision |
| Batch size | 512 per GPU; 1,024 globally |
| Maximum duration | 30 epochs |
| Contrastive temperature | 0.50 |
| Base sequence-context masking probability | 0.15 |
| Model selection | Highest validation AUPRC, evaluated after each epoch |
| Early stopping | Four consecutive epochs without validation AUPRC improvement |

The joint-loss coefficient is not specified here because its version correspondence remains unresolved. Exact random-seed values, software environment versions and checkpoint-to-experiment mappings also require verification before an executable reproduction protocol can be established.

The six comparators are MethyNano, NanoCon, Remora, DeepPlant, DeepBAM and DeepMod2. The manuscript reports retraining them on common training, validation and test splits, retaining original class proportions and tuning hyperparameters on their respective validation sets.

## Metrics and Replication

**AUPRC** is the primary metric. AUROC, precision, recall, F1-score and accuracy are also reported. Standard threshold-dependent metrics use a fixed decision threshold of **0.5**.

The main benchmark reports the mean and **sample standard deviation across three random seeds**. This repetition scheme is not assumed for every ablation or transfer analysis. Noise repetitions in the perturbation protocol below are distinct from independently trained models.

| Quantity | Definition |
| --- | --- |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| FPR | FP / (FP + TN) |
| False discovery rate | FP / (TP + FP) |

TP, FP, TN and FN denote true positives, false positives, true negatives and false negatives. FPR measures errors among truly unmethylated records; it is not the false-positive fraction among predicted methylated records.

## Evaluation Tasks

| Analysis | Reported design |
| --- | --- |
| Ablations | Remove sequence information, raw current or the contrastive configuration, or restrict cross-modal interactions to corresponding positions |
| Cross-species transfer | Apply a source-species model directly to another species' test set, without target-species adaptation |
| Methylation contexts | Evaluate unified plant models separately in CpG, CHG and CHH, and compare with separately trained context-specific models; HG002 uses CpG |
| Error characteristics | Compare signal features of misclassified and correctly classified records matched by central 5-mer |

Plant-to-plant transfer uses R9.4 data on both sides; transfer involving HG002 changes both species and sequencing conditions (R10.4.1). These comparisons do not isolate a species-only effect. CHH positive fractions in the random test subsets are **0.0721%** for *A. thaliana* and **0.120%** for *O. sativa*, distinct from overall dataset prevalence.

## Low-FPR and Candidate-budget Evaluation

Supplementary Note 3 analyzes fixed plant test sets. A record is classified as methylated when its probability is at least the selected threshold.

| Constraint | Threshold rule | Reported quantity |
| --- | --- | --- |
| Recall ≥ 0.90 | Highest threshold satisfying the recall requirement | False positives per 10,000 truly unmethylated records |
| FPR ≤ 0.001 | Lowest threshold satisfying the FPR limit | Recall of methylated records |

Thresholds are selected using **test predictions and test labels**, separately within each test context for context-stratified analyses. These are retrospective operating points, not validation-calibrated thresholds or guarantees of FPR control on new data. Precision and false discovery rate summarize the composition of positive predictions.

For **Precision@K**, the expected positive count equals test-set size multiplied by validation-set positive prevalence. Candidate budgets are 0.5, 1 and 2 times this estimate and are shared across all models within a dataset. The 1× budgets are **19,384** for *A. thaliana* and **457,465** for *O. sativa*. Predictions are ranked by methylation probability, and Precision@K measures the true-positive fraction among the top K records. Validation prevalence determines the budget; test labels score retrieval, without aggregation by genomic locus.

## Controlled Raw-current Perturbations

Supplementary Note 2 evaluates IID Gaussian and 5-mer-conditioned AR(1) noise. Parameters are estimated separately for each dataset using **20,000 sampled training records**. Perturbation strengths are **0.25, 0.50 and 1.00**, with zero as the unperturbed reference.

Only valid, non-padding raw-current samples are modified; sequence, event statistics and labels remain fixed. The same checkpoints are evaluated without retraining, with **five independent noise realizations** per nonzero condition and AUPRC summarized as mean and sample standard deviation. NanoCon is excluded because its adapted input lacks raw-current features. The AR(1) index follows retained sample order, not uniformly spaced physical acquisition time.
