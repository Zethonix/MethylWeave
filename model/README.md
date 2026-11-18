# Model

MethylWeave consumes the [target-site representation](../data_processing/README.md#model-input-representation) produced by Data Processing. This guide summarizes the architecture and learning procedure in manuscript Sections 2.2–2.5 and Supplementary Tables S2–S6. It describes the reported model, without asserting that its implementation or execution interfaces have been verified.

## Architecture

| Component | Operation | Output per record |
| --- | --- | --- |
| Sequence encoder | Combine 5-mer tokens, nucleotide identity, motif and relative position; integrate convolutions with kernel sizes 1, 3, 5 and 7 using adaptive scale weights and local refinement | 13 × 128 |
| Target-guided sequence context | Use the motif-conditioned target-cytosine representation as an attention query over the sequence window, then gate this context into each position | 13 × 128 |
| Signal encoder | Encode event statistics; extract raw-current features with local convolution and bidirectional gated mixing along the position axis; combine them through a gated residual connection with learnable scaling | 13 × 128 |
| Cross-position attention | Let sequence query signal and signal query sequence, with each query accessing all positions in the other modality | 13 × 13 access per attention head |
| Feature fusion | Integrate sequence, signal, bidirectional cross-modal contexts, and event statistics with local variations into a shared 13 × 128 representation; update both branches and concatenate them | 13 × 256 |
| Target-site representation and classification | Form a 128-dimensional target-site representation and classify it with linear layers, batch normalization and ReLU activations | Two class outputs |

The classifier predicts methylated versus unmethylated status for the target cytosine. A separate two-layer projection head maps the target-site representation to an L2-normalized contrastive representation during training.

## Joint Training

Each record undergoes two stochastic forward passes through the same backbone. Event-current standard deviation guides independent masking of peripheral sequence representations after encoding. The target cytosine and neighboring positions are protected, and current inputs remain identical in the two views.

MethylWeave adopts the existing **supervised minority contrastive learning (SupMin)** strategy of Mildenberger et al., [*A Tale of Two Classes: Adapting Supervised Contrastive Learning to Binary Imbalanced Datasets*, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Mildenberger_A_Tale_of_Two_Classes_Adapting_Supervised_Contrastive_Learning_to_CVPR_2025_paper.html).

| Anchor class | Contrastive positives |
| --- | --- |
| Methylated minority | Its paired view and both views of every other methylated record in the batch |
| Unmethylated majority | Its own paired view only |

The contrastive denominator includes every batch view except the anchor. Cross-entropy is averaged over the two classification predictions. Each loss is normalized using an exponential moving average as its reference scale, and the two normalized objectives are jointly optimized with a relative weighting coefficient. The reported procedure uses joint training, without a separate contrastive-pretraining and classification-fine-tuning sequence.

Optimizer settings, batch size, replication and checkpoint selection are recorded in [Reproducibility](../docs/reproducibility.md#training-and-model-selection).

## Inference

The manuscript reports a single forward pass to obtain a methylation probability for each target-site record. The classification output is used for prediction; the contrastive representation supplies the training constraint. No on-disk prediction schema or inference command is specified here.

The [evaluation protocol](../docs/reproducibility.md#metrics-and-replication) defines fixed-threshold classification metrics. Its [low-FPR analyses](../docs/reproducibility.md#low-fpr-and-candidate-budget-evaluation) select retrospective operating points using test labels; those thresholds are not established deployment settings for new data.
