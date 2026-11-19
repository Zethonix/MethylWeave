# MethylWeave model

This directory provides the main MethylWeave model and its training and prediction interfaces. Prepare target-site features with [data_processing](../data_processing/README.md).

## Model architecture

Multiscale sequence encoding combines base and overlapping 5-mer embeddings with motif context and target-centered attention. The signal encoder integrates event statistics with base-aligned raw-current patches. Bidirectional cross-position attention connects the sequence and signal encodings. Four feature branches (sequence, signal, cross-attention and local event statistics) are concatenated and fused by an MLP. A two-layer bidirectional GRU, representation readout and classification head produce binary methylation logits.

Training uses two stochastic views with signal-guided sequence masking while protecting the target cytosine and its neighboring positions. A projection head supports the existing SupMin objective; EMA scaling balances SupMin and classification cross-entropy. Prediction uses the classifier without masked views or the projection head. Attribution and licenses are retained in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Requirements

Use Python 3.11 and install the dependencies in [requirements.txt](requirements.txt) in a separate environment from data processing. Commands below run from the repository root.

```bash
python -m pip install -r model/requirements.txt
```

The package pins PyTorch 2.4.1. For GPU training, choose its build for your CUDA drivers using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/). The data-processing shell workflows require Linux; model caching supports Linux and Windows.

## Input format

Use the headerless, 13-column feature TSV produced by `data_processing/`. Plain TSV and gzip-compressed TSV are supported. Columns, in order:

```text
contig  position  strand  position_copy  read_id  t
sequence_13mer  mean  std  dwell  samples  label  context
```

The sequence has 13 target-relative positions. The loader pads it with two Cs on each end to obtain 13 overlapping 5-mer tokens. Each event column contains 13 comma-separated values; the sample column contains 13 semicolon-separated groups of 16 comma-separated current values. Dwell values undergo log1p transformation and per-record standardization. The loader uses the supplied context. Inputs must already be in target-strand orientation with normalized current; the loader does not perform alignment or repeat signal normalization.

Training and validation labels are 0 (unmethylated) or 1 (methylated). Prediction also accepts -1 for an unlabeled record. Training and validation must use distinct files.

## Training

Use [config.yaml](config.yaml) for the model architecture and training defaults. The default GPU configuration uses two visible GPUs; pass `--devices 1` for a single GPU.

```bash
python model/train.py \
  --train data/train_811.tsv --val data/val_811.tsv \
  --out runs/methylweave
```

Training uses AdamW with cosine learning-rate decay. Validation AUPRC selects the best checkpoint and controls early stopping. Model initialization, masking and training-data shuffling are stochastic.

For a small CPU run:

```bash
python model/train.py \
  --train data/train_811.tsv --val data/val_811.tsv \
  --out runs/methylweave_cpu --cpu --epochs 1 --batch-size 8 --workers 0
```

Choose a fresh output directory and a batch size that does not produce a one-record training batch. The first run builds a disk-backed feature cache, approximately 1 KB per record. Use `--cache-dir` to share a cache across runs with unchanged input files. The best checkpoint path is printed when training finishes; checkpoints, caches and logs remain local outputs.

## Prediction

Use a checkpoint produced by `train.py`. The following path selects the last training checkpoint; use the printed best checkpoint path when preferred.

```bash
python model/predict.py --checkpoint runs/methylweave/last.ckpt \
  --input data/test_811.tsv --output predictions.tsv \
  --device cpu --threshold 0.5
```

Prediction preserves input order and writes the one-based input row, contig, position, strand, read ID, context, supplied label, 5mC probability and binary call. A record is called methylated when its class-1 probability meets the threshold. The output file must not already exist. Use `--device cuda:0` for GPU prediction.
