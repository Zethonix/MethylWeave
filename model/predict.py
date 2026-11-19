"""Predict per-record 5mC probabilities from a final MethylWeave checkpoint."""
import argparse
from contextlib import nullcontext
import csv
from itertools import islice
from pathlib import Path

import torch

if __package__:
    from .dataset import collate, iter_features
    from .methylweave import MethylWeave, load_checkpoint
else:
    from dataset import collate, iter_features
    from methylweave import MethylWeave, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="cpu or a CUDA device such as cuda:0")
    p.add_argument("--precision", type=int, choices=(16, 32), help="Default: CUDA=16, CPU=32")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    device = torch.device(a.device)
    if device.type not in ("cuda", "cpu") or a.batch_size < 1 or not 0 <= a.threshold <= 1:
        p.error("require a CPU/CUDA device, positive batch size and threshold in [0,1]")
    mixed = (a.precision or (16 if device.type == "cuda" else 32)) == 16
    if mixed and device.type != "cuda":
        p.error("FP16 prediction requires CUDA")
    torch.set_float32_matmul_precision("high")
    model = MethylWeave()
    load_checkpoint(model, a.checkpoint)
    model.to(device).eval()
    data = iter_features(a.input)
    with a.output.open("x", newline="") as f, torch.inference_mode():
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["row", "contig", "position", "strand", "read_id", "context",
                         "label", "probability_5mc", "prediction"])
        while True:
            batch = list(islice(data, a.batch_size))
            if not batch:
                break
            identifiers, values = zip(*batch)
            tensors = collate(values)
            with torch.autocast("cuda", dtype=torch.float16) if mixed else nullcontext():
                logits = model.get_logits(*(x.to(device) for x in tensors[:-1]))
                probabilities = logits.softmax(dim=1)[:, 1].cpu().tolist()
            for meta, label, prob in zip(identifiers, tensors[-1].tolist(), probabilities):
                writer.writerow([*meta, label, repr(prob), int(prob >= a.threshold)])


if __name__ == "__main__":
    main()
