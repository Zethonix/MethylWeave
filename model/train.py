"""Train the joint CE + SupMin reference model on prepared train/val splits."""
import argparse
import math
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUPRC
import yaml

if __package__:
    from .dataset import FeatureDataset, collate
    from .losses import combine_losses, supmin, training_loss
    from .methylweave import MethylWeave, MODEL_CONFIG, PROFILE
else:
    from dataset import FeatureDataset, collate
    from losses import combine_losses, supmin, training_loss
    from methylweave import MethylWeave, MODEL_CONFIG, PROFILE


class TrainingModel(MethylWeave, pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.settings = config
        self.val_outputs = []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["methylweave_profile"] = PROFILE
        checkpoint["training_config"] = dict(self.settings)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch[:-1], batch[-1]
        loss, ce, contrast = training_loss(self, inputs, labels)
        for name, value in (("train_loss", loss), ("train_ce", ce), ("train_supmin", contrast)):
            self.log(name, value, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch[:-1], batch[-1]
        v1 = self(*inputs, mask_context=True)
        v2 = self(*inputs, mask_context=True)
        contrast = supmin(self.project_pair(v1, v2), labels)
        logits = self.get_logits(*inputs)
        ce = F.cross_entropy(logits, labels)
        loss = combine_losses(self, contrast, ce)
        self.val_outputs.append((logits.softmax(dim=1)[:, 1].detach().cpu(),
                                 labels.detach().cpu(), loss.detach().cpu()))

    def _gather(self, tensor):
        tensor = tensor.to(self.device)
        if self.trainer.world_size > 1:
            tensor = self.all_gather(tensor).reshape(-1)
        return tensor.detach().cpu()

    def on_validation_epoch_end(self):
        probs, labels, losses = zip(*self.val_outputs)
        y = self._gather(torch.cat(labels))
        p = self._gather(torch.cat(probs))
        val_loss = self._gather(torch.stack(losses)).mean()
        score = BinaryAUPRC().update(p, y).compute() if y.sum() else 0.0
        self.log("avg_val_AUPRC", score, prog_bar=True, sync_dist=True)
        self.log("avg_val_loss", val_loss, sync_dist=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        s = self.settings
        optimizer = torch.optim.AdamW(self.parameters(), lr=s["learning_rate"],
                                      weight_decay=s["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=s["cosine_t_max"], eta_min=s["eta_min"]
        )
        return [optimizer], [scheduler]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--cache-dir", type=Path, help="Defaults to a feature-cache directory under --out")
    p.add_argument("--cpu", action="store_true", help="Single-process FP32; intended for small checks")
    for key in ("batch-size", "devices", "epochs", "workers"):
        p.add_argument(f"--{key}", type=int)
    a = p.parse_args()
    with a.config.open() as f:
        config = yaml.safe_load(f)
    if config["model"] != MODEL_CONFIG:
        p.error("model settings must match the MethylWeave architecture")
    s = dict(config["training"])
    for key in ("batch_size", "devices", "epochs", "workers"):
        if getattr(a, key) is not None:
            s[key] = getattr(a, key)
    if a.cpu:
        s.update(devices=1, precision=32)
    if min(s["devices"], s["epochs"]) < 1 or s["workers"] < 0 or s["batch_size"] < 2:
        p.error("require positive epochs/devices, nonnegative workers and batch size >= 2")
    if not a.cpu and torch.cuda.device_count() < s["devices"]:
        p.error("not enough visible CUDA devices; use --devices or --cpu")
    if a.train.resolve() == a.val.resolve():
        p.error("training and validation must use distinct files")

    torch.set_float32_matmul_precision(s["matmul_precision"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cache = a.cache_dir or a.out / "feature-cache"
    datasets = [FeatureDataset(path, cache) for path in (a.train, a.val)]
    per_rank = math.ceil(len(datasets[0]) / s["devices"])
    if per_rank < 2 or per_rank % s["batch_size"] == 1:
        p.error("training would produce a singleton batch; choose another batch size (BatchNorm needs 2+)")
    loader_options = dict(batch_size=s["batch_size"], num_workers=s["workers"],
                          pin_memory=not a.cpu, collate_fn=collate)
    if s["workers"]:
        loader_options.update(persistent_workers=True, prefetch_factor=s["prefetch_factor"])
    train, val = [DataLoader(ds, shuffle=(i == 0), **loader_options) for i, ds in enumerate(datasets)]
    checkpoint = ModelCheckpoint(dirpath=str(a.out), filename="methylweave-{epoch:02d}-{avg_val_AUPRC:.5f}",
                                 monitor="avg_val_AUPRC", mode="max", save_top_k=1, save_last=True)
    stop = EarlyStopping(monitor="avg_val_AUPRC", mode="max", patience=s["early_stop_patience"],
                         min_delta=s["early_stop_min_delta"])
    devices = {"accelerator": "cpu" if a.cpu else "gpu", "devices": s["devices"], "precision": s["precision"]}
    if not a.cpu and s["devices"] > 1:
        devices["strategy"] = DDPStrategy(find_unused_parameters=False)
    model = TrainingModel(s)
    trainer = pl.Trainer(
        **devices, max_epochs=s["epochs"], callbacks=[checkpoint, stop], benchmark=True,
        deterministic=False, num_sanity_val_steps=0, log_every_n_steps=100,
        default_root_dir=str(a.out), gradient_clip_val=s["gradient_clip"],
        logger=CSVLogger(str(a.out), name="metrics"), enable_model_summary=False,
    )
    trainer.fit(model, train, val)
    if trainer.is_global_zero:
        print(f"Best checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
