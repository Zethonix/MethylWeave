"""Binary SupMin and warm-up EMA scaling for joint classification training.

SupMin follows Mildenberger et al. (CVPR 2025); see THIRD_PARTY_NOTICES.txt.
"""
import torch
from torch.nn import functional as F


def supmin(features, labels):
    """All local views are candidates; class 1 shares positives across records."""
    batch, views = features.shape[:2]
    device = features.device
    labels = labels.reshape(-1).to(device)
    if labels.numel() != batch or views != 2:
        raise ValueError("SupMin expects two views and one binary label per record")

    pos = labels == 1
    positive = (pos[:, None] & pos[None, :]).float()
    positive.fill_diagonal_(1.0)
    positive = positive.repeat(views, views)
    nonself = torch.ones_like(positive).fill_diagonal_(0.0)
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = torch.div(torch.matmul(contrast, contrast.T), 0.50)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive = positive * nonself
    exp_logits = torch.exp(logits) * nonself
    log_probability = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_positive = (positive * log_probability).sum(1) / positive.sum(1).clamp(min=1e-8)
    return (-0.50 / 0.07 * mean_positive).view(views, batch).mean()


def register_loss_state(model):
    model.register_buffer("loss_ref_contrast", torch.tensor(0.08, dtype=torch.float32))
    model.register_buffer("loss_ref_ce", torch.tensor(0.02, dtype=torch.float32))
    model.register_buffer("loss_balance_updates", torch.tensor(0, dtype=torch.long))
    model.register_buffer("loss_balance_frozen", torch.tensor(0, dtype=torch.uint8))


def _mean_across_ranks(value):
    synced = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(synced, op=torch.distributed.ReduceOp.SUM)
        synced = synced / torch.distributed.get_world_size()
    return synced


@torch.no_grad()
def update_loss_scale(model, contrast, ce):
    if bool(model.loss_balance_frozen.item()):
        return
    n = int(model.loss_balance_updates.item())
    if n >= 500:
        model.loss_balance_frozen.fill_(1)
        return
    contrast_value = _mean_across_ranks(contrast).to(model.loss_ref_contrast.device).clamp_min(1e-8)
    ce_value = _mean_across_ranks(ce).to(model.loss_ref_ce.device).clamp_min(1e-8)
    beta = 0.0 if n == 0 else 0.98
    model.loss_ref_contrast.mul_(beta).add_(contrast_value * (1.0 - beta))
    model.loss_ref_ce.mul_(beta).add_(ce_value * (1.0 - beta))
    model.loss_balance_updates.add_(1)
    if n + 1 == 500:
        model.loss_balance_frozen.fill_(1)


def combine_losses(model, contrast, ce, *, update=False):
    if update:
        update_loss_scale(model, contrast, ce)
    target = 0.5 * (model.loss_ref_contrast.detach() + model.loss_ref_ce.detach()).clamp_min(1e-8)
    contrast_component = target.to(contrast.device) * contrast / model.loss_ref_contrast.to(contrast.device).clamp_min(1e-8)
    ce_component = target.to(contrast.device) * ce / model.loss_ref_ce.to(ce.device).clamp_min(1e-8)
    return 0.10 * contrast_component + (1.0 - 0.10) * ce_component


def training_loss(model, inputs, labels):
    """Two masked views, CE averaged over both, and projected SupMin."""
    view1 = model(*inputs, mask_context=True)
    view2 = model(*inputs, mask_context=True)
    ce = (F.cross_entropy(model.cls2(view1), labels) + F.cross_entropy(model.cls2(view2), labels)) / 2.0
    contrast = supmin(model.project_pair(view1, view2), labels)
    return combine_losses(model, contrast, ce, update=True), ce, contrast
