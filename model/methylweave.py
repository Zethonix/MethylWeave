"""MethylWeave's fixed reference architecture. No pretrained sequence encoder."""
import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .losses import register_loss_state
else:
    from losses import register_loss_state


def _local_difference_features(x):
    left = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
    right = torch.cat([x[:, 1:, :], x[:, -1:, :]], dim=1)
    return torch.cat([x - left, right - x, right - left], dim=-1)


class ReliabilityAwareDualPathSignalAdapter(nn.Module):
    """
    Reliability-aware dual-path adapter for nanopore signal statistics.
    It preserves the stable lightweight residual path, then adds a controlled
    channel-independent/channel-mixing correction modulated by std/dwell-derived
    reliability evidence.
    """

    def __init__(self, in_channels=3, out_channels=128, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.base_proj = nn.Linear(in_channels, out_channels)
        self.base_depthwise = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.base_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.channel_value_proj = nn.Linear(2, out_channels)
        self.channel_embedding = nn.Parameter(torch.zeros(1, 1, in_channels, out_channels))
        self.channel_norm = nn.LayerNorm(out_channels)
        self.independent_depthwise = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.independent_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.channel_mixer = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels), nn.GELU(), nn.Linear(out_channels, out_channels))
        self.channel_mix_gate = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels), nn.Sigmoid())
        self.evidence_score = nn.Sequential(nn.LayerNorm(out_channels), nn.Linear(out_channels, 1))
        self.correction_depthwise = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.correction_pointwise = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.02))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def _read_normalize(self, x):
        center = x.mean(dim=1, keepdim=True)
        scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(0.001)
        return ((x - center) / scale).clamp(-5.0, 5.0)

    def forward(self, x):
        base = self.base_proj(x)
        base_local = base.transpose(1, 2)
        base_local = self.base_depthwise(base_local)
        base_local = F.gelu(self.base_pointwise(base_local))
        base_local = base_local.transpose(1, 2)
        x_norm = self._read_normalize(x)
        channel_input = torch.stack([x, x_norm], dim=-1)
        channel_tokens = self.channel_value_proj(channel_input)
        channel_tokens = self.channel_norm(channel_tokens + self.channel_embedding)
        (bsz, length, channels, dim) = channel_tokens.shape
        independent = channel_tokens.permute(0, 2, 3, 1).reshape(bsz * channels, dim, length)
        independent = self.independent_depthwise(independent)
        independent = F.gelu(self.independent_pointwise(independent))
        independent = independent.reshape(bsz, channels, dim, length).permute(0, 3, 1, 2)
        channel_tokens = channel_tokens + independent
        channel_context = channel_tokens.mean(dim=2, keepdim=True)
        mixed = self.channel_mixer(channel_tokens + channel_context)
        mix_gate = self.channel_mix_gate(channel_tokens)
        channel_tokens = channel_tokens + mix_gate * mixed
        evidence_weight = torch.softmax(self.evidence_score(channel_tokens), dim=2)
        correction = torch.sum(evidence_weight * channel_tokens, dim=2)
        correction_local = correction.transpose(1, 2)
        correction_local = self.correction_depthwise(correction_local)
        correction_local = F.gelu(self.correction_pointwise(correction_local))
        correction = correction_local.transpose(1, 2)
        correction = torch.tanh(self.residual_scale) * correction
        return self.norm(base + self.dropout(base_local + correction))


def _basewise_raw_geometry_features(raw_signal):
    """Differentiable base-wise waveform geometry for aligned raw current."""
    raw_len = raw_signal.size(-1)
    zeros = raw_signal.new_zeros(raw_signal.size(0), raw_signal.size(1), 1)
    raw_mean = raw_signal.mean(dim=-1, keepdim=True)
    raw_std = raw_signal.std(dim=-1, keepdim=True, unbiased=False)
    (raw_max, raw_argmax) = raw_signal.max(dim=-1, keepdim=True)
    (raw_min, raw_argmin) = raw_signal.min(dim=-1, keepdim=True)
    raw_range = raw_max - raw_min
    area = raw_signal.sum(dim=-1, keepdim=True) / max(raw_len, 1)
    energy = raw_signal.pow(2).mean(dim=-1, keepdim=True)
    slope = raw_signal[:, :, -1:] - raw_signal[:, :, :1]
    if raw_len > 1:
        diff = raw_signal[:, :, 1:] - raw_signal[:, :, :-1]
        abs_slope = diff.abs().mean(dim=-1, keepdim=True)
        max_up = diff.max(dim=-1, keepdim=True).values
        max_down = diff.min(dim=-1, keepdim=True).values
        local_variance = diff.var(dim=-1, keepdim=True, unbiased=False)
    else:
        abs_slope = zeros
        max_up = zeros
        max_down = zeros
        local_variance = zeros
    if raw_len > 2:
        second_diff = diff[:, :, 1:] - diff[:, :, :-1]
        curvature = second_diff.abs().mean(dim=-1, keepdim=True)
    else:
        curvature = zeros
    denom = max(raw_len - 1, 1)
    peak_pos = raw_argmax.float() / denom
    valley_pos = raw_argmin.float() / denom
    return torch.cat([raw_mean, raw_std, raw_range, area, energy, slope, abs_slope, max_up, max_down, curvature, local_variance, peak_pos, valley_pos], dim=-1)


class GatedBaseStateSpaceMixer(nn.Module):
    """
    Lightweight bidirectional state-space style mixer over 13 base-level tokens.

    It keeps the base-aligned structure: base-wise raw patch tokens are first
    produced independently, then only the 13 aligned bases exchange information.
    """

    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.forward_decay = nn.Parameter(torch.zeros(d_model))
        self.backward_decay = nn.Parameter(torch.zeros(d_model))
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout))
        self.local_refine = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, kernel_size=1), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def _scan(self, value, gate, decay, reverse=False):
        if reverse:
            value = torch.flip(value, dims=[1])
            gate = torch.flip(gate, dims=[1])
        alpha = torch.sigmoid(decay).view(1, -1)
        state = torch.zeros(value.size(0), value.size(-1), device=value.device, dtype=value.dtype)
        states = []
        for idx in range(value.size(1)):
            candidate = gate[:, idx, :] * value[:, idx, :]
            state = alpha * state + (1.0 - alpha) * candidate
            states.append(state)
        mixed = torch.stack(states, dim=1)
        if reverse:
            mixed = torch.flip(mixed, dims=[1])
        return mixed

    def forward(self, tokens):
        (value, gate) = self.in_proj(tokens).chunk(2, dim=-1)
        value = torch.tanh(value)
        gate = torch.sigmoid(gate)
        forward_state = self._scan(value, gate, self.forward_decay, reverse=False)
        backward_state = self._scan(value, gate, self.backward_decay, reverse=True)
        mixed = self.out_proj(torch.cat([forward_state, backward_state], dim=-1))
        local = self.local_refine(mixed.transpose(1, 2)).transpose(1, 2)
        return self.norm(tokens + mixed + local)


class BIPSSMRawSignalEncoder(nn.Module):
    """
    Base-wise Invariant Patch State-Space raw current encoder.

    The statistics encoder remains the trunk. Each 16-point base-aligned raw
    current patch is encoded independently, then a lightweight state-space mixer
    models propagation across the 13 aligned bases. Raw evidence is residual only.
    """

    def __init__(self, in_channels=3, out_channels=128, raw_signal_len=16, dropout=0.1, raw_residual_init=0.02, center_radius=2):
        super().__init__()
        self.raw_signal_len = int(raw_signal_len)
        self.center_radius = int(center_radius)
        self.stats_encoder = ReliabilityAwareDualPathSignalAdapter(
            in_channels=in_channels, out_channels=out_channels, dropout=dropout
        )
        patch_hidden = max(out_channels // 2, 32)
        self.patch_conv = nn.Sequential(nn.Conv1d(1, patch_hidden, kernel_size=3, padding=1), nn.GELU(), nn.Conv1d(patch_hidden, patch_hidden, kernel_size=3, padding=1), nn.GELU())
        self.patch_proj = nn.Sequential(nn.LayerNorm(patch_hidden * 2 + 13), nn.Linear(patch_hidden * 2 + 13, out_channels), nn.GELU(), nn.Dropout(dropout))
        self.patch_local = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels), nn.GELU(), nn.Conv1d(out_channels, out_channels, kernel_size=1), nn.Dropout(dropout))
        self.inter_base_mixer = GatedBaseStateSpaceMixer(d_model=out_channels, dropout=dropout)
        self.stats_control_gate = nn.Sequential(nn.LayerNorm(in_channels + 2), nn.Linear(in_channels + 2, out_channels), nn.Sigmoid())
        nn.init.constant_(self.stats_control_gate[1].bias, -1.0)
        self.raw_scale = nn.Parameter(torch.tensor(float(raw_residual_init)))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def _prepare_raw_signal(self, raw_signal):
        if raw_signal.dim() == 4 and raw_signal.size(-1) == 1:
            raw_signal = raw_signal.squeeze(-1)
        raw_signal = raw_signal.float()
        if raw_signal.size(-1) < self.raw_signal_len:
            raw_signal = F.pad(raw_signal, (0, self.raw_signal_len - raw_signal.size(-1)))
        elif raw_signal.size(-1) > self.raw_signal_len:
            raw_signal = raw_signal[:, :, :self.raw_signal_len]
        return raw_signal.clamp(-6.0, 6.0)

    def _center_weights(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=dtype)
        center = (length - 1) / 2.0
        weights = 0.75 + 0.25 * torch.exp(-0.5 * ((positions - center) / max(self.center_radius, 1)) ** 2)
        return weights.view(1, length, 1)

    def _encode_raw_patches(self, raw_signal):
        (bsz, length, raw_len) = raw_signal.shape
        raw_flat = raw_signal.reshape(bsz * length, 1, raw_len)
        conv_feature = self.patch_conv(raw_flat)
        pooled = torch.cat([conv_feature.mean(dim=-1), conv_feature.max(dim=-1).values], dim=-1)
        geometry = _basewise_raw_geometry_features(raw_signal).reshape(bsz * length, -1)
        patch_tokens = self.patch_proj(torch.cat([pooled, geometry], dim=-1)).reshape(bsz, length, -1)
        local = self.patch_local(patch_tokens.transpose(1, 2)).transpose(1, 2)
        return (patch_tokens + local, geometry.reshape(bsz, length, -1))

    def forward(self, signal_stats, raw_signal):
        stats_feature = self.stats_encoder(signal_stats)
        raw_signal = self._prepare_raw_signal(raw_signal)
        (raw_tokens, raw_geometry) = self._encode_raw_patches(raw_signal)
        raw_tokens = self.inter_base_mixer(raw_tokens)
        raw_tokens = raw_tokens * self._center_weights(raw_tokens.size(1), raw_tokens.device, raw_tokens.dtype)
        raw_norm = raw_tokens.norm(dim=-1, keepdim=True)
        raw_variation = raw_geometry[:, :, 6:7].abs()
        gate_input = torch.cat([signal_stats.float(), raw_norm, raw_variation], dim=-1)
        gate = self.stats_control_gate(gate_input)
        raw_residual = torch.tanh(self.raw_scale) * gate * self.dropout(raw_tokens)
        return self.norm(stats_feature + raw_residual)


def parse_seq_multik_values(values):
    if isinstance(values, str):
        raw_values = [value.strip() for value in values.split(',') if value.strip()]
        parsed = [int(value) for value in raw_values]
    elif isinstance(values, (list, tuple)):
        parsed = [int(value) for value in values]
    else:
        parsed = [int(values)]
    cleaned = []
    for value in parsed:
        if value < 1:
            raise ValueError(f'seq_multik_values must contain positive integers, got {value}')
        if value % 2 == 0:
            raise ValueError(f'seq_multik_values should use odd windows to preserve center alignment, got {value}')
        if value not in cleaned:
            cleaned.append(value)
    return tuple(cleaned)


class CCenteredMultiGranularityMotifSequenceEncoder(nn.Module):
    """
    Center-C anchored multi-granularity motif sequence encoder.

    The encoder keeps a 13-token base-aligned output so sequence tokens remain
    directly aligned with mean/std/dwell statistics and BIP-SSM raw-current
    tokens. It combines explicit 1-mer base embeddings, the existing 5-mer
    DNABERT-style token embeddings, local multi-k motif windows, and an optional
    center-C query readout.
    """

    def __init__(self, d_model=128, seq_len=13, multik_values=(1, 3, 5, 7), dropout=0.1, heads=4):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.multik_values = parse_seq_multik_values(multik_values)
        if any((value > self.seq_len for value in self.multik_values)):
            raise ValueError(f'seq_multik_values must be <= seq_len={self.seq_len}, got {self.multik_values}')
        if self.d_model % heads != 0:
            raise ValueError('CenMGM heads must divide d_model')
        self.base_embedding = nn.Embedding(5, d_model)
        self.kmer_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.position_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.motif_embedding = nn.Embedding(4, d_model)
        self.grain_encoders = nn.ModuleDict()
        for k_value in self.multik_values:
            padding = min(k_value // 2, self.seq_len // 2)
            self.grain_encoders[str(k_value)] = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=k_value, padding=padding, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, kernel_size=1), nn.GELU())
        self.grain_router = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, max(d_model // 2, 1)), nn.GELU(), nn.Linear(max(d_model // 2, 1), len(self.multik_values)))
        self.token_mixer = nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model), nn.GELU(), nn.Conv1d(d_model, d_model, kernel_size=1))
        self.center_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.center_write_gate = nn.Sequential(nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, d_model), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def _position_features(self, device, dtype):
        positions = torch.arange(self.seq_len, device=device, dtype=dtype)
        center = float(self.seq_len // 2)
        denom = max(center, 1.0)
        rel = (positions - center) / denom
        dist = rel.abs()
        return torch.stack([rel, dist], dim=-1).unsqueeze(0)

    def _resize_kmer_embeddings(self, kmer_embeddings):
        if kmer_embeddings.size(1) == self.seq_len:
            return kmer_embeddings
        resized = F.interpolate(kmer_embeddings.transpose(1, 2), size=self.seq_len, mode='linear', align_corners=False)
        return resized.transpose(1, 2)

    def forward(self, input_ids, kmer_embeddings, base_ids=None, motif_ids=None):
        if base_ids is None or motif_ids is None:
            raise ValueError('c_centered_multik_motif sequence encoder requires base_ids and motif_ids')
        base_ids = base_ids.to(kmer_embeddings.device).long().clamp(0, 4)
        motif_ids = motif_ids.to(kmer_embeddings.device).long().clamp(0, 3)
        base_feature = self.base_embedding(base_ids)
        kmer_feature = self.kmer_proj(self._resize_kmer_embeddings(kmer_embeddings))
        pos_feature = self.position_proj(self._position_features(kmer_embeddings.device, kmer_embeddings.dtype))
        motif_feature = self.motif_embedding(motif_ids).unsqueeze(1)
        x = base_feature + kmer_feature + pos_feature
        motif_context = motif_feature.expand(-1, self.seq_len, -1)
        x = x + motif_context
        grain_input = x.transpose(1, 2)
        grain_features = []
        for k_value in self.multik_values:
            encoded = self.grain_encoders[str(k_value)](grain_input).transpose(1, 2)
            if encoded.size(1) != self.seq_len:
                encoded = encoded[:, :self.seq_len, :]
            grain_features.append(encoded)
        grains = torch.stack(grain_features, dim=2)
        route = torch.softmax(self.grain_router(x), dim=-1)
        mixed = (grains * route.unsqueeze(-1)).sum(dim=2)
        x = self.norm1(x + self.dropout(mixed))
        local_mixed = self.token_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + self.dropout(local_mixed))
        center_idx = self.seq_len // 2
        query = x[:, center_idx:center_idx + 1, :]
        query = query + motif_feature
        (center_context, _) = self.center_attention(query, x, x, need_weights=False)
        center_context = center_context.expand(-1, self.seq_len, -1)
        gate = self.center_write_gate(torch.cat([x, center_context, motif_context], dim=-1))
        x = self.norm3(x + self.dropout(gate * center_context))
        return self.norm_out(x + self.dropout(self.ffn(x)))


class BasisDynamicCrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with concat-MLP fusion of four feature bases."""

    def __init__(self, d_model=128, raw_signal_dim=3, heads=4, dropout=0.1, route_concat_hidden=52):
        super().__init__()
        self.seq_norm = nn.LayerNorm(d_model)
        self.signal_norm = nn.LayerNorm(d_model)
        self.seq_to_signal = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.signal_to_seq = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.seq_basis = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.signal_basis = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.cross_basis = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.GELU())
        self.stats_basis = nn.Sequential(nn.LayerNorm(raw_signal_dim * 4), nn.Linear(raw_signal_dim * 4, d_model), nn.GELU())
        route_concat_hidden = int(route_concat_hidden)
        self.route_concat = nn.Sequential(nn.LayerNorm(d_model * 4), nn.Linear(d_model * 4, route_concat_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(route_concat_hidden, d_model))
        self.seq_out = nn.LayerNorm(d_model)
        self.signal_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _stats_features(self, raw_signal):
        diff = _local_difference_features(raw_signal)
        return torch.cat([raw_signal, diff], dim=-1)

    def forward(self, seq_feat, nano_feat, raw_signal):
        seq_norm = self.seq_norm(seq_feat)
        signal_norm = self.signal_norm(nano_feat)
        (seq_attends_signal, _) = self.seq_to_signal(seq_norm, signal_norm, signal_norm, need_weights=False)
        (signal_attends_seq, _) = self.signal_to_seq(signal_norm, seq_norm, seq_norm, need_weights=False)
        seq_basis = self.seq_basis(seq_feat)
        signal_basis = self.signal_basis(nano_feat)
        cross_basis = self.cross_basis(torch.cat([seq_attends_signal, signal_attends_seq], dim=-1))
        basis_tensors = [seq_basis, signal_basis, cross_basis]
        stats_basis = self.stats_basis(self._stats_features(raw_signal.float()))
        basis_tensors.append(stats_basis)
        shared = self.route_concat(torch.cat(basis_tensors, dim=-1))
        seq_out = self.seq_out(seq_feat + self.dropout(shared + seq_attends_signal))
        signal_out = self.signal_out(nano_feat + self.dropout(shared + signal_attends_seq))
        return torch.cat([seq_out, signal_out], dim=-1)


MODEL_CONFIG = {
    "model_dim": 128, "raw_signal_len": 16, "seq_multik_values": "1,3,5,7",
    "fixed_readout_index": 7, "seq_dropout_prob": 0.15, "protect_center_window": 2,
    "use_contrastive_projection_head": True, "contrast_projection_dim": 128,
    "contrast_projection_hidden_dim": 128, "contrast_temperature": 0.50,
    "loss_alpha": 0.10, "loss_balance_mode": "warmup_ema_norm",
    "loss_balance_warmup_steps": 500, "loss_balance_ema_beta": 0.98,
    "loss_ref_contrast_init": 0.08, "loss_ref_ce_init": 0.02,
    "loss_balance_freeze_after_warmup": True, "loss_balance_target_scale": 0.0,
}
PROFILE = "methylweave-paper-final-v1"


class MethylWeave(nn.Module):
    """MethylWeave model. Forward returns the normalized 128-D representation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1100, 128)
        self.c_centered_multik_motif_encoder = CCenteredMultiGranularityMotifSequenceEncoder()
        self.nano_embedding = BIPSSMRawSignalEncoder()
        self.fusion_layer = BasisDynamicCrossAttentionFusion()
        self.bi_gru = nn.GRU(256, 128, num_layers=2, batch_first=True,
                             bidirectional=True, dropout=0.1)
        self.cls2 = nn.Sequential(
            nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2),
        )
        self.cls3 = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),
                                  nn.ReLU(), nn.Linear(128, 128))
        self.contrast_projection_head = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.context_mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        register_loss_state(self)

    def _mask_context(self, seq_embeddings, nano):
        # There are 13 aligned embeddings after the sequence encoder.
        eligible = ((torch.arange(13) - 6).abs() > MODEL_CONFIG["protect_center_window"])
        eligible = eligible.to(seq_embeddings.device).float().view(1, 13, 1)
        stds = nano[:, :, 1].float().abs().unsqueeze(2)
        eligible = eligible.expand_as(stds)
        count = eligible.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (stds * eligible).sum(dim=1, keepdim=True) / count
        relative = (stds / (mean + 1e-6)).clamp(0.25, 4.0)
        base_probability = MODEL_CONFIG["seq_dropout_prob"]
        probability = (base_probability * relative).clamp(max=0.5) * eligible
        drop = torch.rand_like(probability) < probability
        mask_token = self.context_mask_token.to(dtype=seq_embeddings.dtype)
        return torch.where(drop.expand_as(seq_embeddings), mask_token.expand_as(seq_embeddings), seq_embeddings)

    def forward(self, input_ids, nano_data, raw_signal, base_ids, motif_ids, *, mask_context=False):
        kmer = self.embedding(input_ids)
        seq = self.c_centered_multik_motif_encoder(input_ids, kmer, base_ids, motif_ids)
        if mask_context:
            seq = self._mask_context(seq, nano_data)
        signal = self.nano_embedding(nano_data, raw_signal)
        fused = self.fusion_layer(seq, signal, nano_data)
        hidden, _ = self.bi_gru(fused)
        readout = self.cls3(hidden[:, MODEL_CONFIG["fixed_readout_index"], :])
        return F.normalize(readout, p=2, dim=1, eps=1e-8)

    def get_logits(self, input_ids, nano_data, raw_signal, base_ids, motif_ids):
        return self.cls2(self(input_ids, nano_data, raw_signal, base_ids, motif_ids))

    def project_pair(self, view1, view2):
        z1 = F.normalize(self.contrast_projection_head(view1), p=2, dim=1, eps=1e-8)
        z2 = F.normalize(self.contrast_projection_head(view2), p=2, dim=1, eps=1e-8)
        return torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)


def load_checkpoint(model, path):
    """Validate the declared profile and strictly load the active weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if checkpoint.get("methylweave_profile") != PROFILE:
        raise ValueError("Checkpoint does not declare the supported MethylWeave profile")
    return model.load_state_dict(state, strict=True)
