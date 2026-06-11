"""PatchTST self-supervised encoder for federated FedAvg baseline.

Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers" (ICLR 2023); official repo https://github.com/PatchTST/PatchTST

Design (aligned with PatchTST self-supervised pretraining):
  * patch time series into subseries-level tokens;
  * channel-independence: shared Transformer weights per variate;
  * masked patch reconstruction loss on randomly masked positions.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    if seq_len < patch_len:
        return 1
    return max(1, (seq_len - patch_len) // stride + 1)


class _RevIN(nn.Module):
    """Per-sample, per-channel instance normalization (PatchTST-style)."""

    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(self.eps)
        x_norm = (x - mean) / std
        if self.affine_weight is not None and self.affine_bias is not None:
            w = self.affine_weight.view(1, -1, 1)
            b = self.affine_bias.view(1, -1, 1)
            x_norm = x_norm * w + b
        return x_norm, mean, std


class PatchTSTSSLModel(nn.Module):
    """Masked patch reconstruction + pooled embedding for downstream SVM eval."""

    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        head_dropout: float = 0.1,
        mask_ratio: float = 0.4,
        to_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.method = "patchtst"
        self.in_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.num_patch = _num_patches(self.seq_len, self.patch_len, self.stride)
        self.d_model = int(d_model)
        self.mask_ratio = float(mask_ratio)

        self.revin = _RevIN(self.in_channels, affine=True)
        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patch, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.enc_dropout = nn.Dropout(float(attn_dropout) if attn_dropout > 0 else float(dropout))

        self.pretrain_head = nn.Sequential(
            nn.Dropout(float(head_dropout)),
            nn.Linear(self.d_model, self.patch_len),
        )

        self.feature_dim = self.in_channels * self.d_model
        self._to_cuda = bool(to_cuda) and torch.cuda.is_available()
        if self._to_cuda:
            self.cuda()

    @property
    def aggregated_param_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, num_patch, C, patch_len)."""
        if x.size(-1) < self.patch_len:
            pad = self.patch_len - int(x.size(-1))
            x = F.pad(x, (0, pad))
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        return patches.permute(0, 2, 1, 3).contiguous()

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, P, C, L) -> (B, C, d_model, P)."""
        bsz, num_patch, n_vars, patch_len = patches.shape
        tok = self.patch_proj(patches)  # (B, P, C, d_model)
        tok = tok.permute(0, 2, 1, 3).contiguous()  # (B, C, P, d_model)
        tok = tok.reshape(bsz * n_vars, num_patch, self.d_model)
        tok = self.enc_dropout(tok + self.pos_embed)
        tok = self.transformer(tok)
        tok = tok.reshape(bsz, n_vars, num_patch, self.d_model)
        return tok.permute(0, 1, 3, 2).contiguous()

    def _decode_patches(self, encoded: torch.Tensor) -> torch.Tensor:
        """(B, C, d_model, P) -> (B, P, C, patch_len)."""
        z = encoded.permute(0, 3, 1, 2).contiguous()  # (B, P, C, d_model)
        return self.pretrain_head(z)

    def get_last_features(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        x_norm, _mean, _std = self.revin.norm(x)
        patches = self._create_patches(x_norm)
        encoded = self._encode_patches(patches)
        # last patch token per channel (PatchTST classification head style)
        feat = encoded[:, :, :, -1]  # (B, C, d_model)
        feat = feat.reshape(feat.size(0), -1)
        if detach:
            feat = feat.detach()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_last_features(x, detach=False)

    def masked_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_norm, _mean, _std = self.revin.norm(x)
        patches = self._create_patches(x_norm)
        bsz, num_patch, n_vars, patch_len = patches.shape

        # Mask the same patch index for all channels (PatchTST pretrain protocol).
        rand = torch.rand(bsz, num_patch, device=x.device)
        mask = rand < self.mask_ratio
        if not mask.any():
            mask.view(-1)[0] = True

        masked = patches.clone()
        mask_exp = mask.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        masked = masked.masked_fill(mask_exp, 0.0)

        encoded = self._encode_patches(masked)
        pred = self._decode_patches(encoded)

        target = patches
        diff = (pred - target) ** 2
        mask_f = mask.unsqueeze(-1).unsqueeze(-1).float()
        denom = mask_f.sum().clamp_min(1.0) * patch_len * n_vars
        return (diff * mask_f).sum() / denom

    @torch.no_grad()
    def transform(
        self,
        X,
        *,
        batch_size: int = 512,
        normalize: bool = True,
        result_type: str = "numpy",
    ):
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float)
        self.eval()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X), batch_size=batch_size, shuffle=False
        )
        outs = []
        for (x,) in dl:
            if self._to_cuda:
                x = x.cuda()
            outs.append(self.get_last_features(x, detach=True).cpu())
        z = torch.cat(outs, 0)
        if normalize:
            z = F.normalize(z, dim=1)
        if result_type == "tensor":
            return z
        return z.detach().numpy()
