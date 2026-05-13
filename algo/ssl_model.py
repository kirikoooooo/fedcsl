"""Self-supervised shapelet model wrappers for federated baselines.

目标：
- 复用 FedCSL 现有的 shapelet encoder，保证 backbone 一致；
- 在 encoder 之后接 projector / predictor，支持 BYOL；
- 暴露 ``transform()`` / ``get_last_features()``，复用现有下游 SVM 评估协议。
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import LearningShapeletsModel, LearningShapeletsModelMixDistances


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, *, final_bn: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    ]
    if final_bn:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)


class ShapeletSSLModel(nn.Module):
    def __init__(
        self,
        *,
        method: str,
        shapelets_size_and_len: dict,
        in_channels: int,
        num_classes: int,
        dist_measure: str = "mix",
        projector_hidden_dim: int = 256,
        projector_out_dim: int = 128,
        predictor_hidden_dim: int = 256,
        to_cuda: bool = True,
    ) -> None:
        super().__init__()
        method = str(method).lower()
        if method != "byol":
            raise ValueError(f"unsupported ssl method: {method}")
        self.method = method

        if dist_measure == "mix":
            encoder = LearningShapeletsModelMixDistances(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )
        else:
            encoder = LearningShapeletsModel(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )

        self.encoder = encoder
        self.feature_dim = int(encoder.num_shapelets)
        self.projector = _make_mlp(
            self.feature_dim, projector_hidden_dim, projector_out_dim, final_bn=True
        )
        self.predictor = _make_mlp(projector_out_dim, predictor_hidden_dim, projector_out_dim, final_bn=False)

        # BYOL 的 momentum target，不参与联邦聚合。
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)
        self.non_aggregated_param_names: List[str] = [
            name
            for name, _ in self.named_parameters()
            if name.startswith("target_encoder.") or name.startswith("target_projector.")
        ]

        self._to_cuda = bool(to_cuda) and torch.cuda.is_available()
        if self._to_cuda:
            self.cuda()

    @property
    def aggregated_param_names(self) -> List[str]:
        return [
            name for name, _ in self.named_parameters() if name not in set(self.non_aggregated_param_names)
        ]

    def get_last_features(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        z = self.encoder(x, optimize=None, masking=False)
        if detach:
            z = z.detach()
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_last_features(x, detach=False)

    def online_project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.get_last_features(x, detach=False)
        proj = self.projector(feat)
        return feat, proj

    @torch.no_grad()
    def target_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None or self.target_projector is None:
            raise RuntimeError("target_project() is only valid for BYOL")
        feat = self.target_encoder(x, optimize=None, masking=False)
        return self.target_projector(feat)

    @torch.no_grad()
    def reset_target_network(self) -> None:
        if self.target_encoder is None or self.target_projector is None:
            return
        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=False)
        self.target_projector.load_state_dict(self.projector.state_dict(), strict=False)

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        if self.target_encoder is None or self.target_projector is None:
            return
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)

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
            z = nn.functional.normalize(z, dim=1)
        if result_type == "tensor":
            return z
        return z.detach().numpy()


def sknopp(cz: torch.Tensor, lamd: float = 25.0, max_iters: int = 100) -> torch.Tensor:
    """Balanced soft assignments via Sinkhorn-Knopp."""
    with torch.no_grad():
        n_samples, n_centroids = cz.shape
        probs = F.softmax(cz * lamd, dim=1).T
        r = torch.ones((n_centroids, 1), device=probs.device) / max(1, n_centroids)
        c = torch.ones((n_samples, 1), device=probs.device) / max(1, n_samples)
        inv_n_centroids = 1.0 / max(1, n_centroids)
        inv_n_samples = 1.0 / max(1, n_samples)
        err = 1e3
        for it in range(max_iters):
            r = inv_n_centroids / torch.clamp(probs @ c, min=1e-12)
            c_new = inv_n_samples / torch.clamp((r.T @ probs).T, min=1e-12)
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / torch.clamp(c_new, min=1e-12) - 1.0))
            c = c_new
            if float(err) < 1e-2:
                break
        probs *= c.squeeze()
        probs = probs.T
        probs *= r.squeeze()
        return probs * n_samples


class OrchestraShapeletModel(nn.Module):
    """Shapelet-based Orchestra adaptation.

    与原仓保持相同高层结构：
    - online encoder + projector
    - target encoder + projector
    - memory bank
    - global centroids + local centroids
    - clustering loss + degeneracy regularization
    """

    def __init__(
        self,
        *,
        shapelets_size_and_len: dict,
        in_channels: int,
        num_classes: int,
        dist_measure: str = "mix",
        projector_hidden_dim: int = 256,
        projector_out_dim: int = 128,
        ema_tau: float = 0.99,
        num_global_clusters: int = 32,
        num_local_clusters: int = 8,
        cluster_m_size: int = 128,
        temperature: float = 0.2,
        deg_num_classes: int = 5,
        to_cuda: bool = True,
    ) -> None:
        super().__init__()
        if dist_measure == "mix":
            encoder = LearningShapeletsModelMixDistances(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )
        else:
            encoder = LearningShapeletsModel(
                shapelets_size_and_len=shapelets_size_and_len,
                in_channels=in_channels,
                num_classes=num_classes,
                dist_measure=dist_measure,
                to_cuda=False,
            )

        self.method = "orchestra"
        self.encoder = encoder
        self.feature_dim = int(encoder.num_shapelets)
        self.projector_out_dim = int(projector_out_dim)
        self.temperature = float(temperature)
        self.ema_tau = float(ema_tau)
        self.num_global_clusters = int(num_global_clusters)
        self.num_local_clusters = int(num_local_clusters)
        self.cluster_m_size = int(cluster_m_size)

        self.projector = _make_mlp(
            self.feature_dim, projector_hidden_dim, projector_out_dim, final_bn=True
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        self.deg_layer = nn.Linear(projector_out_dim, int(deg_num_classes))
        self.centroids = nn.Linear(projector_out_dim, self.num_global_clusters, bias=False)
        self.local_centroids = nn.Linear(projector_out_dim, self.num_local_clusters, bias=False)
        self.register_buffer(
            "memory_bank",
            F.normalize(torch.randn(self.cluster_m_size, projector_out_dim), dim=1),
        )

        self.non_aggregated_param_names = [
            name
            for name, _ in self.named_parameters()
            if name.startswith("target_encoder.")
            or name.startswith("target_projector.")
            or name.startswith("local_centroids.")
        ]

        self._to_cuda = bool(to_cuda) and torch.cuda.is_available()
        if self._to_cuda:
            self.cuda()
        self.reset_target_network()

    @property
    def aggregated_param_names(self) -> List[str]:
        non_agg = set(self.non_aggregated_param_names)
        return [name for name, _ in self.named_parameters() if name not in non_agg]

    def get_last_features(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        z = self.encoder(x, optimize=None, masking=False)
        if detach:
            z = z.detach()
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_last_features(x, detach=False)

    def online_project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.get_last_features(x, detach=False)
        proj = F.normalize(self.projector(feat), dim=1)
        return feat, proj

    @torch.no_grad()
    def target_project(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.target_encoder(x, optimize=None, masking=False)
        return F.normalize(self.target_projector(feat), dim=1)

    @torch.no_grad()
    def reset_target_network(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=False)
        self.target_projector.load_state_dict(self.projector.state_dict(), strict=False)

    @torch.no_grad()
    def update_target_network(self, tau: float | None = None) -> None:
        tau = self.ema_tau if tau is None else float(tau)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1.0 - tau)

    @torch.no_grad()
    def reset_memory(self, dataloader, device: torch.device) -> None:
        self.eval()
        bank = []
        n_samples = 0
        for x, _ in dataloader:
            if n_samples >= self.cluster_m_size:
                break
            x = x.to(device).float()
            z = self.target_project(x)
            bank.append(z.detach())
            n_samples += int(x.shape[0])
        if bank:
            bank = torch.cat(bank, dim=0)
            bank = bank[: self.cluster_m_size]
            if bank.shape[0] < self.cluster_m_size:
                pad = bank[torch.randint(bank.shape[0], (self.cluster_m_size - bank.shape[0],))]
                bank = torch.cat([bank, pad], dim=0)
            self.memory_bank.copy_(F.normalize(bank, dim=1))

    @torch.no_grad()
    def update_memory(self, z: torch.Tensor) -> None:
        z = F.normalize(z.detach(), dim=1)
        n = min(int(z.shape[0]), int(self.memory_bank.shape[0]))
        if n <= 0:
            return
        if n < self.memory_bank.shape[0]:
            self.memory_bank[:-n] = self.memory_bank[n:].clone()
        self.memory_bank[-n:] = z[:n]

    @torch.no_grad()
    def local_clustering(self) -> None:
        z = self.memory_bank.detach().clone()
        if z.shape[0] < self.num_local_clusters:
            return
        centroids = z[torch.randperm(z.shape[0])[: self.num_local_clusters]].clone()
        for _ in range(5):
            assigns = sknopp(z @ centroids.T, max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            for idx in range(self.num_local_clusters):
                selected = z[choice_cluster == idx]
                if selected.shape[0] == 0:
                    selected = z[torch.randint(z.shape[0], (1,))]
                centroids[idx] = F.normalize(selected.mean(dim=0), dim=0)
        self.local_centroids.weight.data.copy_(centroids)

    def global_clustering(self, z: torch.Tensor, total_rounds: int = 100) -> None:
        z = F.normalize(z.detach(), dim=1)
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        for _ in range(int(total_rounds)):
            with torch.no_grad():
                sk_assigns = sknopp(self.centroids(z))
            optimizer.zero_grad()
            probs = F.softmax(self.centroids(F.normalize(z, dim=1)) / self.temperature, dim=1)
            loss = -F.cosine_similarity(sk_assigns, probs, dim=-1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1))

    def orchestra_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        deg_labels: torch.Tensor,
    ) -> torch.Tensor:
        c = F.normalize(self.centroids.weight, dim=1).T
        _feat1, z1 = self.online_project(x1)
        _feat2, z2 = self.online_project(x2)
        cz2 = z2 @ c
        logp_z2 = torch.log(F.softmax(cz2 / self.temperature, dim=1) + 1e-12)

        with torch.no_grad():
            self.update_target_network()
            tz1 = self.target_project(x1)
            cp1 = tz1 @ c
            tp1 = F.softmax(cp1 / self.temperature, dim=1)

        loss_cluster = -(tp1 * logp_z2).sum(dim=1).mean()
        deg_proj = F.normalize(self.projector(self.encoder(x3, optimize=None, masking=False)), dim=1)
        loss_deg = F.cross_entropy(self.deg_layer(deg_proj), deg_labels.long())

        with torch.no_grad():
            self.update_memory(tz1)
        return loss_cluster + loss_deg

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
