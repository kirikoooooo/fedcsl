import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from utils import compute_gap_scores

from utils import generate_binomial_mask


def _resolve_device(to_cuda=True, device=None):
    if device is not None:
        return torch.device(device)
    if to_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MAST-style backward profiling wrappers (autograd.Function)
# _BwRangeOut → reset peak before block.backward
# _BwRangeIn  → record peak after block.backward
# The x.clone() in forward creates clean autograd boundaries.
# ---------------------------------------------------------------------------
class _BwRangeIn(torch.autograd.Function):
    """Inserted BEFORE block input. Backward: block.backward → _BwRangeIn.backward → x."""

    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        if getattr(module, 'to_cuda', False):
            torch.cuda.synchronize(module.device)
            pre = getattr(module, '_bw_pre_mem', 0)
            if pre > 0:
                delta = max(
                    int(torch.cuda.max_memory_allocated(module.device) - pre), 0
                )
                # Keep the maximum across multiple backward passes
                if delta > module._bw_peak_mem_delta_bytes:
                    module._bw_peak_mem_delta_bytes = delta
            module._bw_peak_mem_measured = True
        return grad_output, None


class _BwRangeOut(torch.autograd.Function):
    """Inserted AFTER block output. Backward: cat → _BwRangeOut.backward → reset peak → block.backward."""

    @staticmethod
    def forward(ctx, x, module):
        ctx.module = module
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        if getattr(module, 'to_cuda', False):
            torch.cuda.synchronize(module.device)
            torch.cuda.reset_peak_memory_stats(module.device)
            module._bw_pre_mem = torch.cuda.memory_allocated(module.device)
        return grad_output, None


class MinEuclideanDistBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, device=None):
        super(MinEuclideanDistBlock, self).__init__()
        self.device = _resolve_device(to_cuda, device)
        self.to_cuda = self.device.type == "cuda"
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        # Per-block forward memory tracking (peak delta in bytes, measured once on first forward)
        self._peak_mem_delta_bytes: int = 0
        self._peak_mem_measured: bool = False
        # Per-block backward memory tracking (measured on first backward pass)
        self._bw_peak_mem_delta_bytes: int = 0
        self._bw_peak_mem_measured: bool = False
        self._bw_pre_mem: int = 0
        # Retained activation size (output tensor bytes)
        self._retained_activation_bytes: int = 0

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                               dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.to(self.device)
        self.shapelets = nn.Parameter(shapelets.contiguous())
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()

    @property
    def peak_mem_measured(self) -> bool:
        """Whether the forward peak memory has been measured on this block."""
        return self._peak_mem_measured

    @property
    def peak_mem_mb(self) -> float:
        """Peak forward memory delta in MiB (0 if not measured yet or on CPU)."""
        return float(self._peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def bw_peak_mem_measured(self) -> bool:
        return self._bw_peak_mem_measured

    @property
    def bw_peak_mem_mb(self) -> float:
        return float(self._bw_peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def retained_activation_mb(self) -> float:
        return float(self._retained_activation_bytes) / (1024.0 * 1024.0)

    @property
    def param_mem_bytes(self) -> int:
        """Total parameter memory in bytes."""
        total = 0
        for p in self.parameters(recurse=True):
            total += p.numel() * p.element_size()
        return total

    def forward(self, x, masking=False):
        # ---- per-block forward memory profiling (MAST-style, device-local) ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            pre_mem = torch.cuda.memory_allocated(self.device)

        # unfold time series to emulate sliding window
        x = x.unfold(2, self.shapelets_size, 1).contiguous()

        # Record retained activation: unfold output, saved by cdist backward
        if self.to_cuda and not self._peak_mem_measured:
            self._retained_activation_bytes = x.numel() * x.element_size()

        # calculate euclidean distance
        x = torch.cdist(x, self.shapelets, p=2, compute_mode='donot_use_mm_for_euclid_dist')

        # add up the distances of the channels in case of
        # multivariate time series
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3)

        # hard min compared to soft-min from the paper
        x, _ = torch.min(x, 3)

        # ---- finalise forward memory measurement ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            self._peak_mem_delta_bytes = max(
                int(torch.cuda.max_memory_allocated(self.device) - pre_mem), 0
            )
            self._peak_mem_measured = True

        return x




# 每个block 负责一个 scale的40个shapelets
class MaxCosineSimilarityBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, device=None):
        super(MaxCosineSimilarityBlock, self).__init__()
        self.device = _resolve_device(to_cuda, device)
        self.to_cuda = self.device.type == "cuda"
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.in_channels = in_channels
        self.relu = nn.ReLU()
        self.mean = 0
        self.var  = 0
        # Per-block forward memory tracking (peak delta in bytes, measured once on first forward)
        self._peak_mem_delta_bytes: int = 0
        self._peak_mem_measured: bool = False
        # Per-block backward memory tracking
        self._bw_peak_mem_delta_bytes: int = 0
        self._bw_peak_mem_measured: bool = False
        self._bw_pre_mem: int = 0
        # Retained activation size
        self._retained_activation_bytes: int = 0

        # if not registered as parameter, the optimizer will not be able to see the parameters
        shapelets = torch.randn(self.in_channels, self.num_shapelets, self.shapelets_size, requires_grad=True,
                                dtype=torch.float)
        if self.to_cuda:
            shapelets = shapelets.to(self.device)
        self.shapelets = nn.Parameter(shapelets.contiguous())
        # otherwise gradients will not be backpropagated
        self.shapelets.retain_grad()


        # mmoe 专属，增加mlp
        self.prodictor = nn.Sequential(nn.Linear(self.num_shapelets, self.num_shapelets),
                                        nn.ReLU(),
                                        nn.Linear(self.num_shapelets, self.num_shapelets))

    @property
    def peak_mem_measured(self) -> bool:
        return self._peak_mem_measured

    @property
    def peak_mem_mb(self) -> float:
        return float(self._peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def bw_peak_mem_measured(self) -> bool:
        return self._bw_peak_mem_measured

    @property
    def bw_peak_mem_mb(self) -> float:
        return float(self._bw_peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def retained_activation_mb(self) -> float:
        return float(self._retained_activation_bytes) / (1024.0 * 1024.0)

    @property
    def param_mem_bytes(self) -> int:
        total = 0
        for p in self.parameters(recurse=True):
            total += p.numel() * p.element_size()
        return total

    def forward(self, x, masking=False):
        # ---- per-block forward memory profiling ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            pre_mem = torch.cuda.memory_allocated(self.device)

        x = x.unfold(2, self.shapelets_size, 1).contiguous()

        # Record retained activation: unfold output, saved by matmul backward
        if self.to_cuda and not self._peak_mem_measured:
            self._retained_activation_bytes = x.numel() * x.element_size()

        x = x / x.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)

        shapelets_norm = (self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8))
        x = torch.matmul(x, shapelets_norm.transpose(1, 2))

        gap_score_vector,gap_score = compute_gap_scores(x)

        n_dims = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True).transpose(2, 3) / n_dims

        x = self.relu(x)
        x, _ = torch.max(x, 3)

        # ---- finalise forward memory measurement ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            self._peak_mem_delta_bytes = max(
                int(torch.cuda.max_memory_allocated(self.device) - pre_mem), 0
            )
            self._peak_mem_measured = True

        return x




class MaxCrossCorrelationBlock(nn.Module):

    def __init__(self, shapelets_size, num_shapelets, in_channels=1, to_cuda=True, device=None):
        super(MaxCrossCorrelationBlock, self).__init__()
        self.device = _resolve_device(to_cuda, device)
        self.shapelets = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelets_size)
        self.num_shapelets = num_shapelets
        self.shapelets_size = shapelets_size
        self.to_cuda = self.device.type == "cuda"
        # Per-block forward memory tracking
        self._peak_mem_delta_bytes: int = 0
        self._peak_mem_measured: bool = False
        # Per-block backward memory tracking
        self._bw_peak_mem_delta_bytes: int = 0
        self._bw_peak_mem_measured: bool = False
        self._bw_pre_mem: int = 0
        # Retained activation size
        self._retained_activation_bytes: int = 0

        if self.to_cuda:
            self.to(self.device)

    @property
    def peak_mem_measured(self) -> bool:
        return self._peak_mem_measured

    @property
    def peak_mem_mb(self) -> float:
        return float(self._peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def bw_peak_mem_measured(self) -> bool:
        return self._bw_peak_mem_measured

    @property
    def bw_peak_mem_mb(self) -> float:
        return float(self._bw_peak_mem_delta_bytes) / (1024.0 * 1024.0)

    @property
    def retained_activation_mb(self) -> float:
        return float(self._retained_activation_bytes) / (1024.0 * 1024.0)

    @property
    def param_mem_bytes(self) -> int:
        total = 0
        for p in self.parameters(recurse=True):
            total += p.numel() * p.element_size()
        return total

    def forward(self, x, masking=False):
        # ---- per-block forward memory profiling ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            pre_mem = torch.cuda.memory_allocated(self.device)

        x = self.shapelets(x)

        # Record retained activation: Conv1d output, saved by max backward
        if self.to_cuda and not self._peak_mem_measured:
            self._retained_activation_bytes = x.numel() * x.element_size()

        if masking:
            mask = generate_binomial_mask(x.shape, device=x.device)
            x *= mask
        x, _ = torch.max(x, 2, keepdim=True)

        # ---- finalise forward memory measurement ----
        if self.to_cuda and not self._peak_mem_measured:
            torch.cuda.synchronize(self.device)
            self._peak_mem_delta_bytes = max(
                int(torch.cuda.max_memory_allocated(self.device) - pre_mem), 0
            )
            self._peak_mem_measured = True

        return x.transpose(2, 1)





class ShapeletsDistBlocks(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True, checkpoint=False,shapelet_weight=None, device=None):
        super(ShapeletsDistBlocks, self).__init__()
        self.checkpoint = checkpoint
        self.device = _resolve_device(to_cuda, device)
        self.to_cuda = self.device.type == "cuda"
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        self.shapelet_weight = shapelet_weight
        if dist_measure == 'euclidean':
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                       in_channels=in_channels, to_cuda=self.to_cuda, device=self.device)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cross-correlation':
            self.blocks = nn.ModuleList(
                [MaxCrossCorrelationBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda, device=self.device)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'cosine':
            self.blocks = nn.ModuleList(
                [MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets,
                                          in_channels=in_channels, to_cuda=self.to_cuda, device=self.device)
                 for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()])
        elif dist_measure == 'mix':
            module_list = []
            for shapelets_size, num_shapelets in self.shapelets_size_and_len.items():
                module_list.append(MinEuclideanDistBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda, device=self.device))
                module_list.append(MaxCosineSimilarityBlock(shapelets_size=shapelets_size, num_shapelets=num_shapelets//3,
                                                         in_channels=in_channels, to_cuda=self.to_cuda, device=self.device))
                module_list.append(MaxCrossCorrelationBlock(shapelets_size=shapelets_size,
                                                            num_shapelets=num_shapelets - 2 * num_shapelets//3,
                                                            in_channels=in_channels, to_cuda=self.to_cuda, device=self.device))
            self.blocks = nn.ModuleList(module_list)

        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def _wrap_block_call(self, block, x, masking):
        """Call block optionally wrapped with BW profiling autograd.Function nodes."""
        if getattr(self, '_bw_profiling_enabled', False):
            x_in = _BwRangeIn.apply(x, block)
            block_out = block(x_in, masking)
            return _BwRangeOut.apply(block_out, block)
        else:
            return block(x, masking)

    def forward(self, x, masking=False):
        parts = []
        for block in self.blocks:
            if self.checkpoint and self.dist_measure != 'cross-correlation':
                parts.append(checkpoint(block, x, masking))
            else:
                parts.append(self._wrap_block_call(block, x, masking))
        return torch.cat(parts, dim=2) #[8,1,320]

    def forward_scale(self, x, scale_idx, masking=False):
        """仅前向一个尺度对应的 block。"""
        if self.dist_measure == 'mix':
            base = int(scale_idx) * 3
            block_indices = range(base, base + 3)
        else:
            block_indices = [int(scale_idx)]
        parts = []
        for idx in block_indices:
            block = self.blocks[idx]
            if self.checkpoint and self.dist_measure != 'cross-correlation':
                parts.append(checkpoint(block, x, masking))
            else:
                parts.append(self._wrap_block_call(block, x, masking))
        return torch.cat(parts, dim=2)

    def forward_subset(self, x, scale_indices, masking=False):
        parts = []
        for si in scale_indices:
            si = int(si)
            if self.dist_measure == 'mix':
                base = si * 3
                block_indices = range(base, base + 3)
            else:
                block_indices = [si]
            for idx in block_indices:
                block = self.blocks[idx]
                if self.checkpoint and self.dist_measure != 'cross-correlation':
                    parts.append(checkpoint(block, x, masking))
                else:
                    parts.append(self._wrap_block_call(block, x, masking))
        return torch.cat(parts, dim=2)

    # ---- per-scale memory aggregation (MAST-style, peak forward delta + param mem) ----
    def get_per_scale_peak_mem_mb(self) -> dict:
        """Return per-scale peak forward memory in MiB.

        For single distance measures: scale_i = block[i].peak_mem_mb.
        For 'mix': scale_i = euclidean[i] + cosine[i] + cross[i] (sum of 3 sub-blocks).

        Returns:
            OrderedDict[int, float]: scale_length → peak_mem_mb.
            Empty dict if no blocks have been forwarded yet.
        """
        lengths = list(self.shapelets_size_and_len.keys())
        if not lengths:
            return {}
        result = {}
        if self.dist_measure == 'mix':
            for i, L in enumerate(lengths):
                eu_mb = self.blocks[i * 3].peak_mem_mb
                co_mb = self.blocks[i * 3 + 1].peak_mem_mb
                cc_mb = self.blocks[i * 3 + 2].peak_mem_mb
                result[L] = eu_mb + co_mb + cc_mb
        else:
            for i, L in enumerate(lengths):
                result[L] = self.blocks[i].peak_mem_mb
        return result

    def get_per_scale_param_mem_mb(self) -> dict:
        """Return per-scale parameter memory in MiB (same grouping as peak mem).

        Returns:
            OrderedDict[int, float]: scale_length → param_mem_mb.
        """
        lengths = list(self.shapelets_size_and_len.keys())
        if not lengths:
            return {}
        result = {}
        if self.dist_measure == 'mix':
            for i, L in enumerate(lengths):
                eu_bytes = self.blocks[i * 3].param_mem_bytes
                co_bytes = self.blocks[i * 3 + 1].param_mem_bytes
                cc_bytes = self.blocks[i * 3 + 2].param_mem_bytes
                result[L] = (eu_bytes + co_bytes + cc_bytes) / (1024.0 * 1024.0)
        else:
            for i, L in enumerate(lengths):
                result[L] = float(self.blocks[i].param_mem_bytes) / (1024.0 * 1024.0)
        return result

    def get_per_scale_memory_summary(self) -> dict:
        """Combined per-scale memory summary (peak + param, all in MiB).

        Returns:
            dict with keys:
              - 'peak_mem_mb', 'param_mem_mb', 'total_mem_mb': per-scale values
              - 'per_scale_measured': bool per scale — ALL blocks for this scale
                have been forwarded at least once (peak memory is valid)
              - 'per_branch': (mix only) per-sub-module breakdown with measured flags
        """
        peak = self.get_per_scale_peak_mem_mb()
        param = self.get_per_scale_param_mem_mb()
        total = {}
        for L in peak:
            total[L] = peak.get(L, 0.0) + param.get(L, 0.0)

        # Per-scale measurement status: True only if ALL blocks for this scale were measured
        lengths = list(self.shapelets_size_and_len.keys())
        per_scale_measured = {}
        if self.dist_measure == 'mix':
            for i, L in enumerate(lengths):
                per_scale_measured[L] = (
                    self.blocks[i * 3].peak_mem_measured
                    and self.blocks[i * 3 + 1].peak_mem_measured
                    and self.blocks[i * 3 + 2].peak_mem_measured
                )
        else:
            for i, L in enumerate(lengths):
                per_scale_measured[L] = self.blocks[i].peak_mem_measured

        result = {
            "peak_mem_mb": peak,
            "param_mem_mb": param,
            "total_mem_mb": total,
            "per_scale_measured": per_scale_measured,
        }

        if self.dist_measure == 'mix':
            eu_peak, co_peak, cc_peak = {}, {}, {}
            eu_param, co_param, cc_param = {}, {}, {}
            eu_measured, co_measured, cc_measured = {}, {}, {}
            for i, L in enumerate(lengths):
                eu_peak[L] = self.blocks[i * 3].peak_mem_mb
                co_peak[L] = self.blocks[i * 3 + 1].peak_mem_mb
                cc_peak[L] = self.blocks[i * 3 + 2].peak_mem_mb
                eu_param[L] = float(self.blocks[i * 3].param_mem_bytes) / (1024.0 * 1024.0)
                co_param[L] = float(self.blocks[i * 3 + 1].param_mem_bytes) / (1024.0 * 1024.0)
                cc_param[L] = float(self.blocks[i * 3 + 2].param_mem_bytes) / (1024.0 * 1024.0)
                eu_measured[L] = self.blocks[i * 3].peak_mem_measured
                co_measured[L] = self.blocks[i * 3 + 1].peak_mem_measured
                cc_measured[L] = self.blocks[i * 3 + 2].peak_mem_measured
            result["per_branch"] = {
                "euclidean":  {"peak": eu_peak,  "param": eu_param,  "measured": eu_measured},
                "cosine":     {"peak": co_peak,  "param": co_param,  "measured": co_measured},
                "cross_corr": {"peak": cc_peak,  "param": cc_param,  "measured": cc_measured},
            }

        return result


class LearningShapeletsModel(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True, checkpoint=False,shapelet_weight=None, device=None):
        super(LearningShapeletsModel, self).__init__()

        self.device = _resolve_device(to_cuda, device)
        self.to_cuda = self.device.type == "cuda"
        self.checkpoint = checkpoint
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=self.to_cuda,
                                                    checkpoint=checkpoint, device=self.device)

        self.linear = nn.Linear(self.num_shapelets, num_classes)

        # LayerNorm：按特征维归一化，batch_size=1 也可用（BatchNorm1d 训练时要求 N>1）
        self.projection = nn.Sequential(nn.LayerNorm(self.num_shapelets),
                                            #   nn.Linear(self.model.num_shapelets, 256),
                                            #   nn.ReLU(),
                                            #   nn.Linear(self.num_shapelets, 128)
                                            # nn.Linear(self.num_shapelets, 256),
                                            # nn.ReLU(),
                                            # nn.Linear(256, 128)
                                            # 这里有buggggggggg！！！！！！！
                                        )

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))

        self.prodictor = nn.Sequential(nn.LayerNorm(self.num_shapelets),
                                        nn.Linear(self.num_shapelets, self.num_shapelets),
                                        nn.ReLU(),
                                        nn.Linear(self.num_shapelets, self.num_shapelets))

        if self.to_cuda:
            self.to(self.device)

    def _scale_feature_bounds(self, scale_idx):
        dims = list(self.shapelets_size_and_len.values())
        if scale_idx < 0 or scale_idx >= len(dims):
            raise IndexError(f"scale_idx={scale_idx} 超出范围 [0, {len(dims) - 1}]")
        start = int(sum(dims[:scale_idx]))
        end = start + int(dims[scale_idx])
        return start, end

    def slice_scale_features(self, feat, scale_idx):
        start, end = self._scale_feature_bounds(scale_idx)
        return feat[:, start:end]

    def _scale_state_prefixes(self, scale_idx):
        if self.shapelets_blocks.dist_measure == 'mix':
            base = int(scale_idx) * 3
            return [f"shapelets_blocks.blocks.{idx}." for idx in range(base, base + 3)]
        return [f"shapelets_blocks.blocks.{int(scale_idx)}."]

    def scale_state_dict(self, scale_idx, clone=True, cpu=False):
        prefixes = self._scale_state_prefixes(scale_idx)
        state = self.state_dict()
        picked = {}
        for key, value in state.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                tensor = value.detach()
                if clone:
                    tensor = tensor.clone()
                if cpu:
                    tensor = tensor.cpu()
                picked[key] = tensor
        return picked

    def encode_scale(self, x, scale_idx, masking=False, normalize=False):
        """仅编码一个尺度，输出该尺度的小模型特征。"""
        x = self.shapelets_blocks.forward_scale(x, scale_idx, masking)
        x = torch.squeeze(x, 1)
        if normalize:
            x = nn.functional.layer_norm(x, (x.shape[1],))
        return x

    def get_per_scale_memory_summary(self) -> dict:
        """Delegate to ShapeletsDistBlocks per-scale memory aggregation."""
        return self.shapelets_blocks.get_per_scale_memory_summary()

    def forward(self, x, optimize='acc', masking=False,isProdictor=False):


        # encoder
        x = self.shapelets_blocks(x, masking) #
        x = torch.squeeze(x, 1)  #[8,320]

        # 对embedding 进行切割并加权乘 再concat--------------------------------------------------------
        # weighted_tensors = []
        # split_tensors = torch.split(x, 40, dim=1)
        # for i, tensor in enumerate(split_tensors):
        #     # Convert the ith row of weights to a tensor with shape [1, -1] for broadcasting
        #     weight_tensor = torch.tensor(self.shapelet_weight[:, i]).view(-1, 1)

        #     # Multiply the tensor by the weight
        #     weighted_tensor = tensor * weight_tensor

        #     # Append the result to the list
        #     weighted_tensors.append(weighted_tensor)

        # # Step 3: Concatenate all weighted tensors back into one tensor of shape [8, 320]
        # x = torch.cat(weighted_tensors, dim=1)

        #---------------------------------------------------------------------------

        # test torch.cat
        #x = torch.cat((x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]), dim=1)

        if isProdictor:
            x = self.projection(x)
            x = self.prodictor(x)
            return x

        # projector
        x = self.projection(x)

        if optimize == 'acc':
            x = self.linear(x)


        return x # [batchsize, 320]





class LearningShapeletsModelMixDistances(nn.Module):

    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='mix',
                 to_cuda=True, checkpoint=False, device=None):
        super(LearningShapeletsModelMixDistances, self).__init__()

        self.checkpoint = checkpoint
        self.device = _resolve_device(to_cuda, device)
        self.to_cuda = self.device.type == "cuda"
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())

        self.shapelets_euclidean = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='euclidean', to_cuda=self.to_cuda,
                                                    checkpoint=checkpoint, device=self.device)


        self.shapelets_cosine = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] // 3 for item in shapelets_size_and_len.items()},
                                                    dist_measure='cosine', to_cuda=self.to_cuda,
                                                    checkpoint=checkpoint, device=self.device)

        self.shapelets_cross_correlation = ShapeletsDistBlocks(in_channels=in_channels,
                                                    shapelets_size_and_len={item[0]: item[1] - 2 * (item[1] // 3) for item in shapelets_size_and_len.items()},
                                                    dist_measure='cross-correlation', to_cuda=self.to_cuda,
                                                    checkpoint=checkpoint, device=self.device)


        self.linear = nn.Linear(self.num_shapelets, num_classes)

        self.projection = nn.Sequential(nn.LayerNorm(self.num_shapelets),
                                              #nn.Linear(self.model.num_shapelets, 256),
                                              #nn.ReLU(),
                                              #nn.Linear(self.num_shapelets, 128)
                                        )

        _n12 = sum(num // 3 for num in self.shapelets_size_and_len.values())
        _n3 = sum(num - 2 * (num // 3) for num in self.shapelets_size_and_len.values())
        self.ln1 = nn.LayerNorm(_n12)
        self.ln2 = nn.LayerNorm(_n12)
        self.ln3 = nn.LayerNorm(_n3)

        self.projection2 = nn.Sequential(nn.Linear(self.num_shapelets, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128))

        if self.to_cuda:
            self.to(self.device)

    def _scale_feature_bounds(self, scale_idx):
        dims = list(self.shapelets_size_and_len.values())
        if scale_idx < 0 or scale_idx >= len(dims):
            raise IndexError(f"scale_idx={scale_idx} 超出范围 [0, {len(dims) - 1}]")
        start = int(sum(dims[:scale_idx]))
        end = start + int(dims[scale_idx])
        return start, end

    def slice_scale_features(self, feat, scale_idx):
        start, end = self._scale_feature_bounds(scale_idx)
        return feat[:, start:end]

    def _scale_state_prefixes(self, scale_idx):
        scale_idx = int(scale_idx)
        return [
            f"shapelets_euclidean.blocks.{scale_idx}.",
            f"shapelets_cosine.blocks.{scale_idx}.",
            f"shapelets_cross_correlation.blocks.{scale_idx}.",
        ]

    def scale_state_dict(self, scale_idx, clone=True, cpu=False):
        prefixes = self._scale_state_prefixes(scale_idx)
        state = self.state_dict()
        picked = {}
        for key, value in state.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                tensor = value.detach()
                if clone:
                    tensor = tensor.clone()
                if cpu:
                    tensor = tensor.cpu()
                picked[key] = tensor
        return picked

    def encode_scale(self, x, scale_idx, masking=False, normalize=False):
        """仅编码一个尺度，保留 mix-distance 的三个分支，但只跑该尺度。"""
        eu = torch.squeeze(self.shapelets_euclidean.forward_scale(x, scale_idx, masking), 1)
        co = torch.squeeze(self.shapelets_cosine.forward_scale(x, scale_idx, masking), 1)
        cc = torch.squeeze(self.shapelets_cross_correlation.forward_scale(x, scale_idx, masking), 1)
        out = torch.cat((eu, co, cc), dim=1)
        if normalize:
            out = nn.functional.layer_norm(out, (out.shape[1],))
        return out

    def get_per_scale_memory_summary(self) -> dict:
        """Combine per-scale memory from all three mix-distance branches.

        Each scale = euclidean[scale_idx] + cosine[scale_idx] + cross[scale_idx].

        Returns:
            dict with keys:
              - peak_mem_mb / param_mem_mb / total_mem_mb: per-scale aggregated
              - per_scale_measured: bool per scale — ALL 3 sub-blocks forwarded
              - per_branch: dict with per-branch peak/param/measured by scale
        """
        lengths = list(self.shapelets_size_and_len.keys())
        if not lengths:
            return {"peak_mem_mb": {}, "param_mem_mb": {}, "total_mem_mb": {}}

        eu_blocks = self.shapelets_euclidean.blocks
        co_blocks = self.shapelets_cosine.blocks
        cc_blocks = self.shapelets_cross_correlation.blocks

        eu_peak = self.shapelets_euclidean.get_per_scale_peak_mem_mb()
        co_peak = self.shapelets_cosine.get_per_scale_peak_mem_mb()
        cc_peak = self.shapelets_cross_correlation.get_per_scale_peak_mem_mb()
        eu_param = self.shapelets_euclidean.get_per_scale_param_mem_mb()
        co_param = self.shapelets_cosine.get_per_scale_param_mem_mb()
        cc_param = self.shapelets_cross_correlation.get_per_scale_param_mem_mb()

        peak = {}
        param = {}
        total = {}
        per_scale_measured = {}
        eu_measured, co_measured, cc_measured = {}, {}, {}
        for i, L in enumerate(lengths):
            peak[L] = eu_peak.get(L, 0.0) + co_peak.get(L, 0.0) + cc_peak.get(L, 0.0)
            param[L] = eu_param.get(L, 0.0) + co_param.get(L, 0.0) + cc_param.get(L, 0.0)
            total[L] = peak[L] + param[L]
            eu_measured[L] = eu_blocks[i].peak_mem_measured
            co_measured[L] = co_blocks[i].peak_mem_measured
            cc_measured[L] = cc_blocks[i].peak_mem_measured
            per_scale_measured[L] = (
                eu_measured[L] and co_measured[L] and cc_measured[L]
            )

        per_branch = {
            "euclidean":  {"peak": eu_peak,  "param": eu_param,  "measured": eu_measured},
            "cosine":     {"peak": co_peak,  "param": co_param,  "measured": co_measured},
            "cross_corr": {"peak": cc_peak,  "param": cc_param,  "measured": cc_measured},
        }

        return {
            "peak_mem_mb": peak,
            "param_mem_mb": param,
            "total_mem_mb": total,
            "per_scale_measured": per_scale_measured,
            "per_branch": per_branch,
        }

    @staticmethod
    def _split_mix_branch_flat(flat, selected_scales_int, branch, shapelets_size_and_len):
        """将某分支上「所选尺度按顺序拼接」的向量切回各尺度行片段。"""
        dims = list(shapelets_size_and_len.values())
        parts = []
        offset = 0
        for si in selected_scales_int:
            num = dims[int(si)]
            if branch in ("eu", "co"):
                w = num // 3
            elif branch == "cc":
                w = num - 2 * (num // 3)
            else:
                raise ValueError(branch)
            parts.append(flat[:, offset : offset + w])
            offset += w
        if offset != flat.shape[1]:
            raise RuntimeError(
                f"分支 {branch} 拼接宽度不匹配：期望 {offset}，实际 {flat.shape[1]}"
            )
        return parts

    def encode_mix_forward_selected_scales(self, x, selected_scales, masking=False):
        """FedCSL ``forward(optimize=None)`` 同源顺序：分支 concat → LN → 按尺度切分 → 三分支沿特征维拼接 → 展平。

        与全模 ``forward`` 的区别仅为：各分支用 ``forward_subset`` 只计算 ``selected_scales``，
        LN 作用在「当前尺度集合在该分支上的拼接维」上（与 ``ln1``/``ln2``/``ln3`` 对全长向量归一化同源，
        但此处用 ``functional.layer_norm``，维度为子集宽度）。

        Returns:
            stitched: ``[B, sum_j dims[scale_j]]``，尺度顺序与 ``selected_scales`` 一致。
            per_scale_cat: 全局尺度索引 → 该尺度 mix 向量（与 ``slice_scale_features(stitched_global, si)``
            在全局次序下的切片语义一致，但此处 stitched 仅含所选尺度按序拼接）。
        """
        if not selected_scales:
            raise ValueError("selected_scales must be non-empty")
        scales_int = [int(s) for s in selected_scales]
        sd = self.shapelets_size_and_len

        eu_raw = torch.squeeze(self.shapelets_euclidean.forward_subset(x, scales_int, masking), 1)
        co_raw = torch.squeeze(self.shapelets_cosine.forward_subset(x, scales_int, masking), 1)
        cc_raw = torch.squeeze(self.shapelets_cross_correlation.forward_subset(x, scales_int, masking), 1)

        eu_ln = torch.nn.functional.layer_norm(eu_raw, (eu_raw.shape[1],))
        co_ln = torch.nn.functional.layer_norm(co_raw, (co_raw.shape[1],))
        cc_ln = torch.nn.functional.layer_norm(cc_raw, (cc_raw.shape[1],))

        eu_rows = self._split_mix_branch_flat(eu_ln, scales_int, "eu", sd)
        co_rows = self._split_mix_branch_flat(co_ln, scales_int, "co", sd)
        cc_rows = self._split_mix_branch_flat(cc_ln, scales_int, "cc", sd)

        mix_per_position = [
            torch.cat([eu_rows[i], co_rows[i], cc_rows[i]], dim=1) for i in range(len(scales_int))
        ]
        stitched = torch.cat(mix_per_position, dim=1)
        per_scale_cat = {scales_int[i]: mix_per_position[i] for i in range(len(scales_int))}
        return stitched, per_scale_cat

    def forward(self, x, optimize='acc', masking=False):



        n_samples = x.shape[0]
        num_lengths = len(self.shapelets_size_and_len)

        outs = []

        x_out = self.shapelets_euclidean(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.ln1(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        outs.append(x_out)

        x_out = self.shapelets_cosine(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.ln2(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        outs.append(x_out)

        x_out = self.shapelets_cross_correlation(x, masking)
        x_out = torch.squeeze(x_out, 1)
        #x_out = torch.nn.functional.normalize(x_out, dim=1)
        x_out = self.ln3(x_out)
        x_out = x_out.reshape(n_samples, num_lengths, -1)
        #print(x_out.shape)
        outs.append(x_out)


        out = torch.cat(outs, dim=2)
        out = out.reshape(n_samples, -1)



        #print(out.shape)
        #out = self.projection(out)

        if optimize == 'acc':
            out = self.linear(out)


        return out

