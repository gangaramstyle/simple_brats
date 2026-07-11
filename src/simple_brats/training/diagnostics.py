"""Representation-collapse diagnostics with explicit, externally locked thresholds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RepresentationStats:
    count: int
    variance: float
    effective_rank: float
    off_diagonal_cosine: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, Integral):
            raise TypeError("count must be an integer")
        if self.count < 2:
            raise ValueError("count must be at least two")
        for name in ("variance", "effective_rank", "off_diagonal_cosine"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.variance < 0:
            raise ValueError("variance must be non-negative")
        if not 1 <= self.effective_rank <= self.count:
            raise ValueError("effective_rank must lie in [1, count]")
        if not -1 <= self.off_diagonal_cosine <= 1:
            raise ValueError("off_diagonal_cosine must lie in [-1, 1]")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "variance": float(self.variance),
            "effective_rank": float(self.effective_rank),
            "off_diagonal_cosine": float(self.off_diagonal_cosine),
        }


@dataclass(frozen=True)
class CollapseThresholds:
    """Thresholds must be selected from baselines before real SSL results."""

    minimum_variance_ratio: float
    minimum_effective_rank_ratio: float
    maximum_off_diagonal_cosine: float

    def __post_init__(self) -> None:
        for name in (
            "minimum_variance_ratio",
            "minimum_effective_rank_ratio",
            "maximum_off_diagonal_cosine",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.minimum_variance_ratio <= 1:
            raise ValueError("minimum_variance_ratio must lie in (0, 1]")
        if not 0 < self.minimum_effective_rank_ratio <= 1:
            raise ValueError("minimum_effective_rank_ratio must lie in (0, 1]")
        if not -1 <= self.maximum_off_diagonal_cosine < 1:
            raise ValueError("maximum_off_diagonal_cosine must lie in [-1, 1)")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_variance_ratio": float(self.minimum_variance_ratio),
            "minimum_effective_rank_ratio": float(self.minimum_effective_rank_ratio),
            "maximum_off_diagonal_cosine": float(self.maximum_off_diagonal_cosine),
        }


def representation_stats(features: Tensor) -> RepresentationStats:
    """Measure rank and concentration without changing the training graph."""

    if features.ndim < 2 or features.shape[-1] < 2:
        raise ValueError("features must contain an embedding dimension of at least two")
    flat = features.detach().float().reshape(-1, features.shape[-1])
    if flat.shape[0] < 2:
        raise ValueError("at least two feature vectors are required")
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("features must be finite")

    variance = flat.var(dim=0, unbiased=False).mean()
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum()
    )
    normalized = F.normalize(flat, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    count = flat.shape[0]
    off_diagonal = (similarities.sum() - similarities.diagonal().sum()) / (count * (count - 1))
    # Clamp only floating-point roundoff at the mathematical boundaries.  The
    # dataclass validates externally supplied references more strictly.
    variance_value = max(float(variance), 0.0)
    effective_rank_value = min(max(float(effective_rank), 1.0), float(count))
    off_diagonal_value = min(max(float(off_diagonal), -1.0), 1.0)
    return RepresentationStats(
        count=count,
        variance=variance_value,
        effective_rank=effective_rank_value,
        off_diagonal_cosine=off_diagonal_value,
    )


def stats_by_modality(features: Tensor, modality_ids: Tensor) -> dict[int, RepresentationStats]:
    """Compute diagnostics separately so one healthy modality cannot hide another's collapse."""

    flat = features.reshape(-1, features.shape[-1])
    flat_ids = modality_ids.reshape(-1).to(device=flat.device)
    if flat.shape[0] != flat_ids.numel():
        raise ValueError("modality IDs must align one-to-one with feature vectors")
    result: dict[int, RepresentationStats] = {}
    for modality_id in flat_ids.unique(sorted=True).tolist():
        selected = flat[flat_ids == modality_id]
        result[int(modality_id)] = representation_stats(selected)
    return result


def collapse_reasons(
    current: RepresentationStats,
    reference: RepresentationStats,
    thresholds: CollapseThresholds,
) -> tuple[str, ...]:
    """Return deterministic abort reasons relative to a locked reference."""

    reasons: list[str] = []
    variance_ratio = current.variance / max(reference.variance, torch.finfo(torch.float32).eps)
    rank_ratio = current.effective_rank / max(
        reference.effective_rank, torch.finfo(torch.float32).eps
    )
    if variance_ratio < thresholds.minimum_variance_ratio:
        reasons.append("variance_ratio")
    if rank_ratio < thresholds.minimum_effective_rank_ratio:
        reasons.append("effective_rank_ratio")
    if current.off_diagonal_cosine > thresholds.maximum_off_diagonal_cosine:
        reasons.append("off_diagonal_cosine")
    return tuple(reasons)
