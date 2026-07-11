"""Deterministic multimodal synthetic data for end-to-end launch verification."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from simple_brats.config import ExperimentConfig
from simple_brats.sampling import (
    CandidatePosition,
    SlabGeometry,
    plan_modality_completion_batch,
)

from .diagnostics import representation_stats, stats_by_modality
from .matching import (
    MatchingBatch,
    build_matching_system,
    optimizer_parameter_groups,
    validate_matching_batch,
)


def _candidate_centers(positions: int, shift: Tensor) -> tuple[tuple[float, float, float], ...]:
    columns = math.ceil(math.sqrt(positions))
    spacing_mm = 6.0
    return tuple(
        (
            float(shift[0]) + spacing_mm * (index % columns),
            float(shift[1]) + spacing_mm * (index // columns),
            float(shift[2]),
        )
        for index in range(positions)
    )


def _synthetic_modalities(
    *,
    batch_size: int,
    positions: int,
    patch_shape: tuple[int, int, int],
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(batch_size * positions, 1, *patch_shape, generator=generator)
    latent = F.avg_pool3d(latent, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
    latent = latent.reshape(batch_size, positions, *patch_shape)
    tissue = torch.randint(0, 4, (batch_size, positions, 1, 1, 1), generator=generator)
    tissue = tissue.to(latent.dtype)
    noise = lambda: 0.03 * torch.randn(latent.shape, generator=generator)  # noqa: E731

    t1n = torch.tanh(latent + 0.15 * tissue) + noise()
    t1c = torch.tanh(1.2 * latent + 0.45 * (tissue == 3)) + noise()
    t2w = torch.tanh(-0.7 * latent + 0.25 * tissue) + noise()
    t2f = torch.tanh(0.5 * latent.square() + 0.35 * (tissue == 2)) + noise()
    return torch.stack((t1n, t1c, t2w, t2f), dim=2)


def make_synthetic_matching_batch(
    config: ExperimentConfig,
    *,
    batch_size: int = 2,
    positions: int = 8,
) -> tuple[MatchingBatch, SlabGeometry]:
    """Create a deterministic batch that exercises the exact v0 sampler."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if positions < 2 * len(config.task.modalities) or positions % len(config.task.modalities):
        raise ValueError(
            "positions must be a modality-balanced count with at least two candidates per modality"
        )
    geometry = SlabGeometry(
        in_plane_footprint_mm=config.patch.footprint_mm,
        thin_extent_mm=config.patch.thin_mm,
        model_shape=config.patch.tensor_shape,
    )
    all_patches = _synthetic_modalities(
        batch_size=batch_size,
        positions=positions,
        patch_shape=config.patch.tensor_shape,
        seed=config.seed,
    )

    source_patches: list[Tensor] = []
    source_modalities: list[Tensor] = []
    source_positions: list[Tensor] = []
    source_coordinates: list[Tensor] = []
    target_patches: list[Tensor] = []
    target_modalities: list[Tensor] = []
    target_positions: list[Tensor] = []
    target_coordinates: list[Tensor] = []
    anchors: list[Tensor] = []
    bag_ids: list[Tensor] = []
    pair_ids: list[Tensor] = []

    for bag_index in range(batch_size):
        shift = torch.tensor([37.0 * bag_index + 0.5, -19.0 * bag_index + 1.25, 3.0 * bag_index])
        centers = _candidate_centers(positions, shift)
        candidates = tuple(
            CandidatePosition(position_id=index, center_mm=center)
            for index, center in enumerate(centers)
        )
        plan = plan_modality_completion_batch(
            candidates,
            batch_size=positions,
            geometry=geometry,
            rng=config.seed + bag_index,
        )
        sources = plan.visible_sources
        targets = plan.targets

        source_patches.append(
            torch.stack(
                [all_patches[bag_index, patch.position_id, patch.modality_id] for patch in sources]
            )
        )
        source_modalities.append(torch.tensor([patch.modality_id for patch in sources]))
        source_positions.append(torch.tensor([patch.position_id for patch in sources]))
        source_coordinates.append(torch.tensor([patch.center_mm for patch in sources]))
        target_patches.append(
            torch.stack(
                [all_patches[bag_index, patch.position_id, patch.modality_id] for patch in targets]
            )
        )
        target_modalities.append(torch.tensor([patch.modality_id for patch in targets]))
        target_positions.append(torch.tensor([patch.position_id for patch in targets]))
        target_coordinates.append(torch.tensor([patch.center_mm for patch in targets]))
        anchors.append(shift + torch.tensor([1.75, -2.25, 0.0]))
        bag_ids.append(torch.full((positions,), bag_index, dtype=torch.long))
        pair_ids.append(
            torch.tensor([bag_index * 1_000_000 + patch.position_id for patch in targets])
        )

    batch = MatchingBatch(
        source_patches=torch.stack(source_patches),
        source_modality_ids=torch.stack(source_modalities).long(),
        source_position_ids=torch.stack(source_positions).long(),
        source_coordinates_mm=torch.stack(source_coordinates).float(),
        query_modality_ids=torch.stack(target_modalities).long(),
        query_position_ids=torch.stack(target_positions).long(),
        query_coordinates_mm=torch.stack(target_coordinates).float(),
        query_bag_ids=torch.stack(bag_ids),
        query_pair_ids=torch.stack(pair_ids).long(),
        target_patches=torch.stack(target_patches),
        target_modality_ids=torch.stack(target_modalities).long(),
        target_position_ids=torch.stack(target_positions).long(),
        target_coordinates_mm=torch.stack(target_coordinates).float(),
        target_bag_ids=torch.stack(bag_ids),
        target_pair_ids=torch.stack(pair_ids).long(),
        anchor_mm=torch.stack(anchors).float(),
    )
    validate_matching_batch(batch, geometry=geometry)
    return batch, geometry


def run_synthetic_smoke(
    config: ExperimentConfig,
    *,
    device: torch.device | str,
    batch_size: int = 2,
    positions: int = 8,
    tiny_model: bool = False,
) -> dict[str, float | int | str]:
    """Run one real forward/backward/EMA step and return machine-readable metrics."""

    torch.manual_seed(config.seed)
    if tiny_model:
        config = replace(
            config,
            model=replace(config.model, width=24, depth=2, heads=3, mlp_ratio=2.0),
        )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")

    cpu_batch, _ = make_synthetic_matching_batch(
        config,
        batch_size=batch_size,
        positions=positions,
    )
    batch = cpu_batch.to(resolved_device)
    system = build_matching_system(config).to(resolved_device).train()
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(system, weight_decay=0.05),
        lr=1e-4,
    )

    ema_parameter_before = next(system.target_teacher.parameters()).detach().clone()
    output = system(batch)
    if not bool(torch.isfinite(output.loss)):
        raise RuntimeError("synthetic matching loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    output.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=10.0)
    if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0:
        raise RuntimeError("synthetic step produced no finite training gradient")
    optimizer.step()
    system.update_teacher()
    ema_parameter_after = next(system.target_teacher.parameters()).detach()
    ema_update = (ema_parameter_after - ema_parameter_before).float().norm()

    diagnostics = representation_stats(output.targets)
    modality_diagnostics = stats_by_modality(output.targets, batch.target_modality_ids)
    metrics: dict[str, float | int | str] = {
        "status": "ok",
        "device": str(resolved_device),
        "loss": float(output.loss.detach()),
        "accuracy": float(output.matching.accuracy),
        "chance": float(output.matching.chance),
        "gradient_norm": float(grad_norm),
        "ema_update_norm": float(ema_update),
        "teacher_updates": int(system.target_teacher.num_updates),
        "num_parameters": sum(parameter.numel() for parameter in system.parameters()),
        "batch_size": batch_size,
        "source_tokens_per_bag": batch.source_patches.shape[1],
        "target_tokens_per_bag": batch.target_patches.shape[1],
        **{f"target_{name}": value for name, value in diagnostics.to_dict().items()},
    }
    for modality_id, stats in modality_diagnostics.items():
        metrics.update(
            {
                f"target_modality_{modality_id}_{name}": value
                for name, value in stats.to_dict().items()
            }
        )
    return metrics
