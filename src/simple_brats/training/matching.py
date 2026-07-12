"""Leakage-checked wiring for contextual prediction against a blind EMA target."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor, nn

from simple_brats.config import ExperimentConfig
from simple_brats.models import (
    CrossModalEncoder,
    EMATeacher,
    EncoderConfig,
    EncoderStemPatchTeacher,
    TargetModalityPredictor,
)
from simple_brats.objectives import MatchingOutput, hard_symmetric_info_nce
from simple_brats.sampling import SlabGeometry


@dataclass(frozen=True)
class MatchingBatch:
    """One batch with explicit identities independent of tensor order."""

    source_patches: Tensor
    source_modality_ids: Tensor
    source_position_ids: Tensor
    source_coordinates_mm: Tensor
    query_modality_ids: Tensor
    query_position_ids: Tensor
    query_coordinates_mm: Tensor
    query_bag_ids: Tensor
    query_pair_ids: Tensor
    target_patches: Tensor
    target_modality_ids: Tensor
    target_position_ids: Tensor
    target_coordinates_mm: Tensor
    target_bag_ids: Tensor
    target_pair_ids: Tensor
    anchor_mm: Tensor
    source_padding_mask: Tensor | None = None

    def to(self, device: torch.device | str) -> MatchingBatch:
        values = {
            field.name: (
                value.to(device)
                if isinstance((value := getattr(self, field.name)), Tensor)
                else value
            )
            for field in fields(self)
        }
        return MatchingBatch(**values)


def validate_matching_batch(
    batch: MatchingBatch,
    *,
    geometry: SlabGeometry,
) -> None:
    """Reject target leakage and malformed tensor/identity tables.

    This validation is intended for CPU batches immediately after collation.
    It checks exact physical same-modality exclusion in addition to ordinary
    tensor shapes.  Closed-patch contact is considered overlap.
    """

    if (
        batch.source_modality_ids.ndim != 2
        or batch.query_modality_ids.ndim != 2
        or batch.target_modality_ids.ndim != 2
    ):
        raise ValueError("source, query, and target modality IDs must be rank-two")
    batch_size, n_sources = batch.source_modality_ids.shape
    query_batch, n_queries = batch.query_modality_ids.shape
    target_batch, n_targets = batch.target_modality_ids.shape
    if query_batch != batch_size or target_batch != batch_size:
        raise ValueError("source, query, and target batch dimensions must match")
    if n_queries != 32 or n_targets != 32:
        raise ValueError("ordering batches require exactly 32 queries and teacher targets")
    if n_sources != 96:
        raise ValueError("ordering batches require exactly 96 source patches")
    if tuple(batch.source_patches.shape[:2]) != (batch_size, n_sources):
        raise ValueError("source patch table does not match source metadata")
    if tuple(batch.target_patches.shape[:2]) != (batch_size, n_targets):
        raise ValueError("target patch table does not match target metadata")
    expected_shape = geometry.model_shape
    if tuple(batch.source_patches.shape[-3:]) != expected_shape:
        raise ValueError(f"source patches must end in model shape {expected_shape}")
    if tuple(batch.target_patches.shape[-3:]) != expected_shape:
        raise ValueError(f"target patches must end in model shape {expected_shape}")
    if batch.source_position_ids.shape != (batch_size, n_sources):
        raise ValueError("source_position_ids must have shape [batch, sources]")
    if batch.query_position_ids.shape != (batch_size, n_queries):
        raise ValueError("query_position_ids must have shape [batch, queries]")
    if batch.target_position_ids.shape != (batch_size, n_targets):
        raise ValueError("target_position_ids must have shape [batch, targets]")
    if batch.source_coordinates_mm.shape != (batch_size, n_sources, 3):
        raise ValueError("source_coordinates_mm must have shape [batch, sources, 3]")
    if batch.target_coordinates_mm.shape != (batch_size, n_targets, 3):
        raise ValueError("target_coordinates_mm must have shape [batch, targets, 3]")
    if batch.query_coordinates_mm.shape != (batch_size, n_queries, 3):
        raise ValueError("query_coordinates_mm must have shape [batch, queries, 3]")
    if batch.anchor_mm.shape != (batch_size, 3):
        raise ValueError("anchor_mm must have shape [batch, 3]")
    if batch.query_bag_ids.shape != (batch_size, n_queries):
        raise ValueError("query_bag_ids must have shape [batch, queries]")
    if batch.query_pair_ids.shape != (batch_size, n_queries):
        raise ValueError("query_pair_ids must have shape [batch, queries]")
    if batch.target_bag_ids.shape != (batch_size, n_targets):
        raise ValueError("target_bag_ids must have shape [batch, targets]")
    if batch.target_pair_ids.shape != (batch_size, n_targets):
        raise ValueError("target_pair_ids must have shape [batch, targets]")
    if batch.source_padding_mask is not None:
        if batch.source_padding_mask.shape != (batch_size, n_sources):
            raise ValueError("source_padding_mask must have shape [batch, sources]")
        if batch.source_padding_mask.dtype != torch.bool:
            raise TypeError("source_padding_mask must be boolean")

    for tensor_name in (
        "source_modality_ids",
        "source_position_ids",
        "query_modality_ids",
        "query_position_ids",
        "query_bag_ids",
        "query_pair_ids",
        "target_modality_ids",
        "target_position_ids",
        "target_bag_ids",
        "target_pair_ids",
    ):
        tensor = getattr(batch, tensor_name)
        if tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"{tensor_name} must contain integer IDs")
    for tensor_name in ("source_modality_ids", "query_modality_ids", "target_modality_ids"):
        tensor = getattr(batch, tensor_name)
        if tensor.numel() and (int(tensor.min()) < 0 or int(tensor.max()) >= 4):
            raise ValueError(f"{tensor_name} must use the v0 modality IDs 0 through 3")

    padding = (
        batch.source_padding_mask
        if batch.source_padding_mask is not None
        else torch.zeros_like(batch.source_modality_ids, dtype=torch.bool)
    )
    if bool(padding.any()):
        raise ValueError("ordering batches require 96 real sources without padding")
    for tensor_name in (
        "source_patches",
        "source_coordinates_mm",
        "query_coordinates_mm",
        "target_patches",
        "target_coordinates_mm",
        "anchor_mm",
    ):
        if not bool(torch.isfinite(getattr(batch, tensor_name)).all()):
            raise ValueError(f"{tensor_name} must contain only finite values")

    same_position = batch.source_position_ids[:, :, None] == batch.query_position_ids[:, None, :]
    same_modality = batch.source_modality_ids[:, :, None] == batch.query_modality_ids[:, None, :]
    exact_answer_visible = same_position & same_modality & ~padding[:, :, None]
    if bool(exact_answer_visible.any()):
        raise ValueError("hidden target modality is visible at its target position")

    # Two equal-size closed patches intersect when their center separation is at
    # most the full extent on every axis.  Non-intersection therefore requires
    # a strict greater-than separation on at least one axis.
    deltas = (
        batch.source_coordinates_mm[:, :, None, :] - batch.query_coordinates_mm[:, None, :, :]
    ).abs()
    extents = deltas.new_tensor(geometry.extents_mm)
    patches_intersect = (deltas <= extents).all(dim=-1)
    same_modality_overlap = same_modality & patches_intersect & ~padding[:, :, None]
    if bool(same_modality_overlap.any()):
        raise ValueError("a visible target-modality patch intersects a held target patch")

    target_deltas = (
        batch.target_coordinates_mm[:, :, None, :] - batch.target_coordinates_mm[:, None, :, :]
    ).abs()
    target_intersection = (target_deltas <= extents).all(dim=-1)
    diagonal = torch.eye(n_targets, dtype=torch.bool, device=target_intersection.device)[None]
    if bool((target_intersection & ~diagonal).any()):
        raise ValueError("teacher target patches must be pairwise non-intersecting")

    for bag_index in range(batch_size):
        query_keys = list(
            zip(
                batch.query_bag_ids[bag_index].tolist(),
                batch.query_modality_ids[bag_index].tolist(),
                batch.query_pair_ids[bag_index].tolist(),
                strict=True,
            )
        )
        target_keys = list(
            zip(
                batch.target_bag_ids[bag_index].tolist(),
                batch.target_modality_ids[bag_index].tolist(),
                batch.target_pair_ids[bag_index].tolist(),
                strict=True,
            )
        )
        if len(set(query_keys)) != n_queries or len(set(target_keys)) != n_targets:
            raise ValueError("query and target pair identities must be unique")
        if set(query_keys) != set(target_keys):
            raise ValueError("query and target identity tables must describe the same pairs")

        query_by_key = {
            key: (
                int(batch.query_position_ids[bag_index, index]),
                batch.query_coordinates_mm[bag_index, index],
            )
            for index, key in enumerate(query_keys)
        }
        target_by_key = {
            key: (
                int(batch.target_position_ids[bag_index, index]),
                batch.target_coordinates_mm[bag_index, index],
            )
            for index, key in enumerate(target_keys)
        }
        for key in query_by_key:
            query_position, query_coordinate = query_by_key[key]
            target_position, target_coordinate = target_by_key[key]
            if query_position != target_position or not torch.equal(
                query_coordinate, target_coordinate
            ):
                raise ValueError("paired query and target records must share one physical location")

        query_modalities = set(batch.query_modality_ids[bag_index].tolist())
        target_modalities = set(batch.target_modality_ids[bag_index].tolist())
        if len(query_modalities) != 1 or target_modalities != query_modalities:
            raise ValueError("all queries and targets must use one shared target modality")
        target_modality_id = next(iter(query_modalities))
        source_counts = torch.bincount(batch.source_modality_ids[bag_index], minlength=4)
        expected_source_counts = torch.full_like(source_counts, 30)
        expected_source_counts[target_modality_id] = 6
        if source_counts.numel() != 4 or not torch.equal(
            source_counts,
            expected_source_counts,
        ):
            raise ValueError(
                "sources must contain 6 target-modality patches and "
                "30 patches from each other modality"
            )

        source_keys = list(
            zip(
                batch.source_position_ids[bag_index].tolist(),
                batch.source_modality_ids[bag_index].tolist(),
                strict=True,
            )
        )
        if len(set(source_keys)) != n_sources:
            raise ValueError("source (position_id, modality_id) identities must be unique")
        if len(set(batch.query_position_ids[bag_index].tolist())) != n_queries:
            raise ValueError("query targets must use 32 distinct position IDs")

        coordinates_by_position: dict[int, Tensor] = {}
        identity_tables = (
            (
                batch.source_position_ids[bag_index],
                batch.source_coordinates_mm[bag_index],
            ),
            (
                batch.query_position_ids[bag_index],
                batch.query_coordinates_mm[bag_index],
            ),
            (
                batch.target_position_ids[bag_index],
                batch.target_coordinates_mm[bag_index],
            ),
        )
        for position_ids, coordinates in identity_tables:
            for position_id, coordinate in zip(
                position_ids.tolist(),
                coordinates,
                strict=True,
            ):
                known = coordinates_by_position.setdefault(position_id, coordinate)
                if not torch.equal(known, coordinate):
                    raise ValueError("a repeated position_id maps to inconsistent coordinates")


@dataclass(frozen=True)
class MatchingStepOutput:
    matching: MatchingOutput
    predictions: Tensor
    targets: Tensor
    source_tokens: Tensor

    @property
    def loss(self) -> Tensor:
        return self.matching.loss


class CrossModalMatchingSystem(nn.Module):
    """Joint source encoder, shallow predictor, and patch-only EMA target."""

    def __init__(
        self,
        encoder: CrossModalEncoder,
        predictor: TargetModalityPredictor,
        *,
        teacher_momentum: float,
        geometry: SlabGeometry,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.online_patch_view = EncoderStemPatchTeacher.from_encoder(encoder)
        self.target_teacher = EMATeacher(
            self.online_patch_view,
            momentum=teacher_momentum,
        )
        self.geometry = geometry
        self.temperature = temperature

    def forward(self, batch: MatchingBatch) -> MatchingStepOutput:
        validate_matching_batch(batch, geometry=self.geometry)
        source_tokens = self.encoder(
            batch.source_patches,
            batch.source_modality_ids,
            batch.source_coordinates_mm,
            batch.anchor_mm,
            batch.source_padding_mask,
        )
        predictions = self.predictor(
            source_tokens,
            batch.source_coordinates_mm,
            batch.query_coordinates_mm,
            batch.query_modality_ids,
            batch.anchor_mm,
            batch.source_padding_mask,
        )
        targets = self.target_teacher(batch.target_patches)
        matching = hard_symmetric_info_nce(
            predictions,
            targets,
            prediction_bag_ids=batch.query_bag_ids,
            prediction_modality_ids=batch.query_modality_ids,
            prediction_pair_ids=batch.query_pair_ids,
            target_bag_ids=batch.target_bag_ids,
            target_modality_ids=batch.target_modality_ids,
            target_pair_ids=batch.target_pair_ids,
            temperature=self.temperature,
        )
        return MatchingStepOutput(
            matching=matching,
            predictions=predictions,
            targets=targets,
            source_tokens=source_tokens,
        )

    @torch.no_grad()
    def update_teacher(self) -> None:
        self.target_teacher.update(self.online_patch_view)


def build_matching_system(config: ExperimentConfig) -> CrossModalMatchingSystem:
    """Build the v0 matching model directly from the resolved config."""

    if config.task.objective != "match":
        raise NotImplementedError(
            f"objective {config.task.objective!r} is specified but not implemented; "
            "the runner will not silently substitute matching"
        )
    encoder = CrossModalEncoder(
        EncoderConfig(
            num_modalities=len(config.task.modalities),
            patch_shape=config.patch.tensor_shape,
            embed_dim=config.model.width,
            depth=config.model.depth,
            num_heads=config.model.heads,
            mlp_ratio=config.model.mlp_ratio,
        )
    )
    predictor = TargetModalityPredictor(
        num_modalities=len(config.task.modalities),
        embed_dim=config.model.width,
        output_dim=config.model.width,
        depth=config.model.predictor_depth,
        num_heads=config.model.heads,
    )
    geometry = SlabGeometry(
        in_plane_footprint_mm=config.patch.footprint_mm,
        thin_extent_mm=config.patch.thin_mm,
        model_shape=config.patch.tensor_shape,
    )
    return CrossModalMatchingSystem(
        encoder,
        predictor,
        teacher_momentum=config.model.teacher_ema_momentum,
        geometry=geometry,
    )


def optimizer_parameter_groups(
    module: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Separate matrix weights from biases, norms, and identity embeddings."""

    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("optimizer grouping unexpectedly produced an empty parameter group")
    if set(map(id, decay)) & set(map(id, no_decay)):
        raise AssertionError("a parameter appeared in both optimizer groups")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
