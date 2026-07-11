"""Small, explicit experiment configuration with fail-fast validation."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODALITIES = ("t1n", "t1c", "t2w", "t2f")


@dataclass(frozen=True)
class PatchConfig:
    """Physical footprint and the tensor shape presented to the patch stem."""

    footprint_mm: float = 4.0
    thin_mm: float = 1.0
    tensor_shape: tuple[int, int, int] = (16, 16, 1)

    def __post_init__(self) -> None:
        if self.footprint_mm <= 0 or self.thin_mm <= 0:
            raise ValueError("patch physical extents must be positive")
        if self.footprint_mm not in {4.0, 8.0, 16.0}:
            raise ValueError("patch footprint must be one of the registered 4, 8, or 16 mm scales")
        if self.thin_mm != 1.0:
            raise ValueError("v0 uses a 1 mm thin extent")
        if self.tensor_shape != (16, 16, 1):
            raise ValueError("v0 requires the model-visible patch shape to be 16x16x1")


@dataclass(frozen=True)
class ModelConfig:
    width: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    predictor_depth: int = 1
    teacher_ema_momentum: float = 0.996

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("model width, depth, and heads must be positive")
        if self.width % self.heads:
            raise ValueError("model width must be divisible by attention heads")
        if (self.width // self.heads) % 2:
            raise ValueError("attention head width must be even for rotary position encoding")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.predictor_depth != 1:
            raise ValueError("v0 deliberately fixes a shallow one-block predictor")
        if not 0.0 < self.teacher_ema_momentum < 1.0:
            raise ValueError("teacher EMA momentum must lie strictly between 0 and 1")


@dataclass(frozen=True)
class TaskConfig:
    modalities: tuple[str, ...] = MODALITIES
    positions_per_bag: int = 32
    objective: str = "match"
    allow_target_modality_elsewhere: bool = True
    allow_target_modality_at_target: bool = False
    pass_scan_statistics_to_teacher: bool = False

    def __post_init__(self) -> None:
        if tuple(self.modalities) != MODALITIES:
            raise ValueError(f"v0 requires modalities in canonical order {MODALITIES}")
        if self.positions_per_bag % len(self.modalities):
            raise ValueError("positions_per_bag must balance the hidden target modalities")
        if self.positions_per_bag < 2 * len(self.modalities):
            raise ValueError("positions_per_bag must provide at least two candidates per modality")
        if self.objective not in {"mae", "match", "both"}:
            raise ValueError("objective must be one of: mae, match, both")
        if self.allow_target_modality_at_target:
            raise ValueError(
                "the hidden target modality may never be visible at its target location"
            )
        if self.pass_scan_statistics_to_teacher:
            raise ValueError("v0 teacher API accepts a patch tensor and no ancillary statistics")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    patch: PatchConfig = field(default_factory=PatchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    checkpoint_every_steps: int = 1_000
    artifact_every_steps: int = 5_000

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.checkpoint_every_steps <= 0 or self.artifact_every_steps <= 0:
            raise ValueError("checkpoint and artifact cadences must be positive")
        if self.artifact_every_steps % self.checkpoint_every_steps:
            raise ValueError("artifact cadence must coincide with a checkpoint cadence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _strict_section(
    value: object,
    *,
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown keys in configuration section {name!r}: {sorted(unknown)}")
    return dict(value)


def experiment_config_from_dict(value: Mapping[str, Any]) -> ExperimentConfig:
    """Construct a fail-closed config from a decoded TOML mapping."""

    top_level = {
        "seed",
        "checkpoint_every_steps",
        "artifact_every_steps",
        "patch",
        "model",
        "task",
    }
    unknown = set(value) - top_level
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")

    patch_values = _strict_section(
        value.get("patch", {}),
        name="patch",
        allowed={"footprint_mm", "thin_mm", "tensor_shape"},
    )
    if "tensor_shape" in patch_values:
        patch_values["tensor_shape"] = tuple(patch_values["tensor_shape"])

    model_values = _strict_section(
        value.get("model", {}),
        name="model",
        allowed={
            "width",
            "depth",
            "heads",
            "mlp_ratio",
            "predictor_depth",
            "teacher_ema_momentum",
        },
    )
    task_values = _strict_section(
        value.get("task", {}),
        name="task",
        allowed={
            "modalities",
            "positions_per_bag",
            "objective",
            "allow_target_modality_elsewhere",
            "allow_target_modality_at_target",
            "pass_scan_statistics_to_teacher",
        },
    )
    if "modalities" in task_values:
        task_values["modalities"] = tuple(task_values["modalities"])

    scalar_values = {
        key: value[key] for key in top_level - {"patch", "model", "task"} if key in value
    }
    return ExperimentConfig(
        **scalar_values,
        patch=PatchConfig(**patch_values),
        model=ModelConfig(**model_values),
        task=TaskConfig(**task_values),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a TOML experiment configuration."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        value = tomllib.load(handle)
    return experiment_config_from_dict(value)
