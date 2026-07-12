"""Small, explicit experiment configuration with fail-fast validation."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODALITIES = ("t1n", "t1c", "t2w", "t2f")
REGISTERED_CUBE_TENSOR_SHAPE = (8, 8, 8)
LEGACY_CUBE_TENSOR_SHAPE = (16, 16, 16)
REGISTERED_SINGLE_D_SCALE_ARMS = {
    ((32.0, 32.0, 32.0), 4.0): "32mm-prism_4mm-cube",
    ((64.0, 64.0, 64.0), 8.0): "64mm-prism_8mm-cube",
}


@dataclass(frozen=True)
class PatchConfig:
    """Physical footprint and the tensor shape presented to the patch stem."""

    footprint_mm: float = 4.0
    thin_mm: float = 4.0
    tensor_shape: tuple[int, int, int] = REGISTERED_CUBE_TENSOR_SHAPE

    def __post_init__(self) -> None:
        if self.footprint_mm <= 0 or self.thin_mm <= 0:
            raise ValueError("patch physical extents must be positive")
        legacy_slab = (
            self.footprint_mm == 4.0
            and self.thin_mm == 1.0
            and self.tensor_shape == (16, 16, 1)
        )
        registered_cube = (
            self.footprint_mm in {4.0, 8.0}
            and self.thin_mm == self.footprint_mm
            and self.tensor_shape
            in {REGISTERED_CUBE_TENSOR_SHAPE, LEGACY_CUBE_TENSOR_SHAPE}
        )
        if not (legacy_slab or registered_cube):
            raise ValueError(
                "patch must be a 4 or 8 mm isotropic cube presented as 8x8x8 "
                "or legacy 16x16x16 (or the load-only 4x4x1 mm / 16x16x1 slab)"
            )

    @property
    def physical_extent_mm(self) -> tuple[float, float, float]:
        """Full physical extent ordered by prepared-grid axis."""

        return (self.footprint_mm, self.footprint_mm, self.thin_mm)

    @property
    def source_shape(self) -> tuple[int, int, int]:
        """Integer crop shape on the registered 1 mm prepared grid."""

        return tuple(int(extent) for extent in self.physical_extent_mm)  # type: ignore[return-value]

    @property
    def is_cubic(self) -> bool:
        return self.physical_extent_mm[0] == self.physical_extent_mm[2]


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
    prism_extent_mm: tuple[float, float, float] = (32.0, 32.0, 32.0)
    target_patches_per_bag: int = 32
    context_patches_per_nontarget_modality: int = 30
    context_patches_target_modality: int = 6
    objective: str = "match"
    allow_target_modality_elsewhere: bool = True
    allow_target_modality_at_target: bool = False
    pass_scan_statistics_to_teacher: bool = False

    def __post_init__(self) -> None:
        if tuple(self.modalities) != MODALITIES:
            raise ValueError(f"v0 requires modalities in canonical order {MODALITIES}")
        try:
            prism_extent_mm = tuple(float(value) for value in self.prism_extent_mm)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("prism_extent_mm must contain three finite extents") from error
        if (
            len(prism_extent_mm) != 3
            or not all(math.isfinite(value) and value > 0 for value in prism_extent_mm)
            or len(set(prism_extent_mm)) != 1
        ):
            raise ValueError("prism_extent_mm must describe one finite positive cube")
        object.__setattr__(self, "prism_extent_mm", prism_extent_mm)
        for value, name in (
            (self.target_patches_per_bag, "target_patches_per_bag"),
            (
                self.context_patches_per_nontarget_modality,
                "context_patches_per_nontarget_modality",
            ),
            (self.context_patches_target_modality, "context_patches_target_modality"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_patches_per_bag != 32:
            raise ValueError("the single-D task requires exactly 32 target patches per bag")
        if self.context_patches_per_nontarget_modality != 30:
            raise ValueError(
                "the single-D task requires 30 context patches for each non-target modality"
            )
        if self.context_patches_target_modality != 6:
            raise ValueError("the single-D task requires 6 target-modality context patches")
        if self.objective not in {"mae", "match", "both"}:
            raise ValueError("objective must be one of: mae, match, both")
        if self.allow_target_modality_at_target:
            raise ValueError(
                "the hidden target modality may never be visible at its target location"
            )
        if self.pass_scan_statistics_to_teacher:
            raise ValueError("v0 teacher API accepts a patch tensor and no ancillary statistics")

    @property
    def positions_per_bag(self) -> int:
        """Compatibility name for the number of target/query identities."""

        return self.target_patches_per_bag

    @property
    def source_patches_per_bag(self) -> int:
        """Exact source-token count for one sampled target modality D."""

        return self.context_patches_target_modality + (
            (len(self.modalities) - 1) * self.context_patches_per_nontarget_modality
        )


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

    @property
    def registered_single_d_arm(self) -> str | None:
        """Return the exact launch arm name, or ``None`` for development configs."""

        arm = REGISTERED_SINGLE_D_SCALE_ARMS.get(
            (self.task.prism_extent_mm, self.patch.footprint_mm)
        )
        if arm is None:
            return None
        exact = (
            self.seed == 0
            and self.checkpoint_every_steps == 1_000
            and self.artifact_every_steps == 5_000
            and self.patch.thin_mm == self.patch.footprint_mm
            and self.patch.tensor_shape == REGISTERED_CUBE_TENSOR_SHAPE
            and self.model
            == ModelConfig(
                width=256,
                depth=8,
                heads=4,
                mlp_ratio=4.0,
                predictor_depth=1,
                teacher_ema_momentum=0.996,
            )
            and self.task.modalities == MODALITIES
            and self.task.target_patches_per_bag == 32
            and self.task.context_patches_per_nontarget_modality == 30
            and self.task.context_patches_target_modality == 6
            and self.task.source_patches_per_bag == 96
            and self.task.objective == "match"
            and self.task.allow_target_modality_elsewhere
            and not self.task.allow_target_modality_at_target
            and not self.task.pass_scan_statistics_to_teacher
        )
        return arm if exact else None


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
            "prism_extent_mm",
            "target_patches_per_bag",
            "context_patches_per_nontarget_modality",
            "context_patches_target_modality",
            "objective",
            "allow_target_modality_elsewhere",
            "allow_target_modality_at_target",
            "pass_scan_statistics_to_teacher",
        },
    )
    if "modalities" in task_values:
        task_values["modalities"] = tuple(task_values["modalities"])
    if "prism_extent_mm" in task_values:
        task_values["prism_extent_mm"] = tuple(task_values["prism_extent_mm"])

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
