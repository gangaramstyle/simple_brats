"""Deterministic, resumable training loop for cross-modal matching.

The runner deliberately owns no experiment tracker.  Checkpoint and artifact
cadence are delegated to :class:`CheckpointManager`, which means cluster runs
can use an offline artifact sink without introducing network access here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from simple_brats.atomic_io import fsync_file_and_parent

from .checkpoints import CheckpointManager
from .diagnostics import (
    CollapseThresholds,
    RepresentationStats,
    collapse_reasons,
    stats_by_modality,
)
from .matching import CrossModalMatchingSystem, MatchingBatch
from .runtime import TrainingRuntimePolicy

_RUNNER_SCHEMA_VERSION = 3
_RUNNER_CONTRACT_SCHEMA_VERSION = 4
_DIAGNOSTICS_EVERY_STEPS = 50

TEACHER_TARGET_DIAGNOSTIC_STREAM = "fixed_probe_ema_teacher_targets_post_update"
TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM = "training_batch_ema_teacher_targets_pre_update"
PREDICTION_DIAGNOSTIC_STREAM = "training_batch_online_predictions_pre_update"


@dataclass(frozen=True)
class FixedTargetPatchProbe:
    """Immutable CPU copy of the exact patches used for collapse decisions.

    The SHA binds tensor metadata, float32 patch bytes, and int64 modality IDs.
    The runner takes a second defensive copy before training so caller mutation
    cannot change an active probe after its contract has been hashed.
    """

    target_patches: Tensor
    target_modality_ids: Tensor

    def __post_init__(self) -> None:
        patches = self.target_patches
        modality_ids = self.target_modality_ids
        if not isinstance(patches, Tensor) or not isinstance(modality_ids, Tensor):
            raise TypeError("fixed probe patches and modality IDs must be tensors")
        if patches.dtype != torch.float32:
            raise TypeError("fixed probe patches must use float32")
        if patches.ndim not in (5, 6):
            raise ValueError(
                "fixed probe patches must have shape [batch, patches, (channels), D, H, W]"
            )
        if modality_ids.ndim != 2 or tuple(patches.shape[:2]) != tuple(modality_ids.shape):
            raise ValueError("fixed probe modality IDs must align with its patch table")
        if modality_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("fixed probe modality IDs must contain integers")
        if patches.shape[0] <= 0 or patches.shape[1] <= 0:
            raise ValueError("fixed probe must contain at least one patch")
        if not bool(torch.isfinite(patches).all()):
            raise ValueError("fixed probe patches must be finite")
        if modality_ids.numel() and int(modality_ids.min()) < 0:
            raise ValueError("fixed probe modality IDs must be non-negative")

        patches = patches.detach().to(device="cpu").contiguous().clone()
        modality_ids = (
            modality_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()
        )
        counts = torch.bincount(modality_ids.reshape(-1))
        observed_counts = counts[counts > 0]
        if observed_counts.numel() == 0 or int(observed_counts.min()) < 2:
            raise ValueError("fixed probe requires at least two patches per observed modality")
        object.__setattr__(self, "target_patches", patches)
        object.__setattr__(self, "target_modality_ids", modality_ids)

    @property
    def sample_count_by_modality(self) -> dict[int, int]:
        values, counts = self.target_modality_ids.reshape(-1).unique(
            sorted=True,
            return_counts=True,
        )
        return {
            int(modality_id): int(count)
            for modality_id, count in zip(values.tolist(), counts.tolist(), strict=True)
        }

    @property
    def sha256(self) -> str:
        header = json.dumps(
            {
                "schema": "simple-brats.fixed-target-patch-probe",
                "schema_version": 1,
                "patch_dtype": "float32",
                "patch_shape": list(self.target_patches.shape),
                "modality_id_dtype": "int64",
                "modality_id_shape": list(self.target_modality_ids.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(header)
        digest.update(b"\0patches\0")
        digest.update(self.target_patches.numpy().tobytes(order="C"))
        digest.update(b"\0modality_ids\0")
        digest.update(self.target_modality_ids.numpy().tobytes(order="C"))
        return digest.hexdigest()


class TrainingRunnerError(RuntimeError):
    """The run cannot continue without violating its training contract."""


class RepresentationCollapseError(TrainingRunnerError):
    """At least one modality crossed a pre-registered collapse boundary."""

    def __init__(
        self,
        *,
        step: int,
        reasons_by_modality: Mapping[int, tuple[str, ...]],
        diagnostics_by_modality: Mapping[int, RepresentationStats],
        checkpoint_path: Path,
    ) -> None:
        self.step = step
        self.reasons_by_modality = dict(reasons_by_modality)
        self.diagnostics_by_modality = dict(diagnostics_by_modality)
        self.diagnostic_stream = TEACHER_TARGET_DIAGNOSTIC_STREAM
        self.checkpoint_path = checkpoint_path
        details = ", ".join(
            f"modality {modality_id}: {','.join(reasons)}"
            for modality_id, reasons in sorted(self.reasons_by_modality.items())
        )
        super().__init__(f"representation collapse at completed step {step}: {details}")


@dataclass(frozen=True)
class StepMetrics:
    """Small, tracker-agnostic record emitted after a completed optimizer step."""

    step: int
    loss: float
    accuracy: float
    chance: float
    ema_update_count: int
    diagnostics_by_stream: Mapping[str, Mapping[int, RepresentationStats]]

    @property
    def diagnostics_measured(self) -> bool:
        return bool(self.diagnostics_by_stream)

    @property
    def diagnostics_by_modality(self) -> Mapping[int, RepresentationStats]:
        """Compatibility alias for the collapse-monitored fixed-probe stream."""

        return self.diagnostics_by_stream.get(TEACHER_TARGET_DIAGNOSTIC_STREAM, {})

    @property
    def teacher_target_diagnostics_by_modality(self) -> Mapping[int, RepresentationStats]:
        return self.diagnostics_by_stream.get(TEACHER_TARGET_DIAGNOSTIC_STREAM, {})

    @property
    def training_teacher_target_diagnostics_by_modality(
        self,
    ) -> Mapping[int, RepresentationStats]:
        return self.diagnostics_by_stream.get(TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM, {})

    @property
    def prediction_diagnostics_by_modality(self) -> Mapping[int, RepresentationStats]:
        return self.diagnostics_by_stream.get(PREDICTION_DIAGNOSTIC_STREAM, {})


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one invocation, which may be only part of a longer run."""

    start_step: int
    end_step: int
    total_steps: int
    ema_update_count: int
    latest_checkpoint: Path | None
    last_metrics: StepMetrics | None
    runner_contract_sha256: str


BatchSource = Iterable[MatchingBatch] | Callable[[], MatchingBatch] | Callable[[int], MatchingBatch]
StepCallback = Callable[[StepMetrics], None]
StopPredicate = Callable[[], bool]


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _canonical_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Require portable metadata and detach it from caller-owned containers."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive: the input is already a Mapping
        raise TypeError(f"{name} must encode as a JSON object")
    return decoded


def _canonical_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical_json_mapping(provenance, name="provenance")


def _diagnostic_schema() -> dict[str, Any]:
    """Describe exactly which forward snapshot each named stream measures."""

    return {
        "schema_version": 2,
        "statistics": {
            "variance": "mean_population_variance_across_embedding_dimensions",
            "effective_rank": "exp_entropy_of_centered_singular_value_energy",
            "off_diagonal_cosine": "mean_ordered_pairwise_cosine_similarity",
            "grouping": "separate_by_modality_id",
        },
        "streams": {
            TEACHER_TARGET_DIAGNOSTIC_STREAM: {
                "tensor": "target_teacher(fixed_probe.target_patches)",
                "snapshot": "after_optimizer_step_and_teacher_ema_update",
                "collapse_monitored": True,
            },
            TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM: {
                "tensor": "MatchingStepOutput.targets",
                "snapshot": "training_forward_before_backward_optimizer_and_teacher_ema_update",
                "collapse_monitored": False,
            },
            PREDICTION_DIAGNOSTIC_STREAM: {
                "tensor": "MatchingStepOutput.predictions",
                "snapshot": "training_forward_before_backward_optimizer_and_teacher_ema_update",
                "collapse_monitored": False,
            },
        },
        "collapse_stream": TEACHER_TARGET_DIAGNOSTIC_STREAM,
        "collapse_check_timing": ("fixed_probe_after_completed_step_when_step_gt_warmup"),
    }


def _runner_contract(
    *,
    gradient_clip_norm: float | None,
    probe: FixedTargetPatchProbe,
    references: Mapping[int, RepresentationStats],
    thresholds: CollapseThresholds,
    warmup_steps: int,
    runtime_policy: TrainingRuntimePolicy,
) -> tuple[dict[str, Any], str]:
    contract = _canonical_json_mapping(
        {
            "schema_version": _RUNNER_CONTRACT_SCHEMA_VERSION,
            "gradient_clipping": {
                "maximum_l2_norm": gradient_clip_norm,
                "timing": "after_finite_gradient_check_before_optimizer_step",
            },
            "fixed_target_patch_probe": {
                "schema_version": 1,
                "sha256": probe.sha256,
                "patch_shape": list(probe.target_patches.shape),
                "sample_count_by_modality": {
                    str(modality_id): count
                    for modality_id, count in probe.sample_count_by_modality.items()
                },
                "model_snapshot": "after_optimizer_step_and_teacher_ema_update",
                "rng_policy": "capture_and_restore_all_runner_rng_streams",
            },
            "collapse_reference_by_modality": {
                str(modality_id): reference.to_dict()
                for modality_id, reference in sorted(references.items())
            },
            "collapse_thresholds": thresholds.to_dict(),
            "collapse_warmup_steps": warmup_steps,
            "diagnostics": _diagnostic_schema(),
            "diagnostic_cadence": {
                "first_completed_step": True,
                "every_completed_steps": _DIAGNOSTICS_EVERY_STEPS,
                "checkpoint_steps": True,
                "invocation_final_step": True,
                "stop_requested_step": True,
                "training_batch_streams_share_cadence": True,
            },
            "step_callback_rng_policy": (
                "capture_and_restore_python_numpy_torch_cpu_and_all_cuda_generators"
            ),
            "training_runtime": runtime_policy.to_dict(),
        },
        name="runner contract",
    )
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return contract, hashlib.sha256(encoded).hexdigest()


def _runner_contract_sha256(contract: Mapping[str, Any]) -> str:
    canonical = _canonical_json_mapping(contract, name="checkpoint runner contract")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ema_update_count(system: CrossModalMatchingSystem) -> int:
    try:
        value = system.target_teacher.num_updates
    except AttributeError as error:
        raise TrainingRunnerError("system target teacher has no EMA update counter") from error
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise TrainingRunnerError("EMA update counter must be scalar")
        value = int(value.detach().cpu().item())
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingRunnerError("EMA update counter must be a non-negative integer")
    return value


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    # Avoid initializing CUDA just to create a CPU checkpoint.  A real CUDA run
    # has necessarily initialized it before reaching its first optimizer step.
    if torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, Mapping) or set(state) != required:
        raise TrainingRunnerError("checkpoint RNG state has an unsupported schema")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, Tensor):
        raise TrainingRunnerError("checkpoint Torch CPU RNG state is not a tensor")
    torch.set_rng_state(cpu_state.cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise TrainingRunnerError("CUDA RNG state cannot be restored without CUDA")
        if not isinstance(cuda_state, (list, tuple)):
            raise TrainingRunnerError("checkpoint CUDA RNG state must be a sequence")
        cpu_cuda_state: list[Tensor] = []
        for device_state in cuda_state:
            if not isinstance(device_state, Tensor):
                raise TrainingRunnerError("checkpoint CUDA RNG states must be tensors")
            if device_state.dtype != torch.uint8 or device_state.ndim != 1:
                raise TrainingRunnerError(
                    "checkpoint CUDA RNG states must be one-dimensional ByteTensors"
                )
            # ``torch.load(..., map_location=device)`` relocates every tensor,
            # including these RNG blobs, to CUDA.  The CUDA RNG API requires
            # CPU ByteTensors even when restoring a CUDA generator.
            cpu_cuda_state.append(device_state.detach().cpu().contiguous())
        torch.cuda.set_rng_state_all(cpu_cuda_state)


@contextmanager
def preserve_runner_rng_state():
    """Prevent observational integrations from advancing any training RNG stream."""

    state = _capture_rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state)


def _state_api(source: object) -> tuple[Callable[[], object], Callable[[object], None]] | None:
    state_dict = getattr(source, "state_dict", None)
    load_state_dict = getattr(source, "load_state_dict", None)
    if state_dict is None and load_state_dict is None:
        return None
    if not callable(state_dict) or not callable(load_state_dict):
        raise TypeError("a stateful batch source must define both state_dict and load_state_dict")
    return state_dict, load_state_dict


class _BatchCursor:
    """Unify indexed factories, streams, and finite materialized sequences."""

    def __init__(self, source: BatchSource, *, start_step: int, restored_state: bool) -> None:
        self.source = source
        self.sequence: Sequence[MatchingBatch] | None = None
        self.iterator: Any | None = None
        self.call_with_step: bool | None = None

        if callable(source):
            self.call_with_step = self._callable_accepts_step(source)
            if start_step and not self.call_with_step and not restored_state:
                raise TrainingRunnerError(
                    "resuming a zero-argument batch factory requires checkpointed "
                    "state_dict/load_state_dict state; use an absolute-step factory otherwise"
                )
        elif isinstance(source, Sequence):
            self.sequence = source
        else:
            if start_step and not restored_state:
                raise TrainingRunnerError(
                    "resuming a streaming iterable requires state_dict/load_state_dict; "
                    "use a sequence or an absolute-step batch factory otherwise"
                )
            self.iterator = iter(source)

    @staticmethod
    def _callable_accepts_step(source: Callable[..., MatchingBatch]) -> bool:
        try:
            signature = inspect.signature(source)
        except (TypeError, ValueError) as error:
            raise TypeError("batch factory must expose an inspectable signature") from error
        try:
            signature.bind(0)
        except TypeError:
            try:
                signature.bind()
            except TypeError as error:
                raise TypeError(
                    "batch factory must accept either zero arguments or one step"
                ) from error
            return False
        return True

    def next(self, absolute_step_index: int) -> MatchingBatch:
        try:
            if self.call_with_step is not None:
                if self.call_with_step:
                    batch = self.source(absolute_step_index)  # type: ignore[call-arg]
                else:
                    batch = self.source()  # type: ignore[operator]
            elif self.sequence is not None:
                batch = self.sequence[absolute_step_index]
            else:
                batch = next(self.iterator)
        except (IndexError, StopIteration) as error:
            raise TrainingRunnerError(
                f"batch source exhausted before absolute step {absolute_step_index + 1}"
            ) from error
        if not isinstance(batch, MatchingBatch):
            raise TypeError("batch source must yield MatchingBatch instances")
        return batch


@dataclass(frozen=True)
class _ResumeState:
    step: int
    rng: Mapping[str, Any]
    batch_source_state: object | None


def _load_resume_checkpoint(
    path: str | Path,
    *,
    system: CrossModalMatchingSystem,
    optimizer: torch.optim.Optimizer,
    provenance: Mapping[str, Any],
    runner_contract: Mapping[str, Any],
    runner_contract_sha256: str,
    map_location: torch.device,
) -> _ResumeState:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except FileNotFoundError as error:
        raise TrainingRunnerError(f"resume checkpoint does not exist: {checkpoint_path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise TrainingRunnerError("checkpoint container has an unsupported schema")
    if payload.get("metadata") != provenance:
        raise TrainingRunnerError("resume provenance does not exactly match the checkpoint")
    state = payload.get("state")
    if (
        not isinstance(state, Mapping)
        or state.get("runner_schema_version") != _RUNNER_SCHEMA_VERSION
    ):
        raise TrainingRunnerError("checkpoint training state has an unsupported schema")
    required = {
        "runner_schema_version",
        "model",
        "optimizer",
        "step",
        "ema_update_count",
        "rng",
        "batch_source_state",
        "provenance",
        "runner_contract",
        "runner_contract_sha256",
    }
    if set(state) != required:
        raise TrainingRunnerError("checkpoint training state has missing or unknown fields")
    if state["provenance"] != provenance:
        raise TrainingRunnerError("checkpoint has inconsistent provenance records")
    checkpoint_contract = state["runner_contract"]
    checkpoint_contract_sha256 = state["runner_contract_sha256"]
    if not isinstance(checkpoint_contract, Mapping):
        raise TrainingRunnerError("checkpoint runner contract must be a mapping")
    if not isinstance(checkpoint_contract_sha256, str):
        raise TrainingRunnerError("checkpoint runner contract hash must be a string")
    canonical_checkpoint_contract = _canonical_json_mapping(
        checkpoint_contract,
        name="checkpoint runner contract",
    )
    if _runner_contract_sha256(canonical_checkpoint_contract) != checkpoint_contract_sha256:
        raise TrainingRunnerError("checkpoint runner contract hash is internally inconsistent")
    if (
        checkpoint_contract_sha256 != runner_contract_sha256
        or canonical_checkpoint_contract != runner_contract
    ):
        raise TrainingRunnerError("resume runner contract does not exactly match the checkpoint")
    step = _non_negative_integer(state["step"], "checkpoint step")
    if payload.get("step") != step:
        raise TrainingRunnerError("checkpoint container and training step disagree")

    system.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    expected_ema_updates = _non_negative_integer(
        state["ema_update_count"], "checkpoint EMA update count"
    )
    if _ema_update_count(system) != expected_ema_updates:
        raise TrainingRunnerError("checkpoint EMA counter disagrees with model state")
    rng = state["rng"]
    if not isinstance(rng, Mapping):
        raise TrainingRunnerError("checkpoint RNG state must be a mapping")
    return _ResumeState(
        step=step,
        rng=rng,
        batch_source_state=state["batch_source_state"],
    )


def _checkpoint_state(
    *,
    system: CrossModalMatchingSystem,
    optimizer: torch.optim.Optimizer,
    step: int,
    provenance: Mapping[str, Any],
    runner_contract: Mapping[str, Any],
    runner_contract_sha256: str,
    batch_source_state: object | None,
) -> dict[str, Any]:
    return {
        "runner_schema_version": _RUNNER_SCHEMA_VERSION,
        "model": system.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "ema_update_count": _ema_update_count(system),
        "rng": _capture_rng_state(),
        "batch_source_state": batch_source_state,
        "provenance": dict(provenance),
        "runner_contract": dict(runner_contract),
        "runner_contract_sha256": runner_contract_sha256,
    }


def _force_save_checkpoint(
    checkpoint_manager: CheckpointManager,
    *,
    step: int,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically persist an off-cadence failure using the normal checkpoint schema."""

    checkpoint_manager.root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_manager.root / f"step-{step:09d}.pt"
    temporary = checkpoint_manager.root / f".{destination.name}.failure-tmp-{os.getpid()}"
    payload = {
        "schema_version": 1,
        "step": step,
        "metadata": dict(metadata),
        "state": dict(state),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        fsync_file_and_parent(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _model_device(system: CrossModalMatchingSystem) -> torch.device:
    parameter = next(system.parameters(), None)
    if parameter is None:
        raise TrainingRunnerError("matching system has no parameters")
    devices = {value.device for value in system.parameters()}
    devices.update(value.device for value in system.buffers())
    if len(devices) != 1:
        raise TrainingRunnerError("matching system parameters and buffers must share one device")
    return parameter.device


def _validate_references(
    references: Mapping[int, RepresentationStats],
    probe: FixedTargetPatchProbe,
) -> dict[int, RepresentationStats]:
    if not isinstance(references, Mapping) or not references:
        raise ValueError("collapse_reference must be a non-empty modality mapping")
    result: dict[int, RepresentationStats] = {}
    for modality_id, reference in references.items():
        if isinstance(modality_id, bool) or not isinstance(modality_id, int) or modality_id < 0:
            raise ValueError("collapse reference modality IDs must be non-negative integers")
        if not isinstance(reference, RepresentationStats):
            raise TypeError("collapse references must contain RepresentationStats")
        if reference.variance <= 0:
            raise ValueError("collapse reference variance must be positive")
        result[modality_id] = reference
    probe_counts = probe.sample_count_by_modality
    if set(result) != set(probe_counts):
        raise ValueError("collapse references must exactly match fixed-probe modalities")
    for modality_id, reference in result.items():
        if reference.count != probe_counts[modality_id]:
            raise ValueError(
                "collapse reference counts must exactly match fixed-probe sample counts"
            )
    return result


def _gradient_clip_norm(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("gradient_clip_norm must be a real number when supplied")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive when supplied")
    return result


def _check_and_clip_gradients(
    system: CrossModalMatchingSystem,
    maximum_norm: float | None,
) -> None:
    parameters = [parameter for parameter in system.parameters() if parameter.grad is not None]
    if not parameters:
        raise TrainingRunnerError("training loss produced no gradients")
    clip_norm = float("inf") if maximum_norm is None else maximum_norm
    try:
        torch.nn.utils.clip_grad_norm_(
            parameters,
            clip_norm,
            error_if_nonfinite=True,
            foreach=False,
        )
    except RuntimeError as error:
        raise TrainingRunnerError("training produced a non-finite gradient norm") from error


def _fixed_probe_diagnostics(
    system: CrossModalMatchingSystem,
    *,
    target_patches: Tensor,
    target_modality_ids: Tensor,
    runtime_policy: TrainingRuntimePolicy,
) -> dict[int, RepresentationStats]:
    """Measure a fixed teacher probe without advancing any RNG stream."""

    ema_before = _ema_update_count(system)
    rng_state = _capture_rng_state()
    try:
        with torch.inference_mode(), runtime_policy.autocast(target_patches.device):
            targets = system.target_teacher(target_patches)
            diagnostics = stats_by_modality(targets, target_modality_ids)
    finally:
        _restore_rng_state(rng_state)
    if _ema_update_count(system) != ema_before:
        raise TrainingRunnerError("fixed-probe evaluation must not update the EMA teacher")
    return diagnostics


def run_matching_training(
    system: CrossModalMatchingSystem,
    optimizer: torch.optim.Optimizer,
    batches: BatchSource,
    checkpoint_manager: CheckpointManager,
    provenance: Mapping[str, Any],
    *,
    total_steps: int,
    max_steps: int | None = None,
    resume_from: str | Path | None = None,
    collapse_probe: FixedTargetPatchProbe,
    collapse_reference: Mapping[int, RepresentationStats],
    collapse_thresholds: CollapseThresholds,
    collapse_warmup_steps: int,
    gradient_clip_norm: float | None = None,
    runtime_policy: TrainingRuntimePolicy | None = None,
    on_step: StepCallback | None = None,
    should_stop: StopPredicate | None = None,
) -> TrainingResult:
    """Train until an absolute target step, optionally bounded per invocation.

    ``max_steps`` limits work in this invocation rather than redefining the
    global schedule.  Indexed batch factories receive a zero-based absolute
    step index, so a resumed job asks for exactly the next planned batch.
    ``should_stop`` is checked only after a complete optimizer and EMA update;
    a true value forces an atomic checkpoint at that step before returning.

    Collapse references and thresholds are mandatory and should be locked from
    the exact fixed probe before the SSL run.  The stochastic training-batch
    teacher and prediction streams are logging-only.  Abort decisions use the
    same fixed patch tensors at the registered sparse diagnostic cadence after
    teacher EMA updates and begin only after ``collapse_warmup_steps`` completed steps.
    """

    total_steps = _non_negative_integer(total_steps, "total_steps")
    if max_steps is not None:
        max_steps = _non_negative_integer(max_steps, "max_steps")
    collapse_warmup_steps = _non_negative_integer(collapse_warmup_steps, "collapse_warmup_steps")
    if not isinstance(collapse_thresholds, CollapseThresholds):
        raise TypeError("collapse_thresholds must be CollapseThresholds")
    if not isinstance(collapse_probe, FixedTargetPatchProbe):
        raise TypeError("collapse_probe must be a FixedTargetPatchProbe")
    # Take a defensive copy so external tensor mutation cannot change the
    # active probe after its content digest enters the runner contract.
    probe = FixedTargetPatchProbe(
        collapse_probe.target_patches,
        collapse_probe.target_modality_ids,
    )
    references = _validate_references(collapse_reference, probe)
    gradient_clip_norm = _gradient_clip_norm(gradient_clip_norm)
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable")
    if should_stop is not None and not callable(should_stop):
        raise TypeError("should_stop must be callable")
    if not isinstance(checkpoint_manager, CheckpointManager):
        raise TypeError("checkpoint_manager must be CheckpointManager")
    metadata = _canonical_provenance(provenance)
    device = _model_device(system)
    selected_runtime = runtime_policy or TrainingRuntimePolicy.eager_for_device(device)
    if not isinstance(selected_runtime, TrainingRuntimePolicy):
        raise TypeError("runtime_policy must be a TrainingRuntimePolicy")
    selected_runtime.require_device(device)
    runner_contract, runner_contract_sha256 = _runner_contract(
        gradient_clip_norm=gradient_clip_norm,
        probe=probe,
        references=references,
        thresholds=collapse_thresholds,
        warmup_steps=collapse_warmup_steps,
        runtime_policy=selected_runtime,
    )
    probe_target_patches = probe.target_patches.to(device)
    probe_target_modality_ids = probe.target_modality_ids.to(device)
    source_state_api = _state_api(batches)

    latest_checkpoint = Path(resume_from) if resume_from is not None else None
    restored_source_state = False
    if resume_from is None:
        start_step = 0
        restored_rng: Mapping[str, Any] | None = None
    else:
        resumed = _load_resume_checkpoint(
            resume_from,
            system=system,
            optimizer=optimizer,
            provenance=metadata,
            runner_contract=runner_contract,
            runner_contract_sha256=runner_contract_sha256,
            map_location=device,
        )
        start_step = resumed.step
        restored_rng = resumed.rng
        if resumed.batch_source_state is not None:
            if source_state_api is None:
                raise TrainingRunnerError(
                    "checkpoint contains batch-source state but the supplied source cannot load it"
                )
            source_state_api[1](resumed.batch_source_state)
            restored_source_state = True

    if start_step > total_steps:
        raise TrainingRunnerError(
            f"checkpoint step {start_step} is beyond requested total_steps {total_steps}"
        )
    cursor = _BatchCursor(
        batches,
        start_step=start_step,
        restored_state=restored_source_state,
    )
    # Loading source state or constructing its iterator is allowed to execute
    # arbitrary user code.  Restore the recorded RNG only after those actions.
    if resume_from is not None:
        assert restored_rng is not None
        _restore_rng_state(restored_rng)
        if checkpoint_manager.policy.is_artifact_step(start_step):
            # A node or transport failure can occur after the checkpoint was
            # durably published but before its W&B upload was acknowledged and
            # locally receipted.  Repair that boundary before requesting the next
            # batch or advancing the optimizer.  Tracking remains observational.
            with preserve_runner_rng_state():
                checkpoint_manager.ensure_artifact_logged(
                    resume_from,
                    step=start_step,
                    metadata=metadata,
                )

    invocation_stop = total_steps
    if max_steps is not None:
        invocation_stop = min(invocation_stop, start_step + max_steps)

    system.train()
    last_metrics: StepMetrics | None = None
    completed_step = start_step
    for absolute_index in range(start_step, invocation_stop):
        batch = cursor.next(absolute_index).to(device)
        ema_before = _ema_update_count(system)
        optimizer.zero_grad(set_to_none=True)
        with selected_runtime.autocast(device):
            output = system(batch)
        if output.loss.numel() != 1 or not bool(torch.isfinite(output.loss)):
            raise TrainingRunnerError("training loss must be one finite scalar")
        output.loss.backward()
        _check_and_clip_gradients(system, gradient_clip_norm)
        if _ema_update_count(system) != ema_before:
            raise TrainingRunnerError("forward/backward must not update the EMA teacher")

        optimizer.step()
        if _ema_update_count(system) != ema_before:
            raise TrainingRunnerError("optimizer step must not update the EMA counter")
        system.update_teacher()
        ema_after = _ema_update_count(system)
        if ema_after != ema_before + 1:
            raise TrainingRunnerError(
                "each optimizer step must be followed by exactly one EMA update"
            )
        completed_step = absolute_index + 1
        stop_requested = should_stop is not None and bool(should_stop())
        diagnostics_due = (
            completed_step == 1
            or completed_step % _DIAGNOSTICS_EVERY_STEPS == 0
            or checkpoint_manager.policy.is_checkpoint_step(completed_step)
            or completed_step == invocation_stop
            or stop_requested
        )
        diagnostics_by_stream: dict[str, Mapping[int, RepresentationStats]] = {}
        diagnostics: Mapping[int, RepresentationStats] = {}
        if diagnostics_due:
            # These stochastic-batch streams are observational only.  Comparing
            # them with a different calibration batch would conflate tissue mix
            # with model collapse.
            diagnostics_by_stream = {
                TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM: stats_by_modality(
                    output.targets,
                    batch.target_modality_ids,
                ),
                PREDICTION_DIAGNOSTIC_STREAM: stats_by_modality(
                    output.predictions,
                    batch.query_modality_ids,
                ),
            }
            diagnostics = _fixed_probe_diagnostics(
                system,
                target_patches=probe_target_patches,
                target_modality_ids=probe_target_modality_ids,
                runtime_policy=selected_runtime,
            )
            diagnostics_by_stream[TEACHER_TARGET_DIAGNOSTIC_STREAM] = diagnostics
            reference_modalities = set(references)
            if set(diagnostics) != reference_modalities:
                raise TrainingRunnerError(
                    "observed fixed-probe modalities do not exactly match collapse references"
                )
            expected_training_modalities = {
                int(value)
                for value in batch.target_modality_ids.detach()
                .reshape(-1)
                .unique()
                .to(device="cpu")
                .tolist()
            }
            expected_prediction_modalities = {
                int(value)
                for value in batch.query_modality_ids.detach()
                .reshape(-1)
                .unique()
                .to(device="cpu")
                .tolist()
            }
            for stream_name, expected_modalities in (
                (
                    TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
                    expected_training_modalities,
                ),
                (PREDICTION_DIAGNOSTIC_STREAM, expected_prediction_modalities),
            ):
                if not expected_modalities or not expected_modalities <= reference_modalities:
                    raise TrainingRunnerError(
                        f"observed {stream_name} batch modalities are absent from "
                        "collapse references"
                    )
                if set(diagnostics_by_stream[stream_name]) != expected_modalities:
                    raise TrainingRunnerError(
                        f"observed {stream_name} modalities do not exactly match its batch"
                    )
        last_metrics = StepMetrics(
            step=completed_step,
            loss=float(output.loss.detach()),
            accuracy=float(output.matching.accuracy.detach()),
            chance=float(output.matching.chance.detach()),
            ema_update_count=ema_after,
            diagnostics_by_stream={
                stream_name: dict(stream_diagnostics)
                for stream_name, stream_diagnostics in diagnostics_by_stream.items()
            },
        )

        collapsed: dict[int, tuple[str, ...]] = {}
        if diagnostics_due and completed_step > collapse_warmup_steps:
            collapsed = {
                modality_id: reasons
                for modality_id, stats in diagnostics.items()
                if (
                    reasons := collapse_reasons(
                        stats,
                        references[modality_id],
                        collapse_thresholds,
                    )
                )
            }
        if on_step is not None:
            with preserve_runner_rng_state():
                on_step(last_metrics)
        if _ema_update_count(system) != ema_after:
            raise TrainingRunnerError("step callback must not update the EMA teacher")
        saved: Path | None = None
        state_required = (
            checkpoint_manager.policy.is_checkpoint_step(completed_step)
            or bool(collapsed)
            or stop_requested
        )
        if state_required:
            batch_source_state = source_state_api[0]() if source_state_api is not None else None
            state = _checkpoint_state(
                system=system,
                optimizer=optimizer,
                step=completed_step,
                provenance=metadata,
                runner_contract=runner_contract,
                runner_contract_sha256=runner_contract_sha256,
                batch_source_state=batch_source_state,
            )
            # Checkpoint serialization and artifact bookkeeping must not perturb the
            # RNG stream that determines subsequent model and data randomness.
            checkpoint_rng = state["rng"]
            try:
                saved = checkpoint_manager.maybe_save(
                    step=completed_step,
                    state=state,
                    metadata=metadata,
                )
                if saved is None and collapsed:
                    saved = _force_save_checkpoint(
                        checkpoint_manager,
                        step=completed_step,
                        state=state,
                        metadata=metadata,
                    )
                if saved is None and stop_requested:
                    saved = _force_save_checkpoint(
                        checkpoint_manager,
                        step=completed_step,
                        state=state,
                        metadata=metadata,
                    )
            finally:
                _restore_rng_state(checkpoint_rng)
        if saved is not None:
            latest_checkpoint = saved
        if collapsed:
            raise RepresentationCollapseError(
                step=completed_step,
                reasons_by_modality=collapsed,
                diagnostics_by_modality=diagnostics,
                checkpoint_path=latest_checkpoint,
            )
        if stop_requested:
            break

    return TrainingResult(
        start_step=start_step,
        end_step=completed_step,
        total_steps=total_steps,
        ema_update_count=_ema_update_count(system),
        latest_checkpoint=latest_checkpoint,
        last_metrics=last_metrics,
        runner_contract_sha256=runner_contract_sha256,
    )
