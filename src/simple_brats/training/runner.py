"""Deterministic, resumable training loop for cross-modal matching.

The runner deliberately owns no experiment tracker.  Checkpoint and artifact
cadence are delegated to :class:`CheckpointManager`, which means cluster runs
can use an offline artifact sink without introducing network access here.
"""

from __future__ import annotations

import inspect
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .checkpoints import CheckpointManager
from .diagnostics import (
    CollapseThresholds,
    RepresentationStats,
    collapse_reasons,
    stats_by_modality,
)
from .matching import CrossModalMatchingSystem, MatchingBatch

_RUNNER_SCHEMA_VERSION = 1


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
    ) -> None:
        self.step = step
        self.reasons_by_modality = dict(reasons_by_modality)
        self.diagnostics_by_modality = dict(diagnostics_by_modality)
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
    diagnostics_by_modality: Mapping[int, RepresentationStats]


@dataclass(frozen=True)
class TrainingResult:
    """Summary of one invocation, which may be only part of a longer run."""

    start_step: int
    end_step: int
    total_steps: int
    ema_update_count: int
    latest_checkpoint: Path | None
    last_metrics: StepMetrics | None


BatchSource = Iterable[MatchingBatch] | Callable[[], MatchingBatch] | Callable[[int], MatchingBatch]
StepCallback = Callable[[StepMetrics], None]


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _canonical_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Require portable metadata and detach it from caller-owned containers."""

    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    try:
        encoded = json.dumps(
            dict(provenance),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("provenance must contain finite JSON-compatible values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # defensive: the input is already a Mapping
        raise TypeError("provenance must encode as a JSON object")
    return decoded


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
        torch.cuda.set_rng_state_all(cuda_state)


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
    }
    if set(state) != required:
        raise TrainingRunnerError("checkpoint training state has missing or unknown fields")
    if state["provenance"] != provenance:
        raise TrainingRunnerError("checkpoint has inconsistent provenance records")
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
    }


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
) -> dict[int, RepresentationStats]:
    if not isinstance(references, Mapping) or not references:
        raise ValueError("collapse_reference must be a non-empty modality mapping")
    result: dict[int, RepresentationStats] = {}
    for modality_id, reference in references.items():
        if isinstance(modality_id, bool) or not isinstance(modality_id, int) or modality_id < 0:
            raise ValueError("collapse reference modality IDs must be non-negative integers")
        if not isinstance(reference, RepresentationStats):
            raise TypeError("collapse references must contain RepresentationStats")
        result[modality_id] = reference
    return result


def _check_finite_gradients(system: CrossModalMatchingSystem) -> None:
    saw_gradient = False
    for parameter in system.parameters():
        if parameter.grad is None:
            continue
        saw_gradient = True
        if not bool(torch.isfinite(parameter.grad).all()):
            raise TrainingRunnerError("training produced a non-finite gradient")
    if not saw_gradient:
        raise TrainingRunnerError("training loss produced no gradients")


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
    collapse_reference: Mapping[int, RepresentationStats],
    collapse_thresholds: CollapseThresholds,
    collapse_warmup_steps: int,
    gradient_clip_norm: float | None = None,
    on_step: StepCallback | None = None,
) -> TrainingResult:
    """Train until an absolute target step, optionally bounded per invocation.

    ``max_steps`` limits work in this invocation rather than redefining the
    global schedule.  Indexed batch factories receive a zero-based absolute
    step index, so a resumed job asks for exactly the next planned batch.

    Collapse references and thresholds are mandatory and should be locked from
    a baseline before the SSL run.  Diagnostics are computed every step, while
    abort decisions begin only after ``collapse_warmup_steps`` completed steps.
    """

    total_steps = _non_negative_integer(total_steps, "total_steps")
    if max_steps is not None:
        max_steps = _non_negative_integer(max_steps, "max_steps")
    collapse_warmup_steps = _non_negative_integer(collapse_warmup_steps, "collapse_warmup_steps")
    if not isinstance(collapse_thresholds, CollapseThresholds):
        raise TypeError("collapse_thresholds must be CollapseThresholds")
    references = _validate_references(collapse_reference)
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when supplied")
    if on_step is not None and not callable(on_step):
        raise TypeError("on_step must be callable")
    if not isinstance(checkpoint_manager, CheckpointManager):
        raise TypeError("checkpoint_manager must be CheckpointManager")
    metadata = _canonical_provenance(provenance)
    device = _model_device(system)
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
        output = system(batch)
        if output.loss.numel() != 1 or not bool(torch.isfinite(output.loss)):
            raise TrainingRunnerError("training loss must be one finite scalar")
        output.loss.backward()
        _check_finite_gradients(system)
        if _ema_update_count(system) != ema_before:
            raise TrainingRunnerError("forward/backward must not update the EMA teacher")
        if gradient_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), gradient_clip_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise TrainingRunnerError("gradient norm is not finite")

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

        diagnostics = stats_by_modality(output.targets, batch.target_modality_ids)
        if set(diagnostics) != set(references):
            raise TrainingRunnerError(
                "observed diagnostic modalities do not exactly match collapse references"
            )
        last_metrics = StepMetrics(
            step=completed_step,
            loss=float(output.loss.detach()),
            accuracy=float(output.matching.accuracy.detach()),
            chance=float(output.matching.chance.detach()),
            ema_update_count=ema_after,
            diagnostics_by_modality=dict(diagnostics),
        )

        collapsed: dict[int, tuple[str, ...]] = {}
        if completed_step > collapse_warmup_steps:
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
            on_step(last_metrics)
        if _ema_update_count(system) != ema_after:
            raise TrainingRunnerError("step callback must not update the EMA teacher")
        batch_source_state = source_state_api[0]() if source_state_api is not None else None
        state = _checkpoint_state(
            system=system,
            optimizer=optimizer,
            step=completed_step,
            provenance=metadata,
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
        finally:
            _restore_rng_state(checkpoint_rng)
        if saved is not None:
            latest_checkpoint = saved
        if collapsed:
            raise RepresentationCollapseError(
                step=completed_step,
                reasons_by_modality=collapsed,
                diagnostics_by_modality=diagnostics,
            )

    return TrainingResult(
        start_step=start_step,
        end_step=completed_step,
        total_steps=total_steps,
        ema_update_count=_ema_update_count(system),
        latest_checkpoint=latest_checkpoint,
        last_metrics=last_metrics,
    )
