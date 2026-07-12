"""Subject-balanced, walltime-resumable real-data matching pretraining.

The validated short run established the scientific configuration.  This
module changes only the operational schedule: every training subject receives
one consecutive block of bags per subject epoch, longitudinal visits rotate
across subject epochs, and independent Slurm invocations resume one immutable
absolute-step stream from exact checkpoints.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import signal
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simple_brats.config import (
    ExperimentConfig,
    ModelConfig,
    PatchConfig,
    TaskConfig,
    load_experiment_config,
)
from simple_brats.data.case_grids import load_case_grid_manifest
from simple_brats.data.manifest import (
    CaseRecord,
    canonical_json_bytes,
    load_manifest,
    sha256_file,
)
from simple_brats.data.scheduled_cache import OptimizedRuntimeConfig
from simple_brats.data.splits import cases_for_splits, load_split, validate_split
from simple_brats.provenance import verify_git_sha
from simple_brats.short_run import (
    DeterministicRealBatchFactory,
    ShortRunError,
    StepAssignment,
    _build_fixed_target_probe,
    _capture_torch_rng,
    _MetricsLogger,
    _require_canonical,
    _resolve_file,
    _restore_torch_rng,
    _stats_record,
    _wandb_for_schedule,
    _write_new_canonical,
)
from simple_brats.training import (
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    CheckpointManager,
    CheckpointPolicy,
    CollapseThresholds,
    TrainingRuntimeError,
    TrainingRuntimePolicy,
    WandbArtifactSink,
    apply_model_runtime,
    build_adamw_optimizer,
    build_matching_system,
    configure_training_runtime,
    run_matching_training,
    stats_by_modality,
)

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_NAME = re.compile(r"^step-([0-9]{9})\.pt$")
_TRAINING_PLAN_NAME = re.compile(r"^step-([0-9]{9})\.(plan\.json|prepared\.json)$")
_ZERO_RESTART_METRICS_NAME = re.compile(r"^start-000000000-stop-000005000-[A-Za-z0-9._-]+\.jsonl$")
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")

DEFAULT_TOTAL_STEPS = 50_000
DEFAULT_MAX_STEPS_PER_INVOCATION = 5_000
DEFAULT_BAGS_PER_SUBJECT = 8
DEFAULT_EXPECTED_TRAIN_CASES = 1_044
DEFAULT_EXPECTED_TRAIN_SUBJECTS = 643
WALLTIME_REQUEUE_EXIT_CODE = 75
_CUDA_WORKSPACE_CONFIG = ":4096:8"

_LEARNING_RATE = 1e-4
_WEIGHT_DECAY = 0.05
_GRADIENT_CLIP_NORM = 10.0
_COLLAPSE_WARMUP_STEPS = 25
_CANDIDATE_POOL_SIZE = 512
_MAX_PLAN_ATTEMPTS = 8
_FIXED_PROBE_SUBJECTS = 4
_COLLAPSE_THRESHOLDS = CollapseThresholds(
    minimum_variance_ratio=0.10,
    minimum_effective_rank_ratio=0.25,
    maximum_off_diagonal_cosine=0.95,
)
_VALIDATED_CONFIG = ExperimentConfig(
    seed=0,
    checkpoint_every_steps=1_000,
    artifact_every_steps=5_000,
    patch=PatchConfig(
        footprint_mm=4.0,
        thin_mm=4.0,
        tensor_shape=(16, 16, 16),
    ),
    model=ModelConfig(
        width=256,
        depth=8,
        heads=4,
        mlp_ratio=4.0,
        predictor_depth=1,
        teacher_ema_momentum=0.996,
    ),
    task=TaskConfig(
        modalities=("t1n", "t1c", "t2w", "t2f"),
        positions_per_bag=32,
        objective="match",
        allow_target_modality_elsewhere=True,
        allow_target_modality_at_target=False,
        pass_scan_statistics_to_teacher=False,
    ),
)


class LongRunError(RuntimeError):
    """The long run cannot satisfy its immutable experiment contract."""


def configure_exact_resume_runtime(device: torch.device) -> dict[str, object]:
    """Enable and report the fail-closed runtime policy used for exact resume."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        _CUDA_WORKSPACE_CONFIG
    ):
        raise LongRunError(
            "CUDA exact resume requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.are_deterministic_algorithms_enabled():
        raise LongRunError("Torch deterministic algorithms could not be enabled")
    if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise LongRunError("cuDNN deterministic runtime settings were not retained")
    if (
        torch.get_float32_matmul_precision() != "highest"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise LongRunError("exact float32 matmul settings were not retained")
    return {
        "schema": "simple-brats.exact-resume-runtime",
        "schema_version": 1,
        "device_type": device.type,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": (
            _CUDA_WORKSPACE_CONFIG if device.type == "cuda" else "not_applicable"
        ),
        "calibration_replay": "recompute_and_require_byte_exact_canonical_statistics",
    }


def _case_identity(case: CaseRecord) -> tuple[str, str, str, str, str]:
    return (case.source, case.release, case.subject_id, case.visit_id, case.case_id)


def _seeded_digest(seed: int, *parts: object) -> str:
    payload = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SubjectStep:
    """Human-readable absolute assignment emitted by the subject schedule."""

    subject_id: str
    case_id: str
    visit_id: str
    subject_epoch: int
    subject_position: int
    visit_rotation_index: int
    bag_index: int
    case_index: int


class SubjectBalancedSchedule:
    """Stateless subject-balanced schedule with rotating longitudinal visits.

    A subject epoch deterministically shuffles all subjects.  Each subject then
    contributes one visit and ``bags_per_subject`` consecutive optimizer bags.
    The visit rotates on every subject epoch, so raw case multiplicity does not
    upweight longitudinal subjects while every visit is eventually exposed.
    """

    algorithm = "sha256_subject_epoch_shuffle_and_cyclic_visit_rotation_v1"

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        *,
        seed: int,
        bags_per_subject: int = DEFAULT_BAGS_PER_SUBJECT,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(bags_per_subject, bool)
            or not isinstance(bags_per_subject, int)
            or bags_per_subject <= 0
        ):
            raise ValueError("bags_per_subject must be a positive integer")
        grouped: dict[str, list[CaseRecord]] = defaultdict(list)
        for case in cases:
            if not isinstance(case, CaseRecord):
                raise TypeError("cases must contain CaseRecord values")
            grouped[case.subject_id].append(case)
        if not grouped:
            raise ValueError("subject schedule requires at least one case")

        self.seed = seed
        self.bags_per_subject = bags_per_subject
        self.subject_ids = tuple(sorted(grouped))
        ordered_cases = tuple(sorted(cases, key=_case_identity))
        self.cases = ordered_cases
        self._case_index = {_case_identity(case): index for index, case in enumerate(ordered_cases)}
        if len(self._case_index) != len(ordered_cases):
            raise LongRunError("training cases do not have unique full identities")
        self._cases_by_subject = {
            subject_id: tuple(
                sorted(
                    subject_cases,
                    key=lambda case: (
                        _seeded_digest(seed, "visit", subject_id, *_case_identity(case)),
                        _case_identity(case),
                    ),
                )
            )
            for subject_id, subject_cases in grouped.items()
        }
        self._visit_offsets = {
            subject_id: int(_seeded_digest(seed, "visit-offset", subject_id), 16)
            % len(subject_cases)
            for subject_id, subject_cases in self._cases_by_subject.items()
        }
        self._epoch_orders: dict[int, tuple[str, ...]] = {}
        self._sha256_cache = hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @property
    def subject_count(self) -> int:
        return len(self.subject_ids)

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def maximum_visits_per_subject(self) -> int:
        return max(len(cases) for cases in self._cases_by_subject.values())

    @property
    def steps_per_subject_epoch(self) -> int:
        return self.subject_count * self.bags_per_subject

    @property
    def all_visits_covered_by_step(self) -> int:
        return self.steps_per_subject_epoch * self.maximum_visits_per_subject

    def _order_for_epoch(self, subject_epoch: int) -> tuple[str, ...]:
        if isinstance(subject_epoch, bool) or not isinstance(subject_epoch, int):
            raise TypeError("subject_epoch must be an integer")
        if subject_epoch < 0:
            raise ValueError("subject_epoch must be non-negative")
        cached = self._epoch_orders.get(subject_epoch)
        if cached is None:
            cached = tuple(
                sorted(
                    self.subject_ids,
                    key=lambda subject_id: (
                        _seeded_digest(
                            self.seed,
                            "subject-order",
                            subject_epoch,
                            subject_id,
                        ),
                        subject_id,
                    ),
                )
            )
            self._epoch_orders[subject_epoch] = cached
        return cached

    def assignment_for_step(self, absolute_step_index: int) -> SubjectStep:
        if isinstance(absolute_step_index, bool) or not isinstance(absolute_step_index, int):
            raise TypeError("absolute_step_index must be an integer")
        if absolute_step_index < 0:
            raise ValueError("absolute_step_index must be non-negative")
        subject_block, bag_index = divmod(absolute_step_index, self.bags_per_subject)
        subject_epoch, subject_position = divmod(subject_block, self.subject_count)
        subject_id = self._order_for_epoch(subject_epoch)[subject_position]
        subject_cases = self._cases_by_subject[subject_id]
        visit_rotation_index = (self._visit_offsets[subject_id] + subject_epoch) % len(
            subject_cases
        )
        case = subject_cases[visit_rotation_index]
        return SubjectStep(
            subject_id=subject_id,
            case_id=case.case_id,
            visit_id=case.visit_id,
            subject_epoch=subject_epoch,
            subject_position=subject_position,
            visit_rotation_index=visit_rotation_index,
            bag_index=bag_index,
            case_index=self._case_index[_case_identity(case)],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "seed": self.seed,
            "bags_per_subject": self.bags_per_subject,
            "subject_count": self.subject_count,
            "case_count": self.case_count,
            "steps_per_subject_epoch": self.steps_per_subject_epoch,
            "maximum_visits_per_subject": self.maximum_visits_per_subject,
            "all_visits_covered_by_step": self.all_visits_covered_by_step,
            "subjects": [
                {
                    "subject_id": subject_id,
                    "visit_offset": self._visit_offsets[subject_id],
                    "cases": [case.case_id for case in self._cases_by_subject[subject_id]],
                }
                for subject_id in self.subject_ids
            ],
        }

    @property
    def sha256(self) -> str:
        return self._sha256_cache


class SubjectBalancedBatchFactory(DeterministicRealBatchFactory):
    """Real batch factory driven by an absolute subject-balanced schedule."""

    def __init__(self, *, schedule: SubjectBalancedSchedule, **kwargs: object) -> None:
        self.schedule = schedule
        super().__init__(
            cases=schedule.cases,
            bags_per_case=schedule.bags_per_subject,
            **kwargs,
        )

    def _assignment_for_absolute_step(self, absolute_step_index: int) -> StepAssignment:
        scheduled = self.schedule.assignment_for_step(absolute_step_index)
        return StepAssignment(
            case_index=scheduled.case_index,
            epoch=scheduled.subject_epoch,
            bag_index=scheduled.bag_index,
        )

    def materialize(
        self,
        absolute_step_index: int,
        *,
        prime_lookahead: bool = True,
    ) -> Any:
        if self._cached_step == absolute_step_index:
            return self._cached_batch
        if not isinstance(prime_lookahead, bool):
            raise TypeError("prime_lookahead must be boolean")
        if prime_lookahead:
            self.prime(absolute_step_index)
        scheduled = self.schedule.assignment_for_step(absolute_step_index)
        assignment = self._assignment_for_absolute_step(absolute_step_index)
        batch = self._batch_for_assignment(absolute_step_index, assignment)
        if self.last_record is None:
            raise LongRunError("subject-balanced batch lacks a provenance record")
        self.last_record.update(
            {
                "schedule_sha256": self.schedule.sha256,
                "subject_epoch": scheduled.subject_epoch,
                "subject_position": scheduled.subject_position,
                "visit_rotation_index": scheduled.visit_rotation_index,
            }
        )
        return batch


@contextmanager
def _managed_batch_factory(
    factory: DeterministicRealBatchFactory,
) -> Iterator[DeterministicRealBatchFactory]:
    """Close a factory on every exit path, including pre-training setup failures."""

    try:
        yield factory
    finally:
        factory.close()


class WalltimeStop:
    """Signal-safe flag checked only after a fully completed optimizer step."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handle(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = signal_number

    def __call__(self) -> bool:
        return self.requested


@dataclass(frozen=True, slots=True)
class InvocationIdentity:
    token: str
    stem: str
    wandb_id: str


def _invocation_identity(
    provenance_sha256: str,
    *,
    start_step: int,
    stop_step: int,
) -> InvocationIdentity:
    if _SHA256.fullmatch(provenance_sha256) is None:
        raise ValueError("provenance_sha256 must be a lowercase SHA-256 digest")
    if not 0 <= start_step <= stop_step:
        raise ValueError("invocation steps must satisfy 0 <= start <= stop")
    token = _safe_invocation_token()
    stem = f"start-{start_step:09d}-stop-{stop_step:09d}-{token}"
    wandb_id = hashlib.sha256(f"{provenance_sha256}\0{start_step}\0{token}".encode()).hexdigest()[
        :24
    ]
    return InvocationIdentity(token=token, stem=stem, wandb_id=wandb_id)


@contextmanager
def _exclusive_run_lock(destination: Path) -> Iterator[None]:
    lock_path = destination / ".long-run.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LongRunError(f"another process is already using {destination}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_or_require(
    path: Path,
    value: Mapping[str, object],
    *,
    resuming: bool,
    description: str,
) -> str:
    if resuming and os.path.lexists(path):
        _require_canonical(path, value, description)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return _write_new_canonical(path, value)


def _resume_step(path: Path | None) -> int:
    if path is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise LongRunError("resume checkpoint must use step-NNNNNNNNN.pt naming")
    return int(match.group(1))


def _safe_invocation_token() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raw = f"pid-{os.getpid()}"
    else:
        restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
        if not restart_count.isdigit():
            raise LongRunError("SLURM_RESTART_COUNT must be a non-negative integer")
        raw = f"{job_id}-restart-{int(restart_count)}"
    token = _SAFE_TOKEN.sub("-", raw).strip("-")
    if not token:
        raise LongRunError("could not derive a safe invocation token")
    return token


def _validate_long_config(config: ExperimentConfig) -> None:
    if config != _VALIDATED_CONFIG:
        raise LongRunError(
            "long pretraining is locked to the exact validated 4 mm small-model config"
        )


def _probe_cases(schedule: SubjectBalancedSchedule) -> tuple[CaseRecord, ...]:
    selected: list[CaseRecord] = []
    subjects: set[str] = set()
    for subject_position in range(schedule.subject_count):
        scheduled = schedule.assignment_for_step(subject_position * schedule.bags_per_subject)
        case = schedule.cases[scheduled.case_index]
        if case.subject_id in subjects:
            continue
        selected.append(case)
        subjects.add(case.subject_id)
        if len(selected) == _FIXED_PROBE_SUBJECTS:
            return tuple(selected)
    raise LongRunError("training split cannot provide four fixed-probe subjects")


def _initialize_destination(
    output_dir: str | os.PathLike[str],
    resume_from: str | os.PathLike[str] | None,
    *,
    resume_existing_output: bool = False,
) -> tuple[Path, Path | None, bool]:
    requested = Path(output_dir).expanduser()
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if resume_from is not None and resume_existing_output:
        raise LongRunError("resume checkpoint and zero-checkpoint recovery are mutually exclusive")
    if resume_from is None and not resume_existing_output:
        destination.mkdir(mode=0o700, exist_ok=False)
        for child in (
            "plans",
            "fixed-probe-plans",
            "checkpoints",
            "metrics",
            "invocations",
        ):
            (destination / child).mkdir(mode=0o700)
        return destination, None, False

    if destination.is_symlink() or not destination.is_dir():
        raise LongRunError("resume output must be an existing non-symlink directory")
    for child in (
        "plans",
        "fixed-probe-plans",
        "checkpoints",
        "metrics",
        "invocations",
    ):
        path = destination / child
        if path.is_symlink():
            raise LongRunError(f"resume output directory {child!r} must not be a symlink")
        if path.exists() and not path.is_dir():
            raise LongRunError(f"resume output entry {child!r} must be a directory")
        if not path.exists():
            if not resume_existing_output:
                raise LongRunError(f"resume output is missing directory {child!r}")
            path.mkdir(mode=0o700)
    if resume_existing_output:
        return destination, None, True
    assert resume_from is not None
    resume = _resolve_file(resume_from, "resume checkpoint")
    checkpoint_root = (destination / "checkpoints").resolve(strict=True)
    if resume.parent != checkpoint_root:
        raise LongRunError("resume checkpoint must belong to this output bundle")
    return destination, resume, True


def _validate_zero_checkpoint_recovery(destination: Path) -> None:
    """Validate a deterministic restart from zero before the first checkpoint."""

    checkpoints = destination / "checkpoints"
    if any(checkpoints.iterdir()):
        raise LongRunError("zero-checkpoint recovery found an unexpected checkpoint entry")
    if os.path.lexists(destination / "result.json"):
        raise LongRunError("zero-checkpoint recovery cannot replace a result artifact")
    if any((destination / "invocations").iterdir()):
        raise LongRunError("zero-checkpoint recovery found a completed invocation record")
    maximum_metric_step = 0
    for path in (destination / "metrics").iterdir():
        if (
            path.is_symlink()
            or not path.is_file()
            or _ZERO_RESTART_METRICS_NAME.fullmatch(path.name) is None
        ):
            raise LongRunError(f"unsafe zero-checkpoint metrics entry: {path.name}")
        observed_steps: list[int] = []
        raw_lines = path.read_bytes().splitlines()
        for line_number, raw_line in enumerate(raw_lines, start=1):
            try:
                value = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                # Metrics are observational and are written after a completed
                # optimizer/EMA step.  A node loss may tear only the final
                # buffered JSONL record; preserving that abandoned attempt
                # must not prevent a deterministic restart from step zero.
                if line_number == len(raw_lines):
                    break
                raise LongRunError(
                    f"invalid recovery metrics JSON at {path.name}:{line_number}"
                ) from error
            if not isinstance(value, dict) or canonical_json_bytes(value) != raw_line:
                raise LongRunError(f"noncanonical recovery metrics at {path.name}:{line_number}")
            step = value.get("step")
            if (
                value.get("schema") != "simple-brats.long-run-step"
                or value.get("schema_version") != 3
                or isinstance(step, bool)
                or not isinstance(step, int)
                or not 1 <= step <= _VALIDATED_CONFIG.checkpoint_every_steps
            ):
                raise LongRunError(f"unsupported recovery metrics at {path.name}:{line_number}")
            observed_steps.append(step)
        if observed_steps and observed_steps != list(range(1, observed_steps[-1] + 1)):
            raise LongRunError(f"recovery metrics are not a step prefix: {path.name}")
        if observed_steps:
            maximum_metric_step = max(maximum_metric_step, observed_steps[-1])

    plan_files: dict[int, set[str]] = {}
    for path in (destination / "plans").iterdir():
        if path.is_symlink() or not path.is_file():
            raise LongRunError(f"unsafe zero-checkpoint plan entry: {path.name}")
        match = _TRAINING_PLAN_NAME.fullmatch(path.name)
        if match is None:
            raise LongRunError(f"unsafe zero-checkpoint plan filename: {path.name}")
        step = int(match.group(1))
        if not 1 <= step <= _VALIDATED_CONFIG.checkpoint_every_steps:
            raise LongRunError(f"zero-checkpoint plan is beyond first checkpoint: {path.name}")
        plan_files.setdefault(step, set()).add(match.group(2))
    if plan_files:
        maximum_plan_step = max(plan_files)
        if set(plan_files) != set(range(1, maximum_plan_step + 1)):
            raise LongRunError("zero-checkpoint plans are not a contiguous step prefix")
        for step, suffixes in plan_files.items():
            required = {"plan.json", "prepared.json"}
            if suffixes == required:
                continue
            if step == maximum_plan_step and suffixes == {"plan.json"}:
                continue
            raise LongRunError(f"incomplete zero-checkpoint plan/audit pair at step {step}")
        if maximum_metric_step > maximum_plan_step:
            raise LongRunError("recovery metrics extend beyond materialized plan prefix")
    elif maximum_metric_step:
        raise LongRunError("recovery metrics exist without materialized plans")


def _discard_stale_atomic_temporaries(destination: Path) -> None:
    """Remove unpublished same-directory temporaries left by node loss."""

    for directory in (
        destination,
        destination / "plans",
        destination / "fixed-probe-plans",
        destination / "checkpoints",
        destination / "metrics",
        destination / "invocations",
    ):
        for path in directory.glob(".*.tmp-*"):
            if path.is_symlink() or not path.is_file():
                raise LongRunError(f"unsafe stale atomic temporary: {path}")
            path.unlink()
        if directory == destination / "checkpoints":
            for path in directory.glob(".*.failure-tmp-*"):
                if path.is_symlink() or not path.is_file():
                    raise LongRunError(f"unsafe stale failure checkpoint: {path}")
                path.unlink()


def _log_terminal_recovery_artifact(
    checkpoint_manager: CheckpointManager,
    result: Any,
    provenance: Mapping[str, object],
) -> bool:
    """Stage the already-validated terminal checkpoint in the recovery W&B run."""

    if result.start_step != result.total_steps:
        return False
    if result.end_step != result.total_steps:
        raise LongRunError("terminal recovery did not remain at the terminal checkpoint")
    checkpoint = result.latest_checkpoint
    if checkpoint is None or _resume_step(Path(checkpoint)) != result.total_steps:
        raise LongRunError("terminal recovery did not validate the expected final checkpoint")
    if not checkpoint_manager.policy.is_artifact_step(result.total_steps):
        raise LongRunError("terminal checkpoint is not on the registered W&B artifact cadence")
    sink = checkpoint_manager.artifact_sink
    if sink is None:
        raise LongRunError("terminal recovery requires a W&B artifact sink")
    sink.log_checkpoint(
        Path(checkpoint),
        step=result.total_steps,
        metadata=provenance,
    )
    return True


def run_long_matching(
    *,
    data_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    expected_manifest_sha256: str,
    split_path: str | os.PathLike[str],
    expected_split_sha256: str,
    case_grid_manifest_path: str | os.PathLike[str],
    expected_case_grid_manifest_sha256: str,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    expected_git_sha: str,
    repo_root: str | os.PathLike[str] = ".",
    total_steps: int = DEFAULT_TOTAL_STEPS,
    max_steps_per_invocation: int = DEFAULT_MAX_STEPS_PER_INVOCATION,
    bags_per_subject: int = DEFAULT_BAGS_PER_SUBJECT,
    expected_train_cases: int = DEFAULT_EXPECTED_TRAIN_CASES,
    expected_train_subjects: int = DEFAULT_EXPECTED_TRAIN_SUBJECTS,
    resume_from: str | os.PathLike[str] | None = None,
    resume_existing_output: bool = False,
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Run one checkpoint-aligned invocation of the immutable long schedule."""

    if _FULL_GIT_SHA.fullmatch(expected_git_sha) is None:
        raise ValueError("expected_git_sha must be one full lowercase commit ID")
    for value, name in (
        (expected_manifest_sha256, "expected_manifest_sha256"),
        (expected_split_sha256, "expected_split_sha256"),
        (expected_case_grid_manifest_sha256, "expected_case_grid_manifest_sha256"),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for value, name in (
        (total_steps, "total_steps"),
        (max_steps_per_invocation, "max_steps_per_invocation"),
        (bags_per_subject, "bags_per_subject"),
        (expected_train_cases, "expected_train_cases"),
        (expected_train_subjects, "expected_train_subjects"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if bags_per_subject != DEFAULT_BAGS_PER_SUBJECT:
        raise LongRunError("the registered long schedule requires eight bags per subject block")

    resolved_device = torch.device(device)
    exact_resume_runtime = configure_exact_resume_runtime(resolved_device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise LongRunError("CUDA was requested but is unavailable")
    try:
        training_runtime = configure_training_runtime(resolved_device)
    except TrainingRuntimeError as error:
        raise LongRunError(f"registered model runtime is unavailable: {error}") from error
    repo = Path(repo_root).expanduser().resolve(strict=True)
    launch_sha = verify_git_sha(expected_git_sha, repo)
    manifest_file = _resolve_file(manifest_path, "filtered manifest")
    split_file = _resolve_file(split_path, "subject split")
    grids_file = _resolve_file(case_grid_manifest_path, "case-grid manifest")
    config_file = _resolve_file(config_path, "experiment config")
    manifest = load_manifest(manifest_file, expected_sha256=expected_manifest_sha256)
    split = load_split(split_file, expected_sha256=expected_split_sha256)
    case_grids = load_case_grid_manifest(
        grids_file,
        expected_sha256=expected_case_grid_manifest_sha256,
    )
    config = load_experiment_config(config_file)
    _validate_long_config(config)
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    _require_canonical(manifest_file, manifest.to_dict(), "filtered manifest")
    _require_canonical(split_file, split.to_dict(), "subject split")

    train_cases = cases_for_splits(manifest, split, ("train",))
    schedule = SubjectBalancedSchedule(
        train_cases,
        seed=config.seed,
        bags_per_subject=bags_per_subject,
    )
    if schedule.case_count != expected_train_cases:
        raise LongRunError(
            f"locked train case count is {expected_train_cases}, observed {schedule.case_count}"
        )
    if schedule.subject_count != expected_train_subjects:
        raise LongRunError(
            "locked train subject count is "
            f"{expected_train_subjects}, observed {schedule.subject_count}"
        )
    if total_steps < schedule.all_visits_covered_by_step:
        raise LongRunError(
            f"total_steps={total_steps} cannot expose every training visit; "
            f"at least {schedule.all_visits_covered_by_step} are required"
        )
    if total_steps % config.checkpoint_every_steps:
        raise LongRunError("total_steps must end on the registered checkpoint cadence")
    if total_steps % config.artifact_every_steps:
        raise LongRunError("total_steps must end on the registered W&B artifact cadence")
    if max_steps_per_invocation % config.checkpoint_every_steps:
        raise LongRunError("max_steps_per_invocation must end on the registered checkpoint cadence")
    if max_steps_per_invocation != config.artifact_every_steps:
        raise LongRunError("registered Slurm segments must contain exactly 5,000 steps")
    if os.environ.get("WANDB_MODE", "offline") != "offline":
        raise LongRunError("compute-node W&B must run in offline mode")
    try:
        wandb_module = _wandb_for_schedule(
            total_steps=total_steps,
            artifact_every_steps=config.artifact_every_steps,
        )
    except ShortRunError as error:
        raise LongRunError(str(error)) from error
    if wandb_module is None:
        raise LongRunError("long pretraining requires the pinned W&B tracking extra")

    destination, resume_checkpoint, resuming = _initialize_destination(
        output_dir,
        resume_from,
        resume_existing_output=resume_existing_output,
    )
    with _exclusive_run_lock(destination):
        if resume_existing_output:
            _discard_stale_atomic_temporaries(destination)
            _validate_zero_checkpoint_recovery(destination)
        return _run_locked(
            data_root=data_root,
            manifest=manifest,
            split=split,
            case_grids=case_grids,
            config=config,
            config_file=config_file,
            repo=repo,
            launch_sha=launch_sha,
            destination=destination,
            schedule=schedule,
            total_steps=total_steps,
            max_steps_per_invocation=max_steps_per_invocation,
            resume_checkpoint=resume_checkpoint,
            resuming=resuming,
            resolved_device=resolved_device,
            exact_resume_runtime=exact_resume_runtime,
            training_runtime=training_runtime,
            wandb_module=wandb_module,
        )


def _run_locked(
    *,
    data_root: str | os.PathLike[str],
    manifest: Any,
    split: Any,
    case_grids: Any,
    config: ExperimentConfig,
    config_file: Path,
    repo: Path,
    launch_sha: str,
    destination: Path,
    schedule: SubjectBalancedSchedule,
    total_steps: int,
    max_steps_per_invocation: int,
    resume_checkpoint: Path | None,
    resuming: bool,
    resolved_device: torch.device,
    exact_resume_runtime: Mapping[str, object],
    training_runtime: TrainingRuntimePolicy,
    wandb_module: Any,
) -> dict[str, object]:
    plans_dir = destination / "plans"
    probe_plans_dir = destination / "fixed-probe-plans"
    start_step = _resume_step(resume_checkpoint)
    if start_step > total_steps:
        raise LongRunError(f"resume step {start_step} is beyond total_steps={total_steps}")
    next_segment_boundary = (
        (start_step // max_steps_per_invocation) + 1
    ) * max_steps_per_invocation
    invocation_stop = min(total_steps, next_segment_boundary)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    system = build_matching_system(config).to(resolved_device).train()
    try:
        apply_model_runtime(system, training_runtime)
    except TrainingRuntimeError as error:
        raise LongRunError(f"could not apply registered model runtime: {error}") from error

    fixed_cases = _probe_cases(schedule)
    probe_build = _build_fixed_target_probe(
        data_root=data_root,
        manifest=manifest,
        case_grids=case_grids,
        cases=fixed_cases,
        config=config,
        plans_dir=probe_plans_dir,
        candidate_pool_size=_CANDIDATE_POOL_SIZE,
        max_plan_attempts=_MAX_PLAN_ATTEMPTS,
        replay_existing=resuming,
    )
    collapse_probe = probe_build.probe
    probe_artifact: dict[str, object] = {
        "schema": "simple-brats.long-run-fixed-target-probe",
        "schema_version": 1,
        "purpose": "representation-collapse-abort-only_no_gradient_or_model_selection",
        "source_split": "train",
        "subject_overlap_with_optimization_population": True,
        "validation_or_test_consumed": False,
        "probe_sha256": collapse_probe.sha256,
        "patch_table_shape": list(collapse_probe.target_patches.shape),
        "modality_id_table_shape": list(collapse_probe.target_modality_ids.shape),
        "sample_count_by_modality": {
            str(modality_id): count
            for modality_id, count in collapse_probe.sample_count_by_modality.items()
        },
        "case_ids": [case.case_id for case in fixed_cases],
        "subject_ids": [case.subject_id for case in fixed_cases],
        "bags_per_case": probe_build.bags_per_case,
        "bag_count": len(probe_build.records),
        "plans_directory": probe_plans_dir.name,
        "bags": list(probe_build.records),
    }
    probe_artifact_sha256 = _write_or_require(
        destination / "fixed-target-probe.json",
        probe_artifact,
        resuming=resuming,
        description="fixed target probe",
    )

    factory = SubjectBalancedBatchFactory(
        schedule=schedule,
        data_root=data_root,
        manifest=manifest,
        case_grids=case_grids,
        config=config,
        plans_dir=plans_dir,
        candidate_pool_size=_CANDIDATE_POOL_SIZE,
        max_plan_attempts=_MAX_PLAN_ATTEMPTS,
        replay_existing=True,
        optimized_runtime=(
            OptimizedRuntimeConfig() if resolved_device.type == "cuda" else None
        ),
        optimized_device=(resolved_device if resolved_device.type == "cuda" else None),
    )
    with _managed_batch_factory(factory):
        return _run_training_factory_lifetime(
            factory=factory,
            system=system,
            collapse_probe=collapse_probe,
            probe_artifact_sha256=probe_artifact_sha256,
            fixed_cases=fixed_cases,
            config=config,
            config_file=config_file,
            repo=repo,
            manifest=manifest,
            split=split,
            case_grids=case_grids,
            launch_sha=launch_sha,
            destination=destination,
            schedule=schedule,
            total_steps=total_steps,
            start_step=start_step,
            invocation_stop=invocation_stop,
            resume_checkpoint=resume_checkpoint,
            resuming=resuming,
            resolved_device=resolved_device,
            exact_resume_runtime=exact_resume_runtime,
            training_runtime=training_runtime,
            wandb_module=wandb_module,
        )


def _run_training_factory_lifetime(
    *,
    factory: SubjectBalancedBatchFactory,
    system: Any,
    collapse_probe: Any,
    probe_artifact_sha256: str,
    fixed_cases: Sequence[CaseRecord],
    config: ExperimentConfig,
    config_file: Path,
    repo: Path,
    manifest: Any,
    split: Any,
    case_grids: Any,
    launch_sha: str,
    destination: Path,
    schedule: SubjectBalancedSchedule,
    total_steps: int,
    start_step: int,
    invocation_stop: int,
    resume_checkpoint: Path | None,
    resuming: bool,
    resolved_device: torch.device,
    exact_resume_runtime: Mapping[str, object],
    training_runtime: TrainingRuntimePolicy,
    wandb_module: Any,
) -> dict[str, object]:
    calibration_batch = factory.materialize(0, prime_lookahead=False).to(resolved_device)
    if factory.last_record is None:
        raise LongRunError("calibration batch lacks a provenance record")
    calibration_batch_record = dict(factory.last_record)
    # Start exact schedule-keyed cold loads before compiled calibration so CPU
    # preparation overlaps Inductor work.  The retained-future barrier below
    # keeps cold I/O out of the optimizer stream even when compilation is
    # already cached on a resumed allocation.
    if start_step < invocation_stop:
        factory.prime(start_step)
    probe_patches = collapse_probe.target_patches.to(resolved_device)
    probe_modality_ids = collapse_probe.target_modality_ids.to(resolved_device)
    rng_state = _capture_torch_rng()
    try:
        try:
            with torch.no_grad(), training_runtime.autocast(resolved_device):
                calibration_output = system(calibration_batch)
                probe_targets = system.target_teacher(probe_patches)
        except Exception as error:
            if training_runtime.compile_enabled:
                raise LongRunError("compiled model calibration warmup failed") from error
            raise
    finally:
        _restore_torch_rng(rng_state)
    factory.wait_for_prefetch()
    references = stats_by_modality(probe_targets, probe_modality_ids)
    training_teacher_baseline = stats_by_modality(
        calibration_output.targets,
        calibration_batch.target_modality_ids,
    )
    prediction_baseline = stats_by_modality(
        calibration_output.predictions,
        calibration_batch.query_modality_ids,
    )
    expected_modalities = set(range(len(config.task.modalities)))
    if (
        set(references) != expected_modalities
        or any(value.variance <= 0 for value in references.values())
        or {modality_id: value.count for modality_id, value in references.items()}
        != collapse_probe.sample_count_by_modality
    ):
        raise LongRunError("initial teacher calibration is incomplete or degenerate")
    calibration: dict[str, object] = {
        "schema": "simple-brats.long-run-calibration",
        "schema_version": 1,
        "timing": "initialized_model_before_optimizer_construction_and_training",
        "training_batch": calibration_batch_record,
        "collapse_stream": TEACHER_TARGET_DIAGNOSTIC_STREAM,
        "fixed_probe": {
            "probe_sha256": collapse_probe.sha256,
            "artifact_file": "fixed-target-probe.json",
            "artifact_sha256": probe_artifact_sha256,
        },
        "teacher_reference_by_modality": _stats_record(references),
        "training_batch_teacher_baseline_by_modality": _stats_record(training_teacher_baseline),
        "training_batch_prediction_baseline_by_modality": _stats_record(prediction_baseline),
        "thresholds": _COLLAPSE_THRESHOLDS.to_dict(),
        "collapse_warmup_steps": _COLLAPSE_WARMUP_STEPS,
    }
    calibration_sha256 = _write_or_require(
        destination / "calibration.json",
        calibration,
        resuming=resuming,
        description="long-run calibration",
    )
    schedule_artifact = schedule.to_dict()
    _write_or_require(
        destination / "subject-schedule.json",
        schedule_artifact,
        resuming=resuming,
        description="subject schedule",
    )

    provenance: dict[str, object] = {
        "schema": "simple-brats.long-real-matching",
        "schema_version": 1,
        "launch_sha": launch_sha,
        "manifest_sha256": manifest.sha256,
        "split_sha256": split.sha256,
        "training_split": "train",
        "validation_or_test_consumed_by_ssl": False,
        "case_grid_manifest_sha256": case_grids.sha256,
        "case_grid_policy_sha256": case_grids.policy.sha256,
        "runtime_extraction_policy_sha256": case_grids.policy.for_patch_config(config.patch).sha256,
        "config_sha256": config.sha256,
        "config_file_sha256": sha256_file(config_file),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
        "calibration_sha256": calibration_sha256,
        "fixed_target_probe": {
            "probe_sha256": collapse_probe.sha256,
            "artifact_sha256": probe_artifact_sha256,
            "source_split": "train",
            "monitoring_only": True,
            "case_ids": [case.case_id for case in fixed_cases],
            "subject_ids": [case.subject_id for case in fixed_cases],
        },
        "objective": "hard_symmetric_conditional_info_nce",
        "run_classification": "checkpointed_representation_pretraining",
        "tracking_mode": "offline_wandb_segment_runs_and_canonical_jsonl",
        "exact_resume_runtime": dict(exact_resume_runtime),
        "training_runtime": training_runtime.to_dict(),
        "data_runtime": factory.runtime_contract,
        "selected_train_case_ids": [case.case_id for case in schedule.cases],
        "selected_train_subject_ids": list(schedule.subject_ids),
        "schedule": {
            "total_steps": total_steps,
            "subject_schedule_sha256": schedule.sha256,
            **schedule_artifact,
            "absolute_step_factory": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": _LEARNING_RATE,
            "weight_decay": _WEIGHT_DECAY,
            "gradient_clip_norm": _GRADIENT_CLIP_NORM,
            "implementation": training_runtime.to_dict()["optimizer"],
        },
    }
    _write_or_require(
        destination / "run-provenance.json",
        provenance,
        resuming=resuming,
        description="long-run provenance",
    )
    provenance_sha256 = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()

    invocation_identity = _invocation_identity(
        provenance_sha256,
        start_step=start_step,
        stop_step=invocation_stop,
    )
    invocation_stem = invocation_identity.stem
    wandb_group = f"long-{provenance_sha256[:20]}"
    wandb_id = invocation_identity.wandb_id
    invocation_metadata = {
        "start_step": start_step,
        "planned_stop_step": invocation_stop,
        "max_steps": invocation_stop - start_step,
        "resume_checkpoint": (resume_checkpoint.name if resume_checkpoint is not None else None),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "global_provenance_sha256": provenance_sha256,
    }
    wandb_run = wandb_module.init(
        project="simple-brats",
        name=f"{destination.name}-{start_step:09d}-{invocation_stop:09d}",
        id=wandb_id,
        group=wandb_group,
        job_type="pretrain-segment",
        dir=str(destination),
        mode="offline",
        config={"global": provenance, "invocation": invocation_metadata},
        reinit=True,
    )
    if wandb_run is None:
        raise LongRunError("offline W&B initialization returned no run")

    with ExitStack() as cleanup:
        # Register cleanup immediately after each resource exists.  ExitStack
        # runs every callback even if an earlier cleanup raises, so a logger or
        # signal-restoration failure cannot strand the offline W&B run (and the
        # outer factory context still closes all prefetch workers).
        cleanup.callback(wandb_run.finish)
        try:
            optimizer = build_adamw_optimizer(
                system,
                learning_rate=_LEARNING_RATE,
                weight_decay=_WEIGHT_DECAY,
                policy=training_runtime,
            )
        except TrainingRuntimeError as error:
            raise LongRunError(
                f"could not build registered optimizer runtime: {error}"
            ) from error
        checkpoint_manager = CheckpointManager(
            destination / "checkpoints",
            policy=CheckpointPolicy(
                checkpoint_every_steps=config.checkpoint_every_steps,
                artifact_every_steps=config.artifact_every_steps,
            ),
            artifact_sink=WandbArtifactSink(wandb_run),
        )
        logger = _MetricsLogger(
            destination / "metrics" / f"{invocation_stem}.jsonl",
            factory,
            wandb_run,
            schema="simple-brats.long-run-step",
        )
        cleanup.callback(logger.close)
        walltime_stop = WalltimeStop()
        previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, walltime_stop.handle)
        cleanup.callback(signal.signal, signal.SIGUSR1, previous_handler)
        terminal_artifact_logged = False
        result = run_matching_training(
            system,
            optimizer,
            factory,
            checkpoint_manager,
            provenance,
            total_steps=total_steps,
            max_steps=invocation_stop - start_step,
            resume_from=resume_checkpoint,
            collapse_probe=collapse_probe,
            collapse_reference=references,
            collapse_thresholds=_COLLAPSE_THRESHOLDS,
            collapse_warmup_steps=_COLLAPSE_WARMUP_STEPS,
            gradient_clip_norm=_GRADIENT_CLIP_NORM,
            runtime_policy=training_runtime,
            on_step=logger,
            should_stop=walltime_stop,
        )
        terminal_artifact_logged = _log_terminal_recovery_artifact(
            checkpoint_manager,
            result,
            provenance,
        )

    if result.latest_checkpoint is None or not result.latest_checkpoint.exists():
        raise LongRunError("long-run invocation ended without a reusable checkpoint")
    report: dict[str, object] = {
        "schema": "simple-brats.long-run-invocation",
        "schema_version": 1,
        "status": "walltime_checkpointed" if walltime_stop.requested else "ok",
        "start_step": result.start_step,
        "end_step": result.end_step,
        "total_steps": result.total_steps,
        "ema_update_count": result.ema_update_count,
        "latest_checkpoint": str(result.latest_checkpoint.relative_to(destination)),
        "runner_contract_sha256": result.runner_contract_sha256,
        "global_provenance_sha256": provenance_sha256,
        "subject_schedule_sha256": schedule.sha256,
        "walltime_signal": walltime_stop.signal_number,
        "data_runtime": factory.runtime_contract,
        "data_runtime_stats": factory.runtime_stats(),
        "terminal_recovery_artifact_logged": terminal_artifact_logged,
        "wandb": {
            "mode": "offline",
            "project": "simple-brats",
            "group": wandb_group,
            "run_id": wandb_id,
            "sync_root": "wandb",
        },
    }
    _write_new_canonical(
        destination / "invocations" / f"{invocation_stem}.json",
        report,
    )
    if result.end_step == total_steps:
        _write_or_require(
            destination / "result.json",
            report,
            resuming=False,
            description="completed long-run result",
        )
    print(canonical_json_bytes(report).decode("utf-8"), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one resumable segment of subject-balanced hard matching"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--case-grid-manifest", required=True)
    parser.add_argument("--expected-case-grid-manifest-sha256", required=True)
    parser.add_argument("--config", default="configs/v0_cross_matching_small.toml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument(
        "--max-steps-per-invocation",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_INVOCATION,
    )
    parser.add_argument(
        "--bags-per-subject",
        type=int,
        default=DEFAULT_BAGS_PER_SUBJECT,
    )
    parser.add_argument(
        "--expected-train-cases",
        type=int,
        default=DEFAULT_EXPECTED_TRAIN_CASES,
    )
    parser.add_argument(
        "--expected-train-subjects",
        type=int,
        default=DEFAULT_EXPECTED_TRAIN_SUBJECTS,
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--resume-existing-output", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_long_matching(
        data_root=args.data_root,
        manifest_path=args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        split_path=args.split,
        expected_split_sha256=args.expected_split_sha256,
        case_grid_manifest_path=args.case_grid_manifest,
        expected_case_grid_manifest_sha256=args.expected_case_grid_manifest_sha256,
        config_path=args.config,
        output_dir=args.output_dir,
        expected_git_sha=args.expected_git_sha,
        repo_root=args.repo_root,
        total_steps=args.total_steps,
        max_steps_per_invocation=args.max_steps_per_invocation,
        bags_per_subject=args.bags_per_subject,
        expected_train_cases=args.expected_train_cases,
        expected_train_subjects=args.expected_train_subjects,
        resume_from=args.resume_from,
        resume_existing_output=args.resume_existing_output,
        device=args.device,
    )
    if report["status"] == "walltime_checkpointed":
        return WALLTIME_REQUEUE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
