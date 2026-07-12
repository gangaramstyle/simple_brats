from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import simple_brats.long_run as long_run_module
from simple_brats.config import load_experiment_config
from simple_brats.data.manifest import CaseRecord, FileRecord, canonical_json_bytes
from simple_brats.long_run import (
    DEFAULT_BAGS_PER_SUBJECT,
    LongRunError,
    SubjectBalancedSchedule,
    _discard_stale_atomic_temporaries,
    _initialize_destination,
    _invocation_identity,
    _log_terminal_recovery_artifact,
    _managed_batch_factory,
    _probe_cases,
    _safe_invocation_token,
    _validate_long_config,
    _validate_zero_checkpoint_recovery,
    _write_or_require,
    configure_exact_resume_runtime,
)
from simple_brats.short_run import ShortRunError
from simple_brats.tracking import OnlineWandbConfig
from simple_brats.training import (
    CheckpointManager,
    CheckpointPolicy,
    RepresentationStats,
    TrainingRuntimePolicy,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_long_run_accepts_exact_two_registered_scale_matched_arms() -> None:
    for path in (
        "configs/v0_cross_matching_small.toml",
        "configs/v0_cross_matching_small_8mm.toml",
    ):
        _validate_long_config(load_experiment_config(path))
    with pytest.raises(LongRunError, match="registered"):
        _validate_long_config(load_experiment_config("configs/v0_cross_matching.toml"))


def _case(subject: int, visit: int) -> CaseRecord:
    case_id = f"BraTS-MET-{subject:05d}-{visit:03d}"
    return CaseRecord.create(
        source="BraTS-MET",
        release="r1",
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality=modality,
                path=f"{case_id}/{case_id}-{modality}.nii.gz",
                sha256=_digest(f"{case_id}:{modality}"),
            )
            for modality in ("t1n", "t1c", "t2w", "t2f")
        ),
    )


def _cases() -> tuple[CaseRecord, ...]:
    return (
        _case(1, 0),
        _case(1, 1),
        _case(1, 2),
        _case(2, 0),
        _case(3, 0),
        _case(3, 1),
        _case(4, 0),
        _case(5, 0),
    )


def test_managed_factory_closes_when_setup_raises() -> None:
    factory = SimpleNamespace(close_calls=0)

    def close() -> None:
        factory.close_calls += 1

    factory.close = close
    with pytest.raises(RuntimeError, match="setup failed"):
        with _managed_batch_factory(factory):  # type: ignore[arg-type]
            raise RuntimeError("setup failed")
    assert factory.close_calls == 1


def test_run_locked_closes_optimized_factory_on_calibration_setup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(1, 0)
    modality_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    probe = SimpleNamespace(
        sha256="a" * 64,
        target_patches=torch.zeros(1, 8, 2),
        target_modality_ids=modality_ids,
        sample_count_by_modality={modality: 2 for modality in range(4)},
    )

    class System:
        def to(self, _device: object) -> System:
            return self

        def train(self) -> System:
            return self

    class RecordingFactory:
        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    created: list[RecordingFactory] = []

    def build_factory(**kwargs: object) -> RecordingFactory:
        factory = RecordingFactory(**kwargs)
        created.append(factory)
        return factory

    monkeypatch.setattr(long_run_module, "build_matching_system", lambda _config: System())
    monkeypatch.setattr(long_run_module, "apply_model_runtime", lambda *_args: None)
    monkeypatch.setattr(long_run_module, "_probe_cases", lambda _schedule: (case,))
    monkeypatch.setattr(
        long_run_module,
        "_build_fixed_target_probe",
        lambda **_kwargs: SimpleNamespace(
            probe=probe,
            bags_per_case=2,
            records=(),
        ),
    )
    monkeypatch.setattr(long_run_module, "_write_or_require", lambda *_args, **_kwargs: "b" * 64)
    monkeypatch.setattr(long_run_module, "SubjectBalancedBatchFactory", build_factory)
    monkeypatch.setattr(
        long_run_module,
        "_run_training_factory_lifetime",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("calibration setup failed")),
    )

    with pytest.raises(RuntimeError, match="calibration setup failed"):
        long_run_module._run_locked(
            config=SimpleNamespace(seed=0),  # type: ignore[arg-type]
            config_file=tmp_path / "config.toml",
            repo=tmp_path,
            data_root=tmp_path,
            manifest=SimpleNamespace(),
            split=SimpleNamespace(),
            case_grids=SimpleNamespace(),
            launch_sha="c" * 40,
            destination=tmp_path,
            schedule=SimpleNamespace(),  # type: ignore[arg-type]
            total_steps=50_000,
            max_steps_per_invocation=5_000,
            resume_checkpoint=None,
            resuming=False,
            resolved_device=torch.device("cpu"),
            exact_resume_runtime={},
            training_runtime=TrainingRuntimePolicy.eager_cpu(),
            wandb_module=SimpleNamespace(),
            wandb_tracking=OnlineWandbConfig(project="simple-brats", entity=None, base_url=None),
        )

    assert len(created) == 1
    assert created[0].close_calls == 1


@pytest.mark.parametrize("start_step", [0, 5_000])
def test_calibration_primes_training_factory_once_without_discarding_lookahead(
    start_step: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modality_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
    calibration_batch = SimpleNamespace(
        target_modality_ids=modality_ids,
        query_modality_ids=modality_ids,
        to=lambda _device: calibration_batch,
    )
    features = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    calibration_output = SimpleNamespace(targets=features, predictions=features)

    class CalibrationSystem:
        def __call__(self, _batch: object) -> object:
            return calibration_output

        @staticmethod
        def target_teacher(_patches: torch.Tensor) -> torch.Tensor:
            return features

    class TrainingFactory:
        runtime_contract = {"optimized": True}

        def __init__(self) -> None:
            self.last_record = None
            self.materialize_calls: list[tuple[int, bool]] = []
            self.prime_calls: list[int] = []
            self.wait_calls = 0
            self.discard_calls = 0

        def materialize(self, step: int, *, prime_lookahead: bool = True) -> object:
            self.materialize_calls.append((step, prime_lookahead))
            self.last_record = {"absolute_step_index": step}
            return calibration_batch

        def prime(self, step: int) -> tuple[int, ...]:
            self.prime_calls.append(step)
            return ()

        def wait_for_prefetch(self) -> tuple[int, ...]:
            self.wait_calls += 1
            return ()

        def discard_prefetch(self) -> tuple[int, ...]:
            self.discard_calls += 1
            return ()

    factory = TrainingFactory()
    stats = {
        modality: RepresentationStats(
            count=2,
            variance=1.0,
            effective_rank=2.0,
            off_diagonal_cosine=0.0,
        )
        for modality in range(4)
    }
    collapse_probe = SimpleNamespace(
        target_patches=torch.zeros(1, 8, 2),
        target_modality_ids=modality_ids,
        sample_count_by_modality={modality: 2 for modality in range(4)},
        sha256="a" * 64,
    )
    monkeypatch.setattr(long_run_module, "stats_by_modality", lambda *_args: stats)
    monkeypatch.setattr(
        long_run_module,
        "_write_or_require",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after calibration")),
    )

    with pytest.raises(RuntimeError, match="stop after calibration"):
        long_run_module._run_training_factory_lifetime(
            factory=factory,  # type: ignore[arg-type]
            system=CalibrationSystem(),
            collapse_probe=collapse_probe,
            probe_artifact_sha256="b" * 64,
            fixed_cases=(),
            config=SimpleNamespace(task=SimpleNamespace(modalities=(0, 1, 2, 3))),
            config_file=tmp_path / "config.toml",
            repo=tmp_path,
            manifest=SimpleNamespace(),
            split=SimpleNamespace(),
            case_grids=SimpleNamespace(),
            launch_sha="c" * 40,
            destination=tmp_path,
            schedule=SimpleNamespace(),  # type: ignore[arg-type]
            total_steps=start_step + 1,
            start_step=start_step,
            invocation_stop=start_step + 1,
            resume_checkpoint=None,
            resuming=bool(start_step),
            resolved_device=torch.device("cpu"),
            exact_resume_runtime={},
            training_runtime=TrainingRuntimePolicy.eager_cpu(),
            wandb_module=SimpleNamespace(),
            wandb_tracking=OnlineWandbConfig(project="simple-brats", entity=None, base_url=None),
        )

    assert factory.materialize_calls == [(0, False)]
    assert factory.prime_calls == [start_step]
    assert factory.wait_calls == 1
    assert factory.discard_calls == 0


def test_subject_schedule_is_deterministic_and_input_order_invariant() -> None:
    cases = _cases()
    first = SubjectBalancedSchedule(cases, seed=17)
    second = SubjectBalancedSchedule(tuple(reversed(cases)), seed=17)

    assert first.sha256 == second.sha256
    assert first.to_dict() == second.to_dict()
    assert [first.assignment_for_step(step) for step in range(500)] == [
        second.assignment_for_step(step) for step in range(500)
    ]


def test_each_subject_gets_one_consecutive_block_per_subject_epoch() -> None:
    schedule = SubjectBalancedSchedule(_cases(), seed=3)
    observed = [
        schedule.assignment_for_step(step) for step in range(schedule.steps_per_subject_epoch)
    ]

    blocks = [
        observed[index : index + DEFAULT_BAGS_PER_SUBJECT]
        for index in range(0, len(observed), DEFAULT_BAGS_PER_SUBJECT)
    ]
    assert len(blocks) == schedule.subject_count
    assert len({block[0].subject_id for block in blocks}) == schedule.subject_count
    assert all(len({item.subject_id for item in block}) == 1 for block in blocks)
    assert all(
        [item.bag_index for item in block] == list(range(DEFAULT_BAGS_PER_SUBJECT))
        for block in blocks
    )
    assert all(item.subject_epoch == 0 for item in observed)


def test_longitudinal_visits_rotate_without_case_count_weighting() -> None:
    schedule = SubjectBalancedSchedule(_cases(), seed=11)
    cases_by_subject = {
        subject_id: {case.case_id for case in schedule.cases if case.subject_id == subject_id}
        for subject_id in schedule.subject_ids
    }
    observed: dict[str, set[str]] = {subject_id: set() for subject_id in schedule.subject_ids}

    for subject_epoch in range(schedule.maximum_visits_per_subject):
        base = subject_epoch * schedule.steps_per_subject_epoch
        for block in range(schedule.subject_count):
            assignment = schedule.assignment_for_step(base + block * DEFAULT_BAGS_PER_SUBJECT)
            observed[assignment.subject_id].add(assignment.case_id)

    assert observed == cases_by_subject
    assert schedule.all_visits_covered_by_step == (
        schedule.maximum_visits_per_subject * schedule.steps_per_subject_epoch
    )


def test_absolute_resume_assignment_has_no_mutable_cursor() -> None:
    schedule = SubjectBalancedSchedule(_cases(), seed=5)
    step = 137
    expected = schedule.assignment_for_step(step)
    for unrelated_step in (0, 999, 16, 80, 1_000_000):
        schedule.assignment_for_step(unrelated_step)
    assert schedule.assignment_for_step(step) == expected


def test_fixed_collapse_probe_uses_training_schedule_subjects_only() -> None:
    schedule = SubjectBalancedSchedule(_cases(), seed=0)
    probe = _probe_cases(schedule)

    assert len(probe) == 4
    assert len({case.subject_id for case in probe}) == 4
    assert set(probe).issubset(set(schedule.cases))


@pytest.mark.parametrize("step", [-1, True, 1.5])
def test_subject_schedule_rejects_invalid_absolute_steps(step: object) -> None:
    schedule = SubjectBalancedSchedule(_cases(), seed=0)
    with pytest.raises((TypeError, ValueError)):
        schedule.assignment_for_step(step)  # type: ignore[arg-type]


def test_zero_checkpoint_output_restart_reconstructs_dirs_and_static_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "long"
    destination, checkpoint, resuming = _initialize_destination(output, None)
    assert checkpoint is None and not resuming
    (destination / "metrics").rmdir()

    recovered, checkpoint, resuming = _initialize_destination(
        output,
        None,
        resume_existing_output=True,
    )
    assert recovered == destination
    assert checkpoint is None and resuming
    _validate_zero_checkpoint_recovery(recovered)

    value = {"schema": "immutable", "version": 1}
    artifact = recovered / "run-provenance.json"
    digest = _write_or_require(
        artifact,
        value,
        resuming=True,
        description="provenance",
    )
    assert digest == hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert (
        _write_or_require(
            artifact,
            value,
            resuming=True,
            description="provenance",
        )
        == digest
    )
    with pytest.raises(ShortRunError, match="not canonical"):
        _write_or_require(
            artifact,
            {"schema": "different", "version": 1},
            resuming=True,
            description="provenance",
        )


def test_zero_checkpoint_restart_accepts_metrics_and_replayable_plan_prefix(
    tmp_path: Path,
) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    rows = [
        canonical_json_bytes(
            {
                "schema": "simple-brats.long-run-step",
                "schema_version": 3,
                "step": step,
            }
        )
        for step in range(1, 4)
    ]
    (output / "metrics" / "start-000000000-stop-000005000-12-restart-0.jsonl").write_bytes(
        b"\n".join(rows) + b"\n"
    )
    for step in range(1, 4):
        for kind in ("plan", "prepared"):
            path = output / "plans" / f"step-{step:09d}.{kind}.json"
            value = {"step": step, "kind": kind}
            path.write_bytes(canonical_json_bytes(value))
            _write_or_require(
                path,
                value,
                resuming=True,
                description=f"step {step} {kind}",
            )

    _validate_zero_checkpoint_recovery(output)


def test_zero_checkpoint_restart_accepts_metrics_ahead_of_async_plan_prefix(
    tmp_path: Path,
) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    rows = [
        canonical_json_bytes(
            {
                "schema": "simple-brats.long-run-step",
                "schema_version": 3,
                "step": step,
            }
        )
        for step in range(1, 4)
    ]
    (output / "metrics" / "start-000000000-stop-000005000-12-restart-0.jsonl").write_bytes(
        b"\n".join(rows) + b"\n"
    )
    for kind in ("plan", "prepared"):
        (output / "plans" / f"step-{1:09d}.{kind}.json").write_bytes(b"{}")

    _validate_zero_checkpoint_recovery(output)


def test_zero_checkpoint_restart_tolerates_only_a_torn_final_metrics_line(
    tmp_path: Path,
) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    complete = canonical_json_bytes(
        {
            "schema": "simple-brats.long-run-step",
            "schema_version": 3,
            "step": 1,
        }
    )
    metrics = output / "metrics" / "start-000000000-stop-000005000-12-restart-0.jsonl"
    metrics.write_bytes(complete + b'\n{"schema":"simple-brats.long-run-step"')
    for kind in ("plan", "prepared"):
        (output / "plans" / f"step-{1:09d}.{kind}.json").write_bytes(b"{}")

    _validate_zero_checkpoint_recovery(output)

    metrics.write_bytes(b"{torn}\n" + complete + b"\n")
    with pytest.raises(LongRunError, match="invalid recovery metrics JSON"):
        _validate_zero_checkpoint_recovery(output)


def test_zero_checkpoint_restart_rejects_plans_beyond_checkpoint(tmp_path: Path) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    path = output / "plans" / "step-000001001.plan.json"
    path.write_bytes(b"{}")

    with pytest.raises(LongRunError, match="beyond first checkpoint"):
        _validate_zero_checkpoint_recovery(output)


def test_zero_checkpoint_restart_discards_stale_checkpoint_temporaries(tmp_path: Path) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    first = output / "checkpoints" / ".step-000001000.pt.tmp-123"
    second = output / "checkpoints" / ".step-000000500.pt.failure-tmp-123"
    first.write_bytes(b"partial")
    second.write_bytes(b"partial")

    _discard_stale_atomic_temporaries(output)
    _validate_zero_checkpoint_recovery(output)

    assert not first.exists()
    assert not second.exists()


def test_requeued_slurm_attempts_have_distinct_invocation_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    first = _safe_invocation_token()
    first_identity = _invocation_identity("a" * 64, start_step=0, stop_step=5_000)
    first_metrics = tmp_path / f"{first_identity.stem}.jsonl"
    first_metrics.write_text("prior attempt")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    second = _safe_invocation_token()
    second_identity = _invocation_identity("a" * 64, start_step=0, stop_step=5_000)

    assert first == "12345-restart-0"
    assert second == "12345-restart-1"
    assert first != second
    assert first_identity.stem != second_identity.stem
    assert first_identity.wandb_id != second_identity.wandb_id
    assert not (tmp_path / f"{second_identity.stem}.jsonl").exists()


def test_exact_resume_runtime_policy_is_cpu_verifiable_and_cuda_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_matmul_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        cpu_policy = configure_exact_resume_runtime(torch.device("cpu"))
        assert cpu_policy["torch_deterministic_algorithms"] is True
        assert cpu_policy["cublas_workspace_config"] == "not_applicable"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32

        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(LongRunError, match="CUBLAS_WORKSPACE_CONFIG"):
            configure_exact_resume_runtime(torch.device("cuda"))
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cuda_policy = configure_exact_resume_runtime(torch.device("cuda"))
        assert cuda_policy["cublas_workspace_config"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.set_float32_matmul_precision(previous_matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def test_terminal_recovery_stages_validated_checkpoint_as_wandb_artifact(
    tmp_path: Path,
) -> None:
    class RecordingSink:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, int, object]] = []

        def log_checkpoint(self, path: Path, *, step: int, metadata: object) -> None:
            self.calls.append((path, step, metadata))

    checkpoint = tmp_path / "step-000005000.pt"
    checkpoint.write_bytes(b"validated-by-runner")
    sink = RecordingSink()
    manager = CheckpointManager(
        tmp_path,
        policy=CheckpointPolicy(checkpoint_every_steps=1_000, artifact_every_steps=5_000),
        artifact_sink=sink,  # type: ignore[arg-type]
    )
    result = SimpleNamespace(
        start_step=5_000,
        end_step=5_000,
        total_steps=5_000,
        latest_checkpoint=checkpoint,
    )
    provenance = {"run": "terminal-recovery"}

    assert _log_terminal_recovery_artifact(manager, result, provenance)
    assert sink.calls == [(checkpoint, 5_000, provenance)]
