from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simple_brats.data.manifest import CaseRecord, FileRecord, canonical_json_bytes
from simple_brats.long_run import (
    DEFAULT_BAGS_PER_SUBJECT,
    LongRunError,
    SubjectBalancedSchedule,
    _discard_stale_atomic_temporaries,
    _initialize_destination,
    _invocation_identity,
    _log_terminal_recovery_artifact,
    _probe_cases,
    _safe_invocation_token,
    _validate_zero_checkpoint_recovery,
    _write_or_require,
    configure_exact_resume_runtime,
)
from simple_brats.short_run import ShortRunError
from simple_brats.training import CheckpointManager, CheckpointPolicy


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
                "schema_version": 2,
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


def test_zero_checkpoint_restart_tolerates_only_a_torn_final_metrics_line(
    tmp_path: Path,
) -> None:
    output, _, _ = _initialize_destination(tmp_path / "long", None)
    complete = canonical_json_bytes(
        {
            "schema": "simple-brats.long-run-step",
            "schema_version": 2,
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
