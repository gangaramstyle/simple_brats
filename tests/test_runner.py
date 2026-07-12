import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from simple_brats.config import ExperimentConfig, ModelConfig
from simple_brats.training.checkpoints import CheckpointManager, CheckpointPolicy
from simple_brats.training.diagnostics import (
    CollapseThresholds,
    RepresentationStats,
    stats_by_modality,
)
from simple_brats.training.matching import build_matching_system, optimizer_parameter_groups
from simple_brats.training.runner import (
    PREDICTION_DIAGNOSTIC_STREAM,
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
    FixedTargetPatchProbe,
    RepresentationCollapseError,
    StepMetrics,
    TrainingRunnerError,
    TrainingRuntimePolicy,
    _restore_rng_state,
    run_matching_training,
)
from simple_brats.training.synthetic import make_synthetic_matching_batch


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=71,
        model=ModelConfig(width=12, depth=1, heads=3, mlp_ratio=1.0),
    )


def _optimizer(system):
    return torch.optim.AdamW(
        optimizer_parameter_groups(system, weight_decay=0.01),
        lr=3e-4,
    )


def _references() -> dict[int, RepresentationStats]:
    return {
        modality_id: RepresentationStats(
            count=32,
            variance=1.0,
            effective_rank=2.0,
            off_diagonal_cosine=0.0,
        )
        for modality_id in (0,)
    }


def _thresholds() -> CollapseThresholds:
    return CollapseThresholds(
        minimum_variance_ratio=0.1,
        minimum_effective_rank_ratio=0.5,
        maximum_off_diagonal_cosine=0.9,
    )


def _probe(batch) -> FixedTargetPatchProbe:
    return FixedTargetPatchProbe(batch.target_patches, batch.target_modality_ids)


def _probe_references(system, probe: FixedTargetPatchProbe) -> dict[int, RepresentationStats]:
    with torch.no_grad():
        targets = system.target_teacher(probe.target_patches)
    return stats_by_modality(targets, probe.target_modality_ids)


def test_single_d_training_diagnostics_accept_all_modality_fixed_probe(tmp_path: Path) -> None:
    config = _tiny_config()
    training_batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    probe_batch, _ = make_synthetic_matching_batch(config, batch_size=4, positions=9)
    probe = FixedTargetPatchProbe(
        probe_batch.target_patches,
        probe_batch.target_modality_ids,
    )
    torch.manual_seed(899)
    system = build_matching_system(config)

    result = run_matching_training(
        system,
        _optimizer(system),
        [training_batch],
        _manager(tmp_path / "single-d-all-modality-probe"),
        {"run": "single-d-all-modality-probe"},
        total_steps=1,
        collapse_probe=probe,
        collapse_reference=_probe_references(system, probe),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=1,
    )

    assert result.last_metrics is not None
    assert set(result.last_metrics.teacher_target_diagnostics_by_modality) == {0, 1, 2, 3}
    assert set(result.last_metrics.training_teacher_target_diagnostics_by_modality) == {0}
    assert set(result.last_metrics.prediction_diagnostics_by_modality) == {0}


def _seed_training_rng() -> None:
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)


def test_restore_rng_state_normalizes_cuda_states_to_detached_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_state = torch.arange(32, dtype=torch.uint8)
    restored: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.append(states))

    _restore_rng_state(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [cuda_state],
        }
    )

    assert len(restored) == 1
    assert len(restored[0]) == 1
    assert restored[0][0] is not cuda_state
    assert restored[0][0].device.type == "cpu"
    assert restored[0][0].dtype == torch.uint8
    assert restored[0][0].is_contiguous()
    torch.testing.assert_close(restored[0][0], cuda_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cuda_state",
    [torch.ones(8), torch.ones((2, 4), dtype=torch.uint8), ["not-a-tensor"]],
)
def test_restore_rng_state_rejects_malformed_cuda_states(
    cuda_state: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(TrainingRunnerError, match="CUDA RNG state"):
        _restore_rng_state(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": cuda_state,
            }
        )


class RandomBatchFactory:
    """Use global Torch RNG so the test exercises checkpoint RNG restoration."""

    def __init__(self, base_batch) -> None:
        self.base_batch = base_batch
        self.requested_indices: list[int] = []

    def __call__(self, absolute_step_index: int):
        self.requested_indices.append(absolute_step_index)
        return replace(
            self.base_batch,
            source_patches=self.base_batch.source_patches
            + 0.01 * torch.randn_like(self.base_batch.source_patches),
            target_patches=self.base_batch.target_patches
            + 0.01 * torch.randn_like(self.base_batch.target_patches),
        )


class StatefulZeroArgumentBatchFactory:
    def __init__(self, base_batch) -> None:
        self.base_batch = base_batch
        self.cursor = 0
        self.loaded_states: list[object] = []

    def __call__(self):
        self.cursor += 1
        return self.base_batch

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state) -> None:
        self.loaded_states.append(state)
        self.cursor = state["cursor"]


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _manager(root: Path) -> CheckpointManager:
    return CheckpointManager(
        root,
        policy=CheckpointPolicy(checkpoint_every_steps=2, artifact_every_steps=100),
        artifact_sink=None,
    )


def test_resume_is_bit_exact_and_uses_absolute_next_batch(tmp_path) -> None:
    config = _tiny_config()
    base_batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(900)
    initial = build_matching_system(config)
    full_system = copy.deepcopy(initial)
    partial_system = copy.deepcopy(initial)
    provenance = {
        "git_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "patch_plan_sha256": "c" * 64,
    }

    full_metrics: list[StepMetrics] = []
    full_batches = RandomBatchFactory(base_batch)
    _seed_training_rng()
    full_optimizer = _optimizer(full_system)
    full_result = run_matching_training(
        full_system,
        full_optimizer,
        full_batches,
        _manager(tmp_path / "full"),
        provenance,
        total_steps=4,
        collapse_probe=_probe(base_batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=4,
        on_step=full_metrics.append,
    )
    full_rng_continuation = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
    )

    partial_metrics: list[StepMetrics] = []
    partial_batches = RandomBatchFactory(base_batch)
    _seed_training_rng()
    partial_optimizer = _optimizer(partial_system)
    partial_result = run_matching_training(
        partial_system,
        partial_optimizer,
        partial_batches,
        _manager(tmp_path / "partial"),
        provenance,
        total_steps=4,
        max_steps=2,
        collapse_probe=_probe(base_batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=4,
        on_step=partial_metrics.append,
    )
    assert partial_result.end_step == 2
    checkpoint = tmp_path / "partial" / "step-000000002.pt"
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["metadata"] == provenance
    assert payload["state"]["step"] == 2
    assert payload["state"]["ema_update_count"] == 2
    contract = payload["state"]["runner_contract"]
    encoded_contract = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert (
        payload["state"]["runner_contract_sha256"] == hashlib.sha256(encoded_contract).hexdigest()
    )
    assert contract["gradient_clipping"]["maximum_l2_norm"] is None
    assert contract["diagnostics"]["collapse_stream"] == TEACHER_TARGET_DIAGNOSTIC_STREAM
    assert contract["fixed_target_patch_probe"]["sha256"] == _probe(base_batch).sha256
    assert contract["training_runtime"] == TrainingRuntimePolicy.eager_cpu().to_dict()
    assert contract["step_callback_rng_policy"] == (
        "capture_and_restore_python_numpy_torch_cpu_and_all_cuda_generators"
    )
    assert contract["diagnostic_cadence"] == {
        "first_completed_step": True,
        "every_completed_steps": 50,
        "checkpoint_steps": True,
        "invocation_final_step": True,
        "stop_requested_step": True,
        "training_batch_streams_share_cadence": True,
    }
    assert set(contract["diagnostics"]["streams"]) == {
        TEACHER_TARGET_DIAGNOSTIC_STREAM,
        TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
        PREDICTION_DIAGNOSTIC_STREAM,
    }
    assert set(payload["state"]["rng"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }

    resumed_system = copy.deepcopy(initial)
    resumed_optimizer = _optimizer(resumed_system)
    resumed_batches = RandomBatchFactory(base_batch)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    resumed_result = run_matching_training(
        resumed_system,
        resumed_optimizer,
        resumed_batches,
        _manager(tmp_path / "resumed"),
        provenance,
        total_steps=4,
        resume_from=checkpoint,
        collapse_probe=_probe(base_batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=4,
        on_step=partial_metrics.append,
    )
    resumed_rng_continuation = (
        random.random(),
        float(np.random.random()),
        torch.rand(5),
    )

    assert full_batches.requested_indices == [0, 1, 2, 3]
    assert partial_batches.requested_indices == [0, 1]
    assert resumed_batches.requested_indices == [2, 3]
    assert resumed_result.start_step == 2
    assert resumed_result.end_step == full_result.end_step == 4
    assert resumed_result.ema_update_count == full_result.ema_update_count == 4
    assert resumed_result.runner_contract_sha256 == full_result.runner_contract_sha256
    assert full_result.last_metrics is not None
    assert set(full_result.last_metrics.diagnostics_by_stream) == {
        TEACHER_TARGET_DIAGNOSTIC_STREAM,
        TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
        PREDICTION_DIAGNOSTIC_STREAM,
    }
    assert (
        full_result.last_metrics.diagnostics_by_modality
        is full_result.last_metrics.teacher_target_diagnostics_by_modality
    )
    assert [metric.loss for metric in partial_metrics] == pytest.approx(
        [metric.loss for metric in full_metrics], rel=0, abs=0
    )
    _assert_nested_equal(full_system.state_dict(), resumed_system.state_dict())
    _assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert resumed_rng_continuation[:2] == full_rng_continuation[:2]
    torch.testing.assert_close(
        resumed_rng_continuation[2], full_rng_continuation[2], rtol=0, atol=0
    )


def test_step_callback_cannot_advance_training_rng_streams(tmp_path: Path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(1919)
    initial = build_matching_system(config)

    def execute(root: Path, callback=None):
        system = copy.deepcopy(initial)
        _seed_training_rng()
        run_matching_training(
            system,
            _optimizer(system),
            [batch],
            _manager(root),
            {"run": "callback-rng-isolation"},
            total_steps=1,
            collapse_probe=_probe(batch),
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=1,
            on_step=callback,
        )
        continuation = (random.random(), float(np.random.random()), torch.rand(5))
        return system.state_dict(), continuation

    baseline_state, baseline_rng = execute(tmp_path / "baseline")

    def consuming_callback(_metrics: StepMetrics) -> None:
        random.random()
        np.random.random()
        torch.rand(17)

    observed_state, observed_rng = execute(tmp_path / "consuming", consuming_callback)

    _assert_nested_equal(observed_state, baseline_state)
    assert observed_rng[:2] == baseline_rng[:2]
    torch.testing.assert_close(observed_rng[2], baseline_rng[2], rtol=0, atol=0)


def test_stop_request_forces_an_exact_off_cadence_resume_checkpoint(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(1900)
    initial = build_matching_system(config)
    system = copy.deepcopy(initial)
    metrics: list[StepMetrics] = []
    provenance = {"run": "walltime-stop"}

    result = run_matching_training(
        system,
        _optimizer(system),
        [batch, batch],
        _manager(tmp_path / "interrupted"),
        provenance,
        total_steps=2,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
        on_step=metrics.append,
        should_stop=lambda: True,
    )

    checkpoint = tmp_path / "interrupted" / "step-000000001.pt"
    assert result.end_step == 1
    assert result.latest_checkpoint == checkpoint
    assert metrics[0].diagnostics_measured
    assert torch.load(checkpoint, weights_only=False)["state"]["step"] == 1

    resumed = copy.deepcopy(initial)
    resumed_result = run_matching_training(
        resumed,
        _optimizer(resumed),
        [batch, batch],
        _manager(tmp_path / "resumed-walltime"),
        provenance,
        total_steps=2,
        resume_from=checkpoint,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )
    assert resumed_result.start_step == 1
    assert resumed_result.end_step == 2


def test_terminal_checkpoint_resume_validates_state_without_optimizer_step(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(1910)
    initial = build_matching_system(config)
    provenance = {"run": "terminal-finalize"}
    trained = copy.deepcopy(initial)
    trained_optimizer = _optimizer(trained)
    manager = CheckpointManager(
        tmp_path / "trained",
        policy=CheckpointPolicy(checkpoint_every_steps=2, artifact_every_steps=100),
        artifact_sink=None,
    )
    run_matching_training(
        trained,
        trained_optimizer,
        [batch, batch],
        manager,
        provenance,
        total_steps=2,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )
    checkpoint = tmp_path / "trained" / "step-000000002.pt"

    finalized = copy.deepcopy(initial)
    optimizer = _optimizer(finalized)
    batches = RandomBatchFactory(batch)
    result = run_matching_training(
        finalized,
        optimizer,
        batches,
        _manager(tmp_path / "finalized"),
        provenance,
        total_steps=2,
        max_steps=0,
        resume_from=checkpoint,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )

    assert result.start_step == result.end_step == result.total_steps == 2
    assert result.latest_checkpoint == checkpoint
    assert result.ema_update_count == 2
    assert batches.requested_indices == []
    _assert_nested_equal(finalized.state_dict(), trained.state_dict())
    _assert_nested_equal(optimizer.state_dict(), trained_optimizer.state_dict())


def test_artifact_upload_failure_resumes_before_next_batch_and_remains_bit_exact(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    base_batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(1920)
    initial = build_matching_system(config)
    provenance = {"run": "artifact-receipt-resume"}
    policy = CheckpointPolicy(checkpoint_every_steps=2, artifact_every_steps=2)

    class RngConsumingSink:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.calls: list[int] = []

        def log_checkpoint(self, _path, *, step: int, metadata) -> dict[str, object]:
            assert metadata == provenance
            self.calls.append(step)
            random.random()
            np.random.random()
            torch.rand(13)
            if self.fail:
                raise RuntimeError("transient artifact upload failure")
            return {"backend": "test", "completion": "confirmed", "step": step}

    full_system = copy.deepcopy(initial)
    full_optimizer = _optimizer(full_system)
    full_batches = RandomBatchFactory(base_batch)
    full_sink = RngConsumingSink()
    full_metrics: list[StepMetrics] = []
    _seed_training_rng()
    full_result = run_matching_training(
        full_system,
        full_optimizer,
        full_batches,
        CheckpointManager(tmp_path / "full", policy=policy, artifact_sink=full_sink),
        provenance,
        total_steps=4,
        collapse_probe=_probe(base_batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=4,
        on_step=full_metrics.append,
    )
    full_continuation = (random.random(), float(np.random.random()), torch.rand(5))

    failed_system = copy.deepcopy(initial)
    failed_batches = RandomBatchFactory(base_batch)
    failed_sink = RngConsumingSink(fail=True)
    interrupted_root = tmp_path / "interrupted"
    _seed_training_rng()
    with pytest.raises(RuntimeError, match="transient artifact upload failure"):
        run_matching_training(
            failed_system,
            _optimizer(failed_system),
            failed_batches,
            CheckpointManager(interrupted_root, policy=policy, artifact_sink=failed_sink),
            provenance,
            total_steps=4,
            max_steps=2,
            collapse_probe=_probe(base_batch),
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=4,
        )
    checkpoint = interrupted_root / "step-000000002.pt"
    receipt = CheckpointManager.artifact_receipt_path(checkpoint)
    assert checkpoint.is_file() and not receipt.exists()
    assert failed_batches.requested_indices == [0, 1]

    resumed_system = copy.deepcopy(initial)
    resumed_optimizer = _optimizer(resumed_system)
    resumed_batches = RandomBatchFactory(base_batch)
    resumed_sink = RngConsumingSink()
    resumed_metrics: list[StepMetrics] = []
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    resumed_result = run_matching_training(
        resumed_system,
        resumed_optimizer,
        resumed_batches,
        CheckpointManager(interrupted_root, policy=policy, artifact_sink=resumed_sink),
        provenance,
        total_steps=4,
        resume_from=checkpoint,
        collapse_probe=_probe(base_batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=4,
        on_step=resumed_metrics.append,
    )
    resumed_continuation = (random.random(), float(np.random.random()), torch.rand(5))

    assert resumed_sink.calls == [2, 4]
    assert receipt.is_file()
    assert resumed_batches.requested_indices == [2, 3]
    assert resumed_result.end_step == full_result.end_step == 4
    assert [metric.loss for metric in resumed_metrics] == pytest.approx(
        [metric.loss for metric in full_metrics[2:]], rel=0, abs=0
    )
    _assert_nested_equal(resumed_system.state_dict(), full_system.state_dict())
    _assert_nested_equal(resumed_optimizer.state_dict(), full_optimizer.state_dict())
    assert resumed_continuation[:2] == full_continuation[:2]
    torch.testing.assert_close(resumed_continuation[2], full_continuation[2], rtol=0, atol=0)


def test_terminal_artifact_failure_is_repaired_without_an_optimizer_step(
    tmp_path: Path,
) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(1921)
    initial = build_matching_system(config)
    provenance = {"run": "terminal-artifact-receipt"}
    policy = CheckpointPolicy(checkpoint_every_steps=2, artifact_every_steps=2)
    root = tmp_path / "terminal"

    class Sink:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail
            self.calls: list[int] = []

        def log_checkpoint(self, _path, *, step: int, metadata) -> dict[str, object]:
            assert metadata == provenance
            self.calls.append(step)
            if self.fail:
                raise RuntimeError("terminal upload failed")
            return {"backend": "test", "completion": "confirmed"}

    trained = copy.deepcopy(initial)
    trained_optimizer = _optimizer(trained)
    _seed_training_rng()
    with pytest.raises(RuntimeError, match="terminal upload failed"):
        run_matching_training(
            trained,
            trained_optimizer,
            [batch, batch],
            CheckpointManager(root, policy=policy, artifact_sink=Sink(fail=True)),
            provenance,
            total_steps=2,
            collapse_probe=_probe(batch),
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=2,
        )
    checkpoint = root / "step-000000002.pt"
    assert checkpoint.is_file()

    finalized = copy.deepcopy(initial)
    finalized_optimizer = _optimizer(finalized)
    finalized_batches = RandomBatchFactory(batch)
    recovered_sink = Sink(fail=False)
    result = run_matching_training(
        finalized,
        finalized_optimizer,
        finalized_batches,
        CheckpointManager(root, policy=policy, artifact_sink=recovered_sink),
        provenance,
        total_steps=2,
        max_steps=0,
        resume_from=checkpoint,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )

    assert recovered_sink.calls == [2]
    assert CheckpointManager.artifact_receipt_path(checkpoint).is_file()
    assert finalized_batches.requested_indices == []
    assert result.start_step == result.end_step == result.total_steps == 2
    _assert_nested_equal(finalized.state_dict(), trained.state_dict())
    _assert_nested_equal(finalized_optimizer.state_dict(), trained_optimizer.state_dict())


def test_collapse_aborts_by_modality_after_one_optimizer_and_ema_step(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    collapsed_batch = replace(batch, target_patches=torch.zeros_like(batch.target_patches))
    torch.manual_seed(901)
    system = build_matching_system(config)
    optimizer = _optimizer(system)
    manager = CheckpointManager(
        tmp_path,
        policy=CheckpointPolicy(checkpoint_every_steps=100, artifact_every_steps=100),
        artifact_sink=None,
    )

    with pytest.raises(RepresentationCollapseError) as caught:
        run_matching_training(
            system,
            optimizer,
            [collapsed_batch],
            manager,
            {"run": "collapse-test"},
            total_steps=1,
            collapse_probe=_probe(collapsed_batch),
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=0,
        )

    error = caught.value
    assert error.step == 1
    assert set(error.reasons_by_modality) == {0}
    assert all("variance_ratio" in reasons for reasons in error.reasons_by_modality.values())
    assert int(system.target_teacher.num_updates) == 1
    checkpoint = tmp_path / "step-000000001.pt"
    assert checkpoint.exists()
    assert error.checkpoint_path == checkpoint
    assert error.diagnostic_stream == TEACHER_TARGET_DIAGNOSTIC_STREAM
    assert torch.load(checkpoint, weights_only=False)["state"]["ema_update_count"] == 1


def test_homogeneous_training_batch_is_logging_only_for_collapse(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    homogeneous_batch = replace(batch, target_patches=torch.zeros_like(batch.target_patches))
    torch.manual_seed(1901)
    system = build_matching_system(config)
    probe = _probe(batch)
    references = _probe_references(system, probe)
    metrics: list[StepMetrics] = []

    result = run_matching_training(
        system,
        _optimizer(system),
        [homogeneous_batch],
        _manager(tmp_path),
        {"run": "homogeneous-training-batch"},
        total_steps=1,
        collapse_probe=probe,
        collapse_reference=references,
        collapse_thresholds=CollapseThresholds(
            minimum_variance_ratio=1e-6,
            minimum_effective_rank_ratio=1e-6,
            maximum_off_diagonal_cosine=0.999999,
        ),
        collapse_warmup_steps=0,
        on_step=metrics.append,
    )

    assert result.end_step == 1
    assert len(metrics) == 1
    stochastic = metrics[0].diagnostics_by_stream[TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM]
    fixed = metrics[0].diagnostics_by_stream[TEACHER_TARGET_DIAGNOSTIC_STREAM]
    assert all(stats.variance == 0 for stats in stochastic.values())
    assert all(stats.variance > 0 for stats in fixed.values())


def test_resume_rejects_stateless_zero_argument_factory(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(902)
    initial = build_matching_system(config)
    provenance = {"run": "zero-argument-resume"}
    first_system = copy.deepcopy(initial)
    run_matching_training(
        first_system,
        _optimizer(first_system),
        [batch],
        CheckpointManager(
            tmp_path / "first",
            policy=CheckpointPolicy(checkpoint_every_steps=1, artifact_every_steps=100),
            artifact_sink=None,
        ),
        provenance,
        total_steps=2,
        max_steps=1,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )
    checkpoint = tmp_path / "first" / "step-000000001.pt"

    resumed_system = copy.deepcopy(initial)
    with pytest.raises(
        TrainingRunnerError,
        match="zero-argument batch factory requires checkpointed",
    ):
        run_matching_training(
            resumed_system,
            _optimizer(resumed_system),
            lambda: batch,
            _manager(tmp_path / "resumed"),
            provenance,
            total_steps=2,
            resume_from=checkpoint,
            collapse_probe=_probe(batch),
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=2,
        )


def test_resume_loads_state_before_using_zero_argument_factory(tmp_path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(903)
    initial = build_matching_system(config)
    provenance = {"run": "stateful-zero-argument-resume"}

    first_system = copy.deepcopy(initial)
    first_source = StatefulZeroArgumentBatchFactory(batch)
    run_matching_training(
        first_system,
        _optimizer(first_system),
        first_source,
        CheckpointManager(
            tmp_path / "first",
            policy=CheckpointPolicy(checkpoint_every_steps=1, artifact_every_steps=100),
            artifact_sink=None,
        ),
        provenance,
        total_steps=2,
        max_steps=1,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )
    checkpoint = tmp_path / "first" / "step-000000001.pt"

    resumed_system = copy.deepcopy(initial)
    resumed_source = StatefulZeroArgumentBatchFactory(batch)
    result = run_matching_training(
        resumed_system,
        _optimizer(resumed_system),
        resumed_source,
        _manager(tmp_path / "resumed"),
        provenance,
        total_steps=2,
        resume_from=checkpoint,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
    )

    assert resumed_source.loaded_states == [{"cursor": 1}]
    assert resumed_source.cursor == 2
    assert result.start_step == 1
    assert result.end_step == 2


@pytest.mark.parametrize(
    "changed_argument",
    [
        "gradient_clip_norm",
        "collapse_probe",
        "collapse_reference",
        "collapse_thresholds",
        "warmup",
        "runtime_policy",
    ],
)
def test_resume_rejects_changed_runner_contract(tmp_path, changed_argument) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(904)
    initial = build_matching_system(config)
    provenance = {"run": f"contract-{changed_argument}"}
    first_system = copy.deepcopy(initial)
    run_matching_training(
        first_system,
        _optimizer(first_system),
        [batch],
        CheckpointManager(
            tmp_path / "first",
            policy=CheckpointPolicy(checkpoint_every_steps=1, artifact_every_steps=100),
            artifact_sink=None,
        ),
        provenance,
        total_steps=2,
        max_steps=1,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
        gradient_clip_norm=1.0,
    )
    checkpoint = tmp_path / "first" / "step-000000001.pt"

    references = _references()
    probe = _probe(batch)
    thresholds = _thresholds()
    warmup = 2
    gradient_clip_norm = 1.0
    runtime_policy = TrainingRuntimePolicy.eager_cpu()
    if changed_argument == "gradient_clip_norm":
        gradient_clip_norm = 2.0
    elif changed_argument == "collapse_probe":
        changed_patches = probe.target_patches.clone()
        changed_patches.reshape(-1)[0] += 1.0
        probe = FixedTargetPatchProbe(changed_patches, probe.target_modality_ids)
    elif changed_argument == "collapse_reference":
        references[0] = replace(references[0], variance=2.0)
    elif changed_argument == "collapse_thresholds":
        thresholds = replace(thresholds, maximum_off_diagonal_cosine=0.8)
    elif changed_argument == "runtime_policy":
        runtime_policy = replace(runtime_policy, fallback_policy="changed-for-resume-test")
    else:
        warmup = 3

    resumed_system = copy.deepcopy(initial)
    with pytest.raises(TrainingRunnerError, match="runner contract does not exactly match"):
        run_matching_training(
            resumed_system,
            _optimizer(resumed_system),
            [batch, batch],
            _manager(tmp_path / "resumed"),
            provenance,
            total_steps=2,
            resume_from=checkpoint,
            collapse_probe=probe,
            collapse_reference=references,
            collapse_thresholds=thresholds,
            collapse_warmup_steps=warmup,
            gradient_clip_norm=gradient_clip_norm,
            runtime_policy=runtime_policy,
        )


def test_diagnostics_use_registered_sparse_cadence_and_final_step(tmp_path: Path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(905)
    system = build_matching_system(config)
    metrics: list[StepMetrics] = []

    run_matching_training(
        system,
        _optimizer(system),
        [batch] * 51,
        CheckpointManager(
            tmp_path,
            policy=CheckpointPolicy(checkpoint_every_steps=100, artifact_every_steps=100),
            artifact_sink=None,
        ),
        {"run": "sparse-diagnostics"},
        total_steps=51,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=51,
        on_step=metrics.append,
    )

    assert [metric.step for metric in metrics if metric.diagnostics_measured] == [1, 50, 51]
    assert all(not metrics[index].diagnostics_by_stream for index in range(1, 49))
    assert metrics[1].diagnostics_by_modality == {}
    assert not list(tmp_path.glob("*.pt"))


def test_noncheckpoint_steps_do_not_build_checkpoint_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simple_brats.training.runner as runner_module

    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)
    torch.manual_seed(906)
    system = build_matching_system(config)

    def forbidden_checkpoint_state(**_kwargs):
        raise AssertionError("checkpoint state built off cadence")

    monkeypatch.setattr(runner_module, "_checkpoint_state", forbidden_checkpoint_state)
    result = run_matching_training(
        system,
        _optimizer(system),
        [batch, batch, batch],
        CheckpointManager(
            tmp_path,
            policy=CheckpointPolicy(checkpoint_every_steps=100, artifact_every_steps=100),
            artifact_sink=None,
        ),
        {"run": "no-off-cadence-state"},
        total_steps=3,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=3,
    )

    assert result.end_step == 3
    assert result.latest_checkpoint is None


def test_checkpoint_flushes_batch_plan_artifacts_before_publication(tmp_path: Path) -> None:
    config = _tiny_config()
    batch, _ = make_synthetic_matching_batch(config, batch_size=1, positions=8)

    class FlushTrackingSource:
        def __init__(self) -> None:
            self.flush_count = 0

        def __call__(self, _absolute_step_index: int):
            return batch

        def flush_plan_artifacts(self) -> None:
            self.flush_count += 1

    source = FlushTrackingSource()
    torch.manual_seed(907)
    system = build_matching_system(config)
    result = run_matching_training(
        system,
        _optimizer(system),
        source,
        CheckpointManager(
            tmp_path,
            policy=CheckpointPolicy(checkpoint_every_steps=1, artifact_every_steps=100),
            artifact_sink=None,
        ),
        {"run": "flush-before-checkpoint"},
        total_steps=1,
        collapse_probe=_probe(batch),
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=1,
    )

    assert source.flush_count == 1
    assert result.latest_checkpoint == tmp_path / "step-000000001.pt"
