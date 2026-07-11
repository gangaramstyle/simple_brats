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
from simple_brats.training.diagnostics import CollapseThresholds, RepresentationStats
from simple_brats.training.matching import build_matching_system, optimizer_parameter_groups
from simple_brats.training.runner import (
    PREDICTION_DIAGNOSTIC_STREAM,
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    RepresentationCollapseError,
    StepMetrics,
    TrainingRunnerError,
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
            count=2,
            variance=1.0,
            effective_rank=2.0,
            off_diagonal_cosine=0.0,
        )
        for modality_id in range(4)
    }


def _thresholds() -> CollapseThresholds:
    return CollapseThresholds(
        minimum_variance_ratio=0.1,
        minimum_effective_rank_ratio=0.5,
        maximum_off_diagonal_cosine=0.9,
    )


def _seed_training_rng() -> None:
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)


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
    assert payload["state"]["runner_contract_sha256"] == hashlib.sha256(
        encoded_contract
    ).hexdigest()
    assert contract["gradient_clipping"]["maximum_l2_norm"] is None
    assert contract["diagnostics"]["collapse_stream"] == TEACHER_TARGET_DIAGNOSTIC_STREAM
    assert set(contract["diagnostics"]["streams"]) == {
        TEACHER_TARGET_DIAGNOSTIC_STREAM,
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
            collapse_reference=_references(),
            collapse_thresholds=_thresholds(),
            collapse_warmup_steps=0,
        )

    error = caught.value
    assert error.step == 1
    assert set(error.reasons_by_modality) == {0, 1, 2, 3}
    assert all("variance_ratio" in reasons for reasons in error.reasons_by_modality.values())
    assert int(system.target_teacher.num_updates) == 1
    checkpoint = tmp_path / "step-000000001.pt"
    assert checkpoint.exists()
    assert error.checkpoint_path == checkpoint
    assert error.diagnostic_stream == TEACHER_TARGET_DIAGNOSTIC_STREAM
    assert torch.load(checkpoint, weights_only=False)["state"]["ema_update_count"] == 1


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
    ["gradient_clip_norm", "collapse_reference", "collapse_thresholds", "warmup"],
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
        collapse_reference=_references(),
        collapse_thresholds=_thresholds(),
        collapse_warmup_steps=2,
        gradient_clip_norm=1.0,
    )
    checkpoint = tmp_path / "first" / "step-000000001.pt"

    references = _references()
    thresholds = _thresholds()
    warmup = 2
    gradient_clip_norm = 1.0
    if changed_argument == "gradient_clip_norm":
        gradient_clip_norm = 2.0
    elif changed_argument == "collapse_reference":
        references[0] = replace(references[0], variance=2.0)
    elif changed_argument == "collapse_thresholds":
        thresholds = replace(thresholds, maximum_off_diagonal_cosine=0.8)
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
            collapse_reference=references,
            collapse_thresholds=thresholds,
            collapse_warmup_steps=warmup,
            gradient_clip_norm=gradient_clip_norm,
        )
