from __future__ import annotations

import copy

import pytest
import torch

from simple_brats.a40_resume_smoke import (
    A40ResumeSmokeError,
    _assert_registered_config,
    compare_smoke_outputs,
    semantic_digest,
)
from simple_brats.config import load_experiment_config


def _state(*, step: int = 2) -> dict[str, object]:
    return {
        "model": {
            "weight": torch.tensor([1.0, 2.0]),
            "target_teacher.num_updates": torch.tensor(step),
            "target_teacher.teacher.weight": torch.tensor([1.5]),
        },
        "optimizer": {"state": {0: {"step": torch.tensor(step)}}},
        "ema_update_count": step,
        "runner_contract": {"schema_version": 2},
        "runner_contract_sha256": "a" * 64,
        "rng": {"torch_cpu": torch.tensor([1, 2], dtype=torch.uint8)},
    }


def test_resume_gate_accepts_both_exact_registered_scale_arms() -> None:
    _assert_registered_config(load_experiment_config("configs/v0_cross_matching_small.toml"))
    _assert_registered_config(load_experiment_config("configs/v0_cross_matching_small_8mm.toml"))
    with pytest.raises(A40ResumeSmokeError, match="registered"):
        _assert_registered_config(load_experiment_config("configs/v0_cross_matching.toml"))


def _reports() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    metric1 = {"step": 1, "loss": 2.0}
    metric2 = {"step": 2, "loss": 1.0}
    common = {
        "calibration_sha256": "b" * 64,
        "real_batch_sha256": "c" * 64,
    }
    return (
        {
            **common,
            "pid": 10,
            "start_step": 0,
            "end_step": 2,
            "metrics": [metric1, metric2],
        },
        {
            **common,
            "pid": 11,
            "start_step": 0,
            "end_step": 1,
            "metrics": [metric1],
        },
        {
            **common,
            "pid": 12,
            "start_step": 1,
            "end_step": 2,
            "metrics": [metric2],
        },
    )


def test_semantic_digest_is_mapping_order_independent_and_tensor_exact() -> None:
    assert semantic_digest({"a": torch.tensor([1]), "b": 2}) == semantic_digest(
        {"b": 2, "a": torch.tensor([1])}
    )
    assert semantic_digest(torch.tensor([1.0])) != semantic_digest(torch.tensor([1.0001]))


def test_compare_smoke_outputs_accepts_exact_resume() -> None:
    step1 = _state(step=1)
    step2 = _state()
    continuous, first, resumed = _reports()
    result = compare_smoke_outputs(
        step1,
        copy.deepcopy(step1),
        step2,
        copy.deepcopy(step2),
        continuous_report=continuous,
        first_report=first,
        resumed_report=resumed,
    )
    assert all(result["checks"].values())


def test_compare_smoke_outputs_rejects_optimizer_drift() -> None:
    step1 = _state(step=1)
    state = _state()
    drifted = copy.deepcopy(state)
    drifted["optimizer"]["state"][0]["step"] = torch.tensor(3)  # type: ignore[index]
    continuous, first, resumed = _reports()
    try:
        compare_smoke_outputs(
            step1,
            copy.deepcopy(step1),
            state,
            drifted,
            continuous_report=continuous,
            first_report=first,
            resumed_report=resumed,
        )
    except RuntimeError as error:
        assert "final_optimizer" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("optimizer drift was accepted")
