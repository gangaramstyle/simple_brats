from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import simple_brats.training.runtime as runtime_module
from simple_brats.config import ExperimentConfig, ModelConfig
from simple_brats.training import (
    TrainingRuntimeError,
    TrainingRuntimePolicy,
    apply_model_runtime,
    build_adamw_optimizer,
    build_matching_system,
    configure_training_runtime,
)


def _tiny_system() -> torch.nn.Module:
    return build_matching_system(
        ExperimentConfig(model=ModelConfig(width=12, depth=1, heads=3, mlp_ratio=1.0))
    )


def _mock_native_cuda(monkeypatch: pytest.MonkeyPatch) -> TrainingRuntimePolicy:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(torch.amp.autocast_mode, "is_autocast_available", lambda device: True)
    monkeypatch.setattr(torch.compiler, "list_backends", lambda: ["inductor"])
    return configure_training_runtime(torch.device("cuda"))


def test_cpu_runtime_is_explicit_eager_fallback() -> None:
    policy = configure_training_runtime(torch.device("cpu"))
    assert policy == TrainingRuntimePolicy.eager_cpu()
    assert policy.to_dict()["autocast"] == {
        "enabled": False,
        "dtype": "float32",
        "gradient_scaler_enabled": False,
    }
    assert policy.to_dict()["compile"]["enabled"] is False  # type: ignore[index]
    with policy.autocast(torch.device("cpu")):
        assert not torch.is_autocast_enabled()


def test_cuda_runtime_requires_native_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: False)
    monkeypatch.setattr(torch.amp.autocast_mode, "is_autocast_available", lambda device: True)
    with pytest.raises(TrainingRuntimeError, match="native bfloat16"):
        configure_training_runtime(torch.device("cuda"))


def test_cuda_runtime_records_strict_optimized_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _mock_native_cuda(monkeypatch)
    record = policy.to_dict()
    assert record["autocast"]["dtype"] == "bfloat16"  # type: ignore[index]
    assert record["optimizer"]["fused"] is True  # type: ignore[index]
    assert record["compile"] == {
        "enabled": True,
        "backend": "inductor",
        "mode": "default",
        "dynamic": False,
        "fullgraph": False,
        "targets": ["encoder.forward", "predictor.forward"],
    }
    assert "fail_closed" in str(record["fallback_policy"])


def test_cuda_runtime_rejects_silent_compile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from torch import _dynamo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(torch.amp.autocast_mode, "is_autocast_available", lambda device: True)
    monkeypatch.setattr(torch.compiler, "list_backends", lambda: ["inductor"])
    monkeypatch.setattr(_dynamo.config, "suppress_errors", True)
    with pytest.raises(TrainingRuntimeError, match="silently fall back"):
        configure_training_runtime(torch.device("cuda"))


def test_cuda_runtime_rejects_globally_disabled_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch import _dynamo

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kwargs: True)
    monkeypatch.setattr(torch.amp.autocast_mode, "is_autocast_available", lambda device: True)
    monkeypatch.setattr(torch.compiler, "list_backends", lambda: ["inductor"])
    monkeypatch.setattr(_dynamo.config, "disable", True)
    with pytest.raises(TrainingRuntimeError, match="disabled"):
        configure_training_runtime(torch.device("cuda"))


def test_compile_wrapping_preserves_checkpoint_keys_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _mock_native_cuda(monkeypatch)
    system = _tiny_system()
    keys_before = tuple(system.state_dict())
    calls: list[dict[str, object]] = []

    def fake_compile(function, **kwargs):
        calls.append(kwargs)

        def compiled(*args, **call_kwargs):
            return function(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(runtime_module, "_module_device", lambda module: torch.device("cuda"))
    monkeypatch.setattr(torch, "compile", fake_compile)

    apply_model_runtime(system, policy)
    apply_model_runtime(system, policy)

    assert tuple(system.state_dict()) == keys_before
    assert len(calls) == 2
    assert all(
        call
        == {
            "backend": "inductor",
            "mode": "default",
            "dynamic": False,
            "fullgraph": False,
        }
        for call in calls
    )


def test_optimizer_factory_pins_fused_cuda_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _mock_native_cuda(monkeypatch)
    system = _tiny_system()
    monkeypatch.setattr(runtime_module, "_module_device", lambda module: torch.device("cuda"))

    optimizer = build_adamw_optimizer(
        system,
        learning_rate=1e-4,
        weight_decay=0.05,
        policy=policy,
    )

    assert optimizer.defaults["fused"] is True
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["capturable"] is False


def test_runtime_policy_rejects_compile_target_drift() -> None:
    with pytest.raises(ValueError, match="targets changed"):
        replace(
            TrainingRuntimePolicy.eager_for_device(torch.device("cuda")),
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="default",
            compile_targets=("encoder.forward",),
        )
