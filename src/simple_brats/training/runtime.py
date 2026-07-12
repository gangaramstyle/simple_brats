"""Pinned model-execution policy for optimized, exactly resumable training."""

from __future__ import annotations

import inspect
import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

_COMPILE_TARGETS = ("encoder.forward", "predictor.forward")


class TrainingRuntimeError(RuntimeError):
    """The requested model runtime cannot be established without silent drift."""


@dataclass(frozen=True, slots=True)
class TrainingRuntimePolicy:
    """Complete execution policy that becomes part of every resume contract."""

    device_type: str
    autocast_enabled: bool
    autocast_dtype: str
    grad_scaler_enabled: bool
    optimizer_fused: bool
    optimizer_foreach: bool
    optimizer_capturable: bool
    compile_enabled: bool
    compile_backend: str
    compile_mode: str
    compile_dynamic: bool
    compile_fullgraph: bool
    compile_targets: tuple[str, ...]
    fallback_policy: str
    torch_version: str

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("training runtime supports only CPU and CUDA")
        if self.grad_scaler_enabled:
            raise ValueError("the registered BF16 runtime must not use gradient scaling")
        if self.autocast_enabled:
            if self.device_type != "cuda" or self.autocast_dtype != "bfloat16":
                raise ValueError("autocast is registered only as CUDA bfloat16")
        elif self.autocast_dtype != "float32":
            raise ValueError("disabled autocast must declare float32 execution")
        if self.optimizer_fused and self.device_type != "cuda":
            raise ValueError("fused AdamW is registered only for CUDA")
        if self.optimizer_fused and self.optimizer_foreach:
            raise ValueError("fused and foreach AdamW modes are mutually exclusive")
        if self.compile_enabled:
            if self.device_type != "cuda":
                raise ValueError("model compilation is registered only for CUDA")
            if not self.compile_backend or not self.compile_mode:
                raise ValueError("compiled execution requires a backend and mode")
            if self.compile_targets != _COMPILE_TARGETS:
                raise ValueError("compiled execution targets changed from the registered paths")
        elif self.compile_targets:
            raise ValueError("eager execution must not declare compile targets")
        if not self.fallback_policy or not self.torch_version:
            raise ValueError("runtime fallback policy and Torch version must be recorded")

    @classmethod
    def eager_cpu(cls) -> TrainingRuntimePolicy:
        """Return the explicit test/development fallback; never used for CUDA launches."""

        return cls(
            device_type="cpu",
            autocast_enabled=False,
            autocast_dtype="float32",
            grad_scaler_enabled=False,
            optimizer_fused=False,
            optimizer_foreach=False,
            optimizer_capturable=False,
            compile_enabled=False,
            compile_backend="not_applicable",
            compile_mode="not_applicable",
            compile_dynamic=False,
            compile_fullgraph=False,
            compile_targets=(),
            fallback_policy="cpu_explicit_eager_only_cuda_never_silently_falls_back",
            torch_version=str(torch.__version__),
        )

    @classmethod
    def eager_for_device(cls, device: torch.device) -> TrainingRuntimePolicy:
        """Compatibility policy for generic runner callers outside registered long runs."""

        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if device.type == "cpu":
            return cls.eager_cpu()
        if device.type != "cuda":
            raise TrainingRuntimeError(f"unsupported eager device type: {device.type}")
        return cls(
            device_type="cuda",
            autocast_enabled=False,
            autocast_dtype="float32",
            grad_scaler_enabled=False,
            optimizer_fused=False,
            optimizer_foreach=False,
            optimizer_capturable=False,
            compile_enabled=False,
            compile_backend="not_applicable",
            compile_mode="not_applicable",
            compile_dynamic=False,
            compile_fullgraph=False,
            compile_targets=(),
            fallback_policy=(
                "generic_runner_explicit_eager_compatibility_registered_cuda_runs_pass_policy"
            ),
            torch_version=str(torch.__version__),
        )

    def require_device(self, device: torch.device) -> None:
        if not isinstance(device, torch.device):
            raise TypeError("device must be a torch.device")
        if device.type != self.device_type:
            raise TrainingRuntimeError(
                f"runtime policy is for {self.device_type}, but model is on {device.type}"
            )

    def autocast(self, device: torch.device) -> AbstractContextManager[Any]:
        """Return the exact forward context registered by this policy."""

        self.require_device(device)
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "simple-brats.training-runtime",
            "schema_version": 1,
            "device_type": self.device_type,
            "torch_version": self.torch_version,
            "autocast": {
                "enabled": self.autocast_enabled,
                "dtype": self.autocast_dtype,
                "gradient_scaler_enabled": self.grad_scaler_enabled,
            },
            "optimizer": {
                "name": "AdamW",
                "fused": self.optimizer_fused,
                "foreach": self.optimizer_foreach,
                "capturable": self.optimizer_capturable,
            },
            "compile": {
                "enabled": self.compile_enabled,
                "backend": self.compile_backend,
                "mode": self.compile_mode,
                "dynamic": self.compile_dynamic,
                "fullgraph": self.compile_fullgraph,
                "targets": list(self.compile_targets),
            },
            "fallback_policy": self.fallback_policy,
        }


def configure_training_runtime(device: torch.device) -> TrainingRuntimePolicy:
    """Resolve the registered policy, failing closed for an unoptimized CUDA runtime."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type == "cpu":
        return TrainingRuntimePolicy.eager_cpu()
    if device.type != "cuda":
        raise TrainingRuntimeError(f"unsupported training device type: {device.type}")
    if not torch.cuda.is_available():
        raise TrainingRuntimeError("CUDA training runtime requested while CUDA is unavailable")
    if not torch.amp.autocast_mode.is_autocast_available("cuda"):
        raise TrainingRuntimeError("CUDA autocast is unavailable")
    try:
        native_bfloat16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError as error:  # pragma: no cover - guarded by the pinned Torch release
        raise TrainingRuntimeError(
            "Torch cannot distinguish native from emulated CUDA bfloat16"
        ) from error
    if not native_bfloat16:
        raise TrainingRuntimeError("CUDA device lacks native bfloat16 support")
    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        raise TrainingRuntimeError("torch.compile is unavailable")
    compiler = getattr(torch, "compiler", None)
    list_backends = getattr(compiler, "list_backends", None)
    if not callable(list_backends) or "inductor" not in list_backends():
        raise TrainingRuntimeError("Torch inductor compile backend is unavailable")
    try:
        from torch import _dynamo
    except ImportError as error:  # pragma: no cover - torch.compile requires this package
        raise TrainingRuntimeError("Torch Dynamo is unavailable") from error
    if _dynamo.config.disable:
        raise TrainingRuntimeError("Torch Dynamo is disabled; compiled execution is mandatory")
    if _dynamo.config.suppress_errors:
        raise TrainingRuntimeError("Torch Dynamo suppress_errors would silently fall back to eager")
    try:
        from torch.utils._triton import has_triton, has_triton_package
    except ImportError as error:  # pragma: no cover - guarded by the pinned Torch release
        raise TrainingRuntimeError("Torch cannot validate its Triton runtime") from error
    if not has_triton_package() or not has_triton():
        raise TrainingRuntimeError(
            "Torch Inductor requires a working Triton installation on the CUDA device"
        )
    try:
        optimizer_parameters = inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError) as error:
        raise TrainingRuntimeError("cannot inspect AdamW fused capability") from error
    if "fused" not in optimizer_parameters:
        raise TrainingRuntimeError("Torch AdamW does not expose fused execution")
    return TrainingRuntimePolicy(
        device_type="cuda",
        autocast_enabled=True,
        autocast_dtype="bfloat16",
        grad_scaler_enabled=False,
        optimizer_fused=True,
        optimizer_foreach=False,
        optimizer_capturable=False,
        compile_enabled=True,
        compile_backend="inductor",
        compile_mode="default",
        compile_dynamic=False,
        compile_fullgraph=False,
        compile_targets=_COMPILE_TARGETS,
        fallback_policy="fail_closed_cuda_cpu_explicit_eager_only",
        torch_version=str(torch.__version__),
    )


def _module_device(module: nn.Module) -> torch.device:
    parameters = tuple(module.parameters())
    if not parameters:
        raise TrainingRuntimeError("training module has no parameters")
    devices = {parameter.device for parameter in parameters}
    devices.update(buffer.device for buffer in module.buffers())
    if len(devices) != 1:
        raise TrainingRuntimeError("training module parameters and buffers span devices")
    return parameters[0].device


def apply_model_runtime(module: nn.Module, policy: TrainingRuntimePolicy) -> None:
    """Compile registered forward paths without changing checkpoint parameter names."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    if not isinstance(policy, TrainingRuntimePolicy):
        raise TypeError("policy must be a TrainingRuntimePolicy")
    policy.require_device(_module_device(module))
    existing = getattr(module, "_simple_brats_compile_targets", None)
    if existing is not None:
        if existing != policy.compile_targets:
            raise TrainingRuntimeError("model already has a different compile policy")
        return
    if not policy.compile_enabled:
        module._simple_brats_compile_targets = ()  # type: ignore[attr-defined]
        return

    compile_function = getattr(torch, "compile", None)
    if not callable(compile_function):
        raise TrainingRuntimeError("torch.compile disappeared after runtime configuration")
    state_keys_before = tuple(module.state_dict())
    originals: list[tuple[nn.Module, Any]] = []
    try:
        for target in policy.compile_targets:
            owner_name, attribute = target.split(".", 1)
            owner = getattr(module, owner_name, None)
            if not isinstance(owner, nn.Module) or attribute != "forward":
                raise TrainingRuntimeError(f"compile target is unavailable: {target}")
            original = owner.forward
            compiled = compile_function(
                original,
                backend=policy.compile_backend,
                mode=policy.compile_mode,
                dynamic=policy.compile_dynamic,
                fullgraph=policy.compile_fullgraph,
            )
            if not callable(compiled):
                raise TrainingRuntimeError(f"torch.compile returned a non-callable for {target}")
            originals.append((owner, original))
            owner.forward = compiled  # type: ignore[method-assign]
    except Exception as error:
        for owner, original in originals:
            owner.forward = original  # type: ignore[method-assign]
        if isinstance(error, TrainingRuntimeError):
            raise
        raise TrainingRuntimeError("could not wrap registered model compile targets") from error
    if tuple(module.state_dict()) != state_keys_before:
        for owner, original in originals:
            owner.forward = original  # type: ignore[method-assign]
        raise TrainingRuntimeError("model compilation changed checkpoint state keys")
    module._simple_brats_compile_targets = policy.compile_targets  # type: ignore[attr-defined]


def build_adamw_optimizer(
    module: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    policy: TrainingRuntimePolicy,
) -> torch.optim.AdamW:
    """Construct AdamW with the implementation pinned by ``policy``."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    if not isinstance(policy, TrainingRuntimePolicy):
        raise TypeError("policy must be a TrainingRuntimePolicy")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    policy.require_device(_module_device(module))
    from .matching import optimizer_parameter_groups

    try:
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(module, weight_decay=weight_decay),
            lr=learning_rate,
            foreach=policy.optimizer_foreach,
            capturable=policy.optimizer_capturable,
            fused=policy.optimizer_fused,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingRuntimeError("could not construct the registered AdamW runtime") from error
    for key, expected in (
        ("foreach", policy.optimizer_foreach),
        ("capturable", policy.optimizer_capturable),
        ("fused", policy.optimizer_fused),
    ):
        if optimizer.defaults.get(key) is not expected:
            raise TrainingRuntimeError(f"AdamW did not retain registered {key}={expected}")
    return optimizer


__all__ = [
    "TrainingRuntimeError",
    "TrainingRuntimePolicy",
    "apply_model_runtime",
    "build_adamw_optimizer",
    "configure_training_runtime",
]
