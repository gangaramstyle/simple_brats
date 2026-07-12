"""Fail-closed online W&B transport configuration for registered cluster jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class TrackingError(RuntimeError):
    """The requested experiment-tracking transport is unavailable or unsafe."""


@dataclass(frozen=True, slots=True)
class OnlineWandbConfig:
    """Non-secret online W&B settings inherited by login and compute nodes."""

    project: str
    entity: str | None
    base_url: str | None
    mode: str = "online"

    def __post_init__(self) -> None:
        if self.mode != "online":
            raise ValueError("registered W&B transport must be online")
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError("W&B project must be a non-empty string")
        if self.project != self.project.strip():
            raise ValueError("W&B project must not have surrounding whitespace")
        for value, name in ((self.entity, "entity"), (self.base_url, "base_url")):
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"W&B {name} must be a non-empty trimmed string when supplied")

    @classmethod
    def from_environment(cls) -> OnlineWandbConfig:
        mode = os.environ.get("WANDB_MODE", "online")
        if mode != "online":
            raise TrackingError(
                f"registered cluster tracking requires WANDB_MODE=online, observed {mode!r}"
            )
        project = os.environ.get("WANDB_PROJECT", "simple-brats")
        entity = os.environ.get("WANDB_ENTITY") or None
        base_url = os.environ.get("WANDB_BASE_URL") or None
        try:
            return cls(project=project, entity=entity, base_url=base_url)
        except ValueError as error:
            raise TrackingError(str(error)) from error

    def init_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "project": self.project,
            "mode": self.mode,
            "force": True,
            "resume": "allow",
        }
        if self.entity is not None:
            values["entity"] = self.entity
        return values

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "base_url": self.base_url or "sdk_default",
            "credential_source": "wandb_environment_settings_or_shared_home_never_serialized",
            "resume_policy": "allow_for_deterministic_run_id",
        }


def require_verified_online_login(wandb_module: Any) -> None:
    """Verify credentials and server reachability without exposing an API key."""

    login = getattr(wandb_module, "login", None)
    if not callable(login):
        raise TrackingError("pinned W&B module does not expose callable login")
    try:
        authenticated = login(verify=True, force=True)
    except Exception as error:
        raise TrackingError("W&B credential/server verification failed") from error
    if not authenticated:
        raise TrackingError("W&B credential/server verification did not authenticate")


def online_run_url(run: Any) -> str:
    """Require the server URL that proves an online run was created."""

    value = getattr(run, "url", None)
    if not isinstance(value, str) or not value.strip():
        raise TrackingError("online W&B initialization returned no visible run URL")
    return value.strip()


__all__ = [
    "OnlineWandbConfig",
    "TrackingError",
    "online_run_url",
    "require_verified_online_login",
]
