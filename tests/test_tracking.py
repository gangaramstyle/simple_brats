from types import SimpleNamespace

import pytest

from simple_brats.tracking import (
    OnlineWandbConfig,
    TrackingError,
    online_run_url,
    require_verified_online_login,
)


def test_online_wandb_config_defaults_and_inherits_nonsecret_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setenv("WANDB_ENTITY", "research-team")
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.example.test")

    config = OnlineWandbConfig.from_environment()

    assert config.project == "simple-brats"
    assert config.entity == "research-team"
    assert config.base_url == "https://wandb.example.test"
    assert config.init_kwargs() == {
        "project": "simple-brats",
        "entity": "research-team",
        "mode": "online",
        "force": True,
        "resume": "allow",
    }
    assert "api_key" not in config.to_dict()


def test_registered_tracking_rejects_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(TrackingError, match="requires WANDB_MODE=online"):
        OnlineWandbConfig.from_environment()


def test_verified_login_is_fail_closed() -> None:
    calls: list[dict[str, object]] = []

    def login(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    require_verified_online_login(SimpleNamespace(login=login))
    assert calls == [{"verify": True, "force": True}]

    with pytest.raises(TrackingError, match="did not authenticate"):
        require_verified_online_login(SimpleNamespace(login=lambda **_kwargs: False))


def test_online_run_requires_visible_url() -> None:
    assert online_run_url(SimpleNamespace(url="https://wandb.example/run/abc")) == (
        "https://wandb.example/run/abc"
    )
    with pytest.raises(TrackingError, match="no visible run URL"):
        online_run_url(SimpleNamespace(url=None))
