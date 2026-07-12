from pathlib import Path


def test_every_slurm_job_drops_login_node_resource_limits() -> None:
    repository = Path(__file__).resolve().parents[1]
    scripts = sorted((repository / "slurm").glob("*.sbatch"))

    assert scripts
    for script in scripts:
        source = script.read_text()
        assert source.count("#SBATCH --propagate=NONE") == 1, script.name


def test_registered_training_and_evaluation_require_live_online_wandb() -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative in ("slurm/long_run.sbatch", "slurm/evaluate_checkpoint.sbatch"):
        source = (repository / relative).read_text()
        assert '"${WANDB_MODE}" != "online"' in source
        assert "WANDB_MODE=offline" not in source
        assert "export WANDB_MODE WANDB_PROJECT" in source
        assert "unset WANDB_ENTITY" in source

    for relative in (
        "cluster/prepare_and_submit_long_run.sh",
        "cluster/prepare_and_submit_checkpoint_evaluation.sh",
        "cluster/prepare_and_submit_wandb_online_smoke.sh",
    ):
        source = (repository / relative).read_text()
        assert '"${WANDB_MODE:=online}"' in source
        assert "login --verify" in source
        assert "--export=ALL" in source


def test_wandb_recovery_sync_includes_online_and_offline_transactions() -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative in (
        "cluster/sync_long_run_wandb.sh",
        "cluster/sync_evaluation_wandb.sh",
    ):
        source = (repository / relative).read_text()
        assert "--include-online" in source
        assert "--include-offline" in source
        assert "'offline-run-*'" in source
        assert "'run-*'" in source
