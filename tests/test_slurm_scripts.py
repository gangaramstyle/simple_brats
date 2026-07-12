from pathlib import Path


def test_every_slurm_job_drops_login_node_resource_limits() -> None:
    repository = Path(__file__).resolve().parents[1]
    scripts = sorted((repository / "slurm").glob("*.sbatch"))

    assert scripts
    for script in scripts:
        source = script.read_text()
        assert source.count("#SBATCH --propagate=NONE") == 1, script.name
