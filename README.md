# simple_brats

Leakage-audited experiments for learning semantic, modality-specific MRI patch representations
through cross-modal completion.

Each v0 bag chooses one target modality D and one foreground-centred physical prism. The encoder sees
random patches of the other three modalities plus a small amount of non-overlapping D context, all
inside that prism. Coordinate-conditioned queries must recover the ordering of independently sampled
D patches encoded by an EMA teacher that has no access to position, order, or neighboring targets.
The two registered scale-matched arms are a 32 mm prism with 4 mm cubes and a 64 mm prism with 8 mm
cubes. Every cube is resampled to `16 x 16 x 16` for the same model architecture.

This repository starts from the scientific invariants and tests rather than copying the historical
`xmodal` trainers. See [the experiment specification](docs/EXPERIMENT_SPEC.md), the
[locked experiment matrix](docs/EXPERIMENT_MATRIX.md), and the
[historical leakage audit](docs/LEAKAGE_AUDIT.md). Operational contracts for the real experiment are
documented in [long pretraining](docs/LONG_RUN.md), the
[full-cohort cold-path gate](docs/COHORT_PREFLIGHT.md), and
[held-out evaluation](docs/HELD_OUT_EVALUATION.md).

For a visual, audit-oriented tour of the scientific task, information boundaries, known shortcut
risks, experiment status, evaluation conflicts, and cluster runtime, open the
[standalone interactive explainer](docs/repo-explainer/index.html). It uses one small attributed
BraTS case derivative and has no build or network dependency.

The registered base matching config is `configs/v0_cross_matching.toml`. The first capacity ablation
is deliberately downward: `configs/v0_cross_matching_small.toml` reduces the trainable model from
24.20M to 7.96M parameters while leaving the task and patch exposure unchanged. Its scale-matched
8 mm / 64 mm companion is `configs/v0_cross_matching_small_8mm.toml`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Runs that can reach the 5,000-step artifact cadence must install `uv sync --extra tracking`.
Registered long training and held-out evaluation require `WANDB_MODE=online`: their launch wrappers
verify credentials on the login node, and Python verifies the same server from scheduled compute
before heavy work. Canonical JSONL, plans, reports, and checkpoints remain local independently of
W&B. Checkpoints remain every 1,000 steps, and every 5,000-step checkpoint is also a version in one
provenance-keyed W&B artifact collection.

Before a real launch, verify the exact immutable checkout's compute-node connection without mixing
network tracking into the throughput gate:

```bash
LAUNCH_SHA=<full-commit-sha> \
bash cluster/prepare_and_submit_wandb_online_smoke.sh
```

The live MET layout can be locked into a content-addressed manifest and subject-level split with:

```bash
simple-brats build-met-manifest \
  --root /path/to/mets_train \
  --source BraTS-MET \
  --release 2026-training \
  --output manifests/met-2026.json

simple-brats build-split \
  --manifest manifests/met-2026.json \
  --expected-manifest-sha <sha256-from-previous-command> \
  --output manifests/met-2026-split.json
```

Manifest construction hashes all images and belongs in a scheduled preprocessing job on the cluster,
not on a login node.

Every cluster launch requires a detached, verified git SHA and a versioned subject-level data
manifest. A synthetic A40 smoke job is submitted from a cluster login node with:

```bash
LAUNCH_SHA=<full-commit-sha> bash cluster/prepare_and_submit_smoke.sh
```

The preparation step checks out the literal SHA and materializes the uv environment before
`sbatch`; the compute job only verifies provenance and runs `.venv/bin/python`.
