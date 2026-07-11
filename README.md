# simple_brats

Leakage-controlled experiments for learning semantic, modality-specific MRI patch representations
through cross-modal completion.

The v0 task hides one modality at each sampled location, lets the encoder jointly process the other
visible modality tokens, and matches a shallow contextual prediction to an EMA patch target that has
no access to position or neighboring target patches. The primary token footprint is a 4 mm
isotropic cube, sampled as `16 x 16 x 16` for the model; an 8 mm cube uses the same model-visible
shape as the first physical-scale ablation.

This repository starts from the scientific invariants and tests rather than copying the historical
`xmodal` trainers. See [the experiment specification](docs/EXPERIMENT_SPEC.md), the
[locked experiment matrix](docs/EXPERIMENT_MATRIX.md), and the
[historical leakage audit](docs/LEAKAGE_AUDIT.md).

The registered base matching config is `configs/v0_cross_matching.toml`. The first capacity ablation
is deliberately downward: `configs/v0_cross_matching_small.toml` reduces the trainable model from
24.20M to 7.96M parameters while leaving the task and patch exposure unchanged. The registered
8 mm scale ablation is `configs/v0_cross_matching_small_8mm.toml`.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

Runs that can reach the 5,000-step artifact cadence must install `uv sync --extra tracking` and use
`WANDB_MODE=offline`. Shorter diagnostics may use canonical JSONL alone. Checkpoints remain every
1,000 steps, and every 5,000-step checkpoint must also be recorded as a W&B artifact.

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
