# Subject-balanced long pretraining

The registered long run extends the validated 4 mm small-model experiment without changing its
scientific hyperparameters. It uses `configs/v0_cross_matching_small.toml`, AdamW at `1e-4`, weight
decay `0.05`, gradient clipping at `10`, 32 positions per bag, the hard symmetric conditional
InfoNCE objective, and teacher EMA momentum `0.996`.

## Schedule contract

The locked training partition contains 1,044 cases from 643 subjects. Raw case sampling would give
more SSL weight to subjects with more visits, so the long schedule is subject balanced:

- every subject epoch deterministically reshuffles all 643 training subjects from the experiment
  seed;
- each subject receives eight consecutive bags (256 target positions) before the next subject;
- exactly one visit is used for that subject block, and visits rotate deterministically across
  subject epochs;
- one subject epoch is therefore 5,144 optimizer steps;
- the locked snapshot has at most seven visits for one subject, so every one of its 1,044 cases has
  been exposed by step 36,008; the program recomputes and enforces this coverage bound before
  training.

The 50,000-step target is 9.72 subject epochs. Its registered schedule digest for the current
filtered manifest and seed is
`4797321042581e25984038abc0ccb57dfe8859598f777502c96f02612c970912`; the runtime records and
checkpoint provenance recompute this value rather than trusting the documentation.

Eight bags is an operational amortization choice: canonical volume preparation and construction of
the safe foreground candidate universe happen once per active case block, then support eight model
updates. It is deliberately smaller than the 25-bag stability pilot so the full subject population
is reached by step 5,144 while retaining meaningful preparation reuse.

Validation and test subjects never enter SSL sampling or collapse monitoring. The fixed collapse
probe uses four deterministic training subjects, is monitoring-only, receives no gradients, and is
not an evaluation or model-selection result. Held-out representation and downstream evaluations
are separate jobs.

## Launch

Launches are split into ten dependency-chained 5,000-step A40 jobs. Each successful segment ends on
both a checkpoint and W&B artifact boundary. Checkpoints are still written every 1,000 absolute
steps. If Slurm signals impending walltime, the runner atomically writes an additional exact
off-cadence checkpoint after the current optimizer/EMA step. The same Slurm job is then requeued,
resumes that absolute step, and runs to its next 5,000-step boundary. Downstream `afterok` jobs are
released only after that boundary is complete, so walltime does not consume a segment in the chain.

```bash
LAUNCH_SHA=<full-commit-sha> \
DATA_ROOT="$HOME/xmodal/data/brats26/mets_train" \
DATA_GATE_BUNDLE="$HOME/simple_brats_artifacts/data-gates/brats-met-2025-training-clean-v0" \
EXPECTED_MANIFEST_SHA256=<filtered-manifest-sha256> \
EXPECTED_SPLIT_SHA256=<subject-split-sha256> \
EXPECTED_CASE_GRID_MANIFEST_SHA256=<case-grid-manifest-sha256> \
bash cluster/prepare_and_submit_long_run.sh
```

The wrapper checks out the literal detached `LAUNCH_SHA`, verifies the locked environment, requires
the pinned W&B package, and prints all chained Slurm job IDs. To continue a stopped chain from the
latest checkpoint, use the same SHA and contract:

```bash
RESUME_EXISTING=1 \
LAUNCH_SHA=<same-full-commit-sha> \
DATA_ROOT="$HOME/xmodal/data/brats26/mets_train" \
DATA_GATE_BUNDLE="$HOME/simple_brats_artifacts/data-gates/brats-met-2025-training-clean-v0" \
EXPECTED_MANIFEST_SHA256=<same-sha256> \
EXPECTED_SPLIT_SHA256=<same-sha256> \
EXPECTED_CASE_GRID_MANIFEST_SHA256=<same-sha256> \
bash cluster/prepare_and_submit_long_run.sh
```

Resume reconstructs and verifies the same subject schedule, fixed probe, calibration, and
provenance before the training runner accepts the checkpoint. When a plan is actually requested
during replay, an already-published plan and its audit are byte-compared with that reconstruction;
resume does not claim to scan every historical plan. The runner then restores model, optimizer, EMA
count, Python RNG, NumPy RNG, CPU Torch RNG, and CUDA RNG before requesting the exact next
absolute-step batch.

CUDA invocations fail closed unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` was present before Python
started. The provenance also binds Torch deterministic algorithms, disabled cuDNN benchmarking,
enabled deterministic cuDNN kernels, disabled CUDA/cuDNN TF32 paths, highest float32 matmul
precision, and byte-exact calibration-statistic replay. A pre-checkpoint node restart can reuse an
initialized output only through the internal recovery flag; existing
immutable artifacts must match reconstruction. Metrics and canonical plan prefixes from earlier
restart attempts are preserved under their restart-specific invocation tokens; plans requested by
the new attempt are required to match before reuse. Final checkpoint, invocation, result, unsafe
filename, or plan steps beyond the first checkpoint boundary are rejected in this restart-from-zero
path. A terminal checkpoint with a missing `result.json` is validated through the normal runner,
staged as a final offline W&B artifact, and finalized without another optimizer step.

## W&B

Compute nodes always use `WANDB_MODE=offline`. Every Slurm invocation is a separate W&B segment run
under one deterministic group derived from the immutable global provenance. This preserves retry
and walltime history while making all scalar logs and 5,000-step model artifacts syncable. On a
networked login node with W&B credentials:

```bash
OUTPUT_BUNDLE="$HOME/simple_brats_artifacts/long-runs/brats-met-small-4mm-subject-balanced-50k-v0" \
bash cluster/sync_long_run_wandb.sh
```

The output bundle also retains canonical `run-provenance.json`, `subject-schedule.json`, fixed-probe
and calibration records, per-step materialized plans, per-invocation JSONL metrics, 1,000-step
checkpoints, and per-invocation result records independently of W&B.
