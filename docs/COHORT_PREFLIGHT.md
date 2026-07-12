# Full training-cohort cold-path preflight

Before a long SSL launch consumes a new visit, this restartable gate validates all 1,044 training
cases from 643 subjects under the exact registered 4 mm configuration. For each case it derives
the visit's first real bag-zero assignment from `SubjectBalancedSchedule`, cold-prepares all four
MRI modalities, requires at least 32 strict all-modality foreground centers, materializes the
32-target hard-match plan with pool 512 / eight attempts, and assembles the complete batch.

Segmentation files and validation/test images are never passed to the loading API. Each passed case
is published as one canonical atomic result. A requeued job validates and skips those immutable
results, then continues with the first missing case. `result.json` is written only after every case
and subject passes; it includes candidate-count and timing quantiles plus content-addressed result,
candidate, prepared-volume, plan, and matching-batch digest sets.

```bash
LAUNCH_SHA=<full-commit-sha> \
DATA_ROOT="$HOME/xmodal/data/brats26/mets_train" \
DATA_GATE_BUNDLE="$HOME/simple_brats_artifacts/data-gates/brats-met-2025-training-clean-v0" \
EXPECTED_MANIFEST_SHA256=<filtered-manifest-sha256> \
EXPECTED_SPLIT_SHA256=<subject-split-sha256> \
EXPECTED_CASE_GRID_MANIFEST_SHA256=<case-grid-manifest-sha256> \
bash cluster/prepare_and_submit_cohort_preflight.sh
```

The wrapper creates a detached literal-SHA launch and submits one 12-hour `ai` / A40 job. Slurm
walltime signals are handled between cases and requeue the same job; relaunching the wrapper against
the same output is also safe. `AFTER_JOB_ID=<job>` adds an optional `afterok` dependency.
