# Held-out scale-specific representation evaluation

This evaluation answers a narrower question than segmentation quality: does a frozen local MRI
representation carry useful local tissue/pathology information on subjects that supplied no SSL
updates? The patch set, labels, feature views, probes, and controls are fixed before inspecting a
trained checkpoint's validation performance.

## Data and label contract

The original subject split remains authoritative:

- `probe_train` is a deterministic 128-subject strict subset of the SSL `train` partition;
- every subject in `validation` is reported, using one deterministic eligible visit per subject;
- locked `test` images and segmentations are not opened and no test record can enter the patch
  manifest;
- repeated visits never cross a subject boundary.

Materialization consumes the actual segmentation-label audit JSON, not a caller-provided claim
about it. The file bytes must match
`5dc6ead2008d6b8763a050af7de6e27deb77e2540a851e2cb1d6b7afb2977222`; its manifest and split
pins must match the evaluation inputs, `semantic_names_assigned` must be false, and its aggregate
numeric values must be nonnegative integers. The audited values are `[0, 1, 2, 3, 4, 6, 8]`.
Values 6 and 8 remain ordinary `seg > 0` foreground for this binary task, but no numeric label is
given a compartment name. A future compartment probe requires a separate semantic audit.

The binary patch task deliberately leaves an ambiguity gap:

- positive: at least 16 voxels (16 mm³ on the registered 1 mm grid) of the exact source crop are
  `seg > 0`; this is 25% of the 4 mm crop and 3.125% of the 8 mm crop, so small metastases are not
  declared ineligible merely because the representation patch covers more context;
- negative: the crop and a 4 mm axis-aligned halo contain no `seg > 0` voxel;
- boundary occupancy and tumor-adjacent zero-occupancy patches: excluded;
- sampling: up to 32 positives and 32 negatives per subject, with equal nonzero class counts for
  every included subject.

Candidate locations first pass the label-free four-modality brain-foreground/support rule used by
SSL. Segmentation is loaded only afterward to label this downstream artifact. The canonical JSON
records every center, exact positive voxel count, halo status, class, selected case, subject,
partition, raw segmentation SHA, observed numeric values, and all upstream hashes.

## Frozen feature views

The primary downstream view is a co-located four-token bag. At each exact patch location, the four
normalized modality patches enter the frozen online encoder together in canonical order
`[t1n, t1c, t2w, t2f]`. All four coordinates and the anchor are identically zero. The encoder may
therefore use the cross-modality attention available to a real downstream task, but it receives no
spatial coordinate, neighboring location, or scan-level statistic. Its four still-separate output
tokens are concatenated in canonical order only after encoding; there is no learned evaluation
fusion layer.

Four singleton online-encoder views are also reported separately as isolation diagnostics. Each
patch is a one-token bag with its modality token and an internally fixed zero relative coordinate.
Changing another evaluation sample cannot affect it. The checkpoint loader accepts runner state
schema v3, loads the complete model strictly, then exposes only `system.encoder`; the predictor and
EMA target stem are unreachable from the feature path. All encoder parameters are frozen.

Two secondary controls use the identical patch locations, labels, subject budgets, and metrics:

- an architecture-matched deterministic random online encoder, including the same co-located
  four-token attention path and singleton diagnostics;
- exact normalized source-crop voxels (4x4x4 or 8x8x8), concatenated in canonical modality order
  for the joint control and reported separately per modality as well.

Raw values are never inputs to the primary learned-token probes. They are an explicitly labeled
mechanics/control arm.

## Probe and metric contract

All classifiers receive a frozen vector only. Coordinates, raw pixels, patch context, case IDs,
subject IDs, and neighboring tokens are absent from the design matrix. Each subject budget is a
nested prefix of the predeclared probe-subject order: 8, 32, then 128 subjects.

For the joint view and each modality separately, the report contains:

- a deterministic class-balanced affine ridge probe with train-only centering/scaling, fixed L2
  penalty, and fixed zero threshold;
- cross-patient cosine kNN at k=1, 5, and 20, whose reference bank contains only probe-train
  subjects and is globally disjoint from every validation query subject;
- retrieval label agreement and positive/negative query precision at each k;
- ROC AUC, average precision, accuracy, balanced accuracy, sensitivity, and specificity;
- validation micro metrics and equal-subject macro metrics;
- feature variance, effective rank, and exact mean off-diagonal cosine for probe-train and
  validation features. The cosine calculation is algebraically exact without an O(N squared)
  similarity allocation.

Modality results are reported independently and then macro-averaged, so a healthy sequence cannot
hide a collapsed one. No validation label selects L2, k, threshold, subject budget, label rule, or
patch sampling. Repeated 5,000-step validation curves are progress readouts; treating the best
observed checkpoint as a final unbiased result would require a newly unlocked subject set.

## Materialize once

Run this scheduled CPU job once per physical-scale arm, before checkpoint evaluation. The
materializer is NIfTI I/O and CPU work and intentionally does not reserve a GPU. Set
`CONFIG_RELATIVE_PATH` to the matching 4 mm or 8 mm registered config:

```bash
LAUNCH_SHA=<full-evaluator-commit> \
DATA_ROOT="$HOME/xmodal/data/brats26/mets_train" \
DATA_GATE_BUNDLE="$HOME/simple_brats_artifacts/data-gates/brats-met-2025-training-clean-v0" \
EXPECTED_MANIFEST_SHA256=<filtered-manifest-sha> \
EXPECTED_SPLIT_SHA256=<subject-split-sha> \
EXPECTED_CASE_GRID_MANIFEST_SHA256=<case-grid-manifest-sha> \
SEGMENTATION_LABEL_AUDIT_PATH=<absolute-label-audit-json> \
EXPECTED_SEGMENTATION_LABEL_AUDIT_SHA256=5dc6ead2008d6b8763a050af7de6e27deb77e2540a851e2cb1d6b7afb2977222 \
CONFIG_RELATIVE_PATH=configs/v0_cross_matching_small.toml \
bash cluster/prepare_and_submit_heldout_materialization.sh
```

The job prints the canonical evaluation-patch-manifest SHA. Keep that exact scale-specific file and
SHA fixed for every checkpoint and control in that arm. The evaluator rejects a patch manifest whose
physical or model-visible geometry differs from the checkpoint config, or whose label rule is not
the registered 16-voxel positive and 4 mm negative-halo contract. The default output stem is `v2`;
the historical 8 mm `v1` task used a 128-voxel positive threshold and is not a primary registered
evaluation artifact.

The two native-scale manifests are materialized independently. Their eligible centers, visits,
probe subjects, and binary labels may differ because the physical crops contain 64 versus 512
voxels. Each is a valid within-arm transfer readout, but an absolute 4 mm-versus-8 mm delta is not a
paired causal comparison. That claim requires a later common-center/intersection manifest with an
explicit cross-scale label contract.

## Evaluate checkpoints and W&B

The loader derives subjects actually exposed by the checkpoint's absolute step and registered
schedule; the static full-cohort declaration is not treated as evidence of exposure. The 5,000-step
checkpoint is therefore a deliberately partial progress readout (the last training subject first
appears by step 5,144). The first checkpoint with complete cohort exposure is step 6,000. Report
step 5,000 with `ALLOW_PARTIAL_SSL_TRAIN=1`, report step 6,000 once as the first full-cohort readout,
then schedule the regular full-cohort curve at steps 10,000, 15,000, and so on. A future checkpoint
path is allowed only with an `afterok` dependency:

```bash
AFTER_JOB_ID=<matching-segment-job-id> \
LAUNCH_SHA=<full-evaluator-commit> \
DATA_ROOT="$HOME/xmodal/data/brats26/mets_train" \
DATA_GATE_BUNDLE="$HOME/simple_brats_artifacts/data-gates/brats-met-2025-training-clean-v0" \
EXPECTED_MANIFEST_SHA256=<filtered-manifest-sha> \
EXPECTED_SPLIT_SHA256=<subject-split-sha> \
EXPECTED_CASE_GRID_MANIFEST_SHA256=<case-grid-manifest-sha> \
EVALUATION_PATCH_MANIFEST_PATH=<absolute-materialized-patch-json> \
EXPECTED_EVALUATION_PATCH_MANIFEST_SHA256=<exact-patch-manifest-sha> \
CHECKPOINT_PATH=<absolute-long-run-bundle>/checkpoints/step-000005000.pt \
bash cluster/prepare_and_submit_checkpoint_evaluation.sh
```

For step 5,000 add `ALLOW_PARTIAL_SSL_TRAIN=1`; omit it for step 6,000 and all later evaluations.
Each job verifies online W&B from scheduled compute before loading the evaluation data, writes a
canonical JSON report, creates a deterministic grouped W&B evaluation run, logs all scalar metrics
at the checkpoint's absolute step, and records the report as a versioned W&B evaluation artifact.
The visible run URL is printed and saved in a sibling `*.wandb.json` transport record. The canonical
scientific report is already durable before W&B logging. If an online upload was interrupted (or to
recover an older offline run), sync its retained local transaction from a networked login node:

```bash
EVALUATION_OUTPUT_DIR="$HOME/simple_brats_artifacts/heldout-evaluation/checkpoints" \
bash cluster/sync_evaluation_wandb.sh
```

The existing 1,000-step four-subject run is useful only to validate labeled mechanics. Add
`ALLOW_PARTIAL_SSL_TRAIN=1` to the preceding submission command and replace `CHECKPOINT_PATH` with
that short-run checkpoint. It remains train-only and is branded in the report as partial SSL
coverage rather than a representation result.

The loader accepts either historical `selected_train_subject_ids` or `selected_subject_ids`
checkpoint provenance, rejects disagreement when both exist, and proves that every declared and
step-exposed subject belongs only to the train partition. Registered long and short schemas derive
actual exposure from their immutable schedule prefix.
