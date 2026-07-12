# Locked experiment matrix

This matrix is deliberately staged. A later stage starts only after the earlier comparison has a
locked manifest, split, compute budget, and evaluation implementation. BraTS-MET is the primary
downstream methodology testbed; challenge eligibility and leaderboard optimization are not goals.

## Unit of comparison

All objective arms share the exact subject manifest, materialized and hashed patch-plan records,
held-target count, optimizer schedule, encoder architecture, initialization artifact, seeds, and
frozen evaluation code. We report both optimizer steps and encoder tokens processed. Head-specific
parameters are not counted as encoder capacity, but their depth is fixed before training. Source
evidence and visible-token count are identical within each declared causal comparison; a conventional
same-modality MAE is not presented as if it differed only in loss.

The first comparison uses the `(32 mm prism, 4 mm cube)` arm sampled to `16 x 16 x 16`, one target
modality D per bag, separate modality tokens, and a one-block prediction head. Its encoder evidence
is fixed at `30A + 30B + 30C + 6D`; its candidate table contains 32 independently sampled D targets
from the same prism. Use three seeds for screening and five independently initialized seeds for any
confirmatory claim.

## Stage 0: reject shortcuts before training

A configuration cannot launch unless the following tests pass:

- teacher embeddings are equivariant to target reordering, while loss and gradients are invariant;
- encoder/predictor output is invariant to a common physical-coordinate translation;
- source reordering leaves target-query predictions invariant; query reordering equivariantly
  reorders predictions and leaves the loss invariant;
- every complete source and target footprint lies inside the one materialized bag prism;
- held D patches are absent from the source bag and all 32 candidates in a bag share modality D;
- D is exactly balanced over each shuffled four-bag assignment cycle;
- every target/target and same-modality visible/target 3D patch pair is exactly disjoint, including
  boundaries and the interpolation kernel's physical support;
- subject IDs and image digests are disjoint across locked train, validation, and test partitions;
- a warm-start checkpoint is rejected on subject or image-digest overlap with final evaluation;
- per-modality target/student rank, variance, cosine concentration, and EMA drift exceed thresholds
  frozen on an exact subject-held-out probe with at least 64 target patches per modality. Only that
  same fixed probe can abort; stochastic training-batch statistics are logging-only. Violations
  abort rather than add an unregistered anti-collapse loss.

Patch plans are either materialized or generated statelessly from the manifest digest, canonical
case identity, epoch, bag index, and seed. Objective arms consume identical plan IDs, and a complete
bag stays on one distributed rank.

## Data-generation gate

No real training starts until extraction is locked and hashed. The four modalities must share a
RAS shape and numerically equivalent affines within a pinned header tolerance; every original
affine is retained. Each case may have a different shape, spacing, and origin. A deterministic
case-specific 1 mm grid covers that case's physical bounds;
target and source patches are integer crops from that grid when possible. The model receives
anchor-relative millimetre coordinates, so a patient-wide translation remains a coordinate gauge
without being erased from provenance. If patch-level interpolation is used, the exclusion slab is
expanded by its kernel halo. Centers lie on one fixed lattice per case so subvoxel interpolation
phase cannot identify a location.

Record orientation, voxel spacing, interpolation, padding, foreground-mask construction,
normalization, and augmentation versions in provenance. Candidate validity must be modality-agnostic
and label-free. Cross-modal registration QC and paired target-offset tests at +/-1 and +/-2 mm are
mandatory, especially for external cohorts. Shared spatial augmentation preserves alignment;
stochastic intensity noise is independent by modality so it cannot create a shared location code.
Release de-duplication uses both compressed-file SHA-256 and a canonical voxel-content digest, so a
recompressed or re-exported copy cannot cross the evaluation boundary.

## Stage 1: objective comparison at 4 mm

| Arm | Encoder evidence | Prediction target | Purpose |
| --- | --- | --- | --- |
| A | Same-modality visible context | masked pixels | conventional MAE control |
| B | Same-modality visible context | exact blind-teacher patch embedding | same-evidence target-type control |
| C | Joint allowed modalities plus context | hidden-modality pixels | cross-modal reconstruction |
| D | Joint allowed modalities plus context | exact blind-teacher patch embedding | primary hard-matching arm |
| E | Joint allowed modalities plus context | pixels and exact embedding | secondary combined-loss arm |

The primary causal comparison is C versus D: identical cross-modal sources, different prediction
target. A versus B repeats the target-type comparison with same-modality evidence. Cross-evidence
comparisons are only interpreted when their visible-token budgets have been explicitly equalized.
The combined arm E has a pre-registered loss weight chosen from development-set gradient-scale
measurements, never from final semantic scores.

Hard matching compares a query only with the 32 D targets from the same bag and prism.
The exact paired patch is the sole positive. No soft labels, spatial tolerance labels, target
coordinates, target-order signal, or scan statistics as separate features enter the teacher in
Stage 1. The normalization recipe that produced its patch tensor is nevertheless recorded.

Matching uses an EMA of the shared encoder content stem with fixed non-affine target normalization.
A frozen-initial-stem target is a required control, because both adaptive towers admit a stationary
collapsed solution and a learned patch projection can reward low-level correspondence.

## Stage 1b: downward capacity ablation

Before paying for a larger model, compare the registered base model (`width=384`, `depth=12`, six
heads; 24.20 million trainable parameters) with the compound-scaled small model (`width=256`,
`depth=8`, four heads; 7.96 million trainable parameters). The small arm is about 3.04 times smaller.
It replays the same materialized patch-plan IDs and uses the same objective, predictor depth, physical
footprint, model-visible tensor size, target count, optimizer steps, encoder-token budget, schedule,
and paired seeds. Report trainable parameters, FLOPs, tokens/second, peak memory, and wall time, but do
not equalize wall time: this experiment asks about capacity sensitivity at equal data exposure.

Interpret the downward comparison asymmetrically:

- If small is within the pre-registered equivalence margin of base on the frozen primary endpoint,
  model capacity is unlikely to be the current cap. Use small for broad screening and deprioritize a
  larger run.
- If base reliably beats small, capacity may matter, but this does not prove that larger-than-base
  will help. First run a limited development-only learning-rate/regularization check for small and
  inspect whether base training loss and semantic transfer remain improvement-limited rather than
  data- or objective-limited.
- A larger model is promoted only after that check supports a genuine capacity trend. It is not the
  default ablation.

## Stage 2: identify what the winning objective uses

Run these paired interventions with the winning Stage 1 objective:

1. co-located source modalities only;
2. co-located source modalities plus surrounding context;
3. surrounding context only;
4. source-content shuffle with geometry preserved;
5. source-coordinate shuffle with content preserved;
6. source-modality dropout, including each leave-one-source-modality-out case;
7. target-coordinate shuffle after target construction.
8. no-coordinate training and coordinate-only matching;
9. independent re-anchoring of identical content and randomized candidate-table order;
10. fixed-random target stem versus the adaptive EMA target;
11. all-four fully visible context locations versus leave-one-out-only context.

The expected useful signal is a combination of co-located cross-modal tissue evidence and local
context. Good matching accuracy that survives destructive shuffles is treated as a shortcut, not a
positive result.

## Stage 3: physical footprint

Compare the scale-matched `(32 mm prism, 4 mm cube)` and `(64 mm prism, 8 mm cube)` arms while
keeping the model tensor exactly `16 x 16 x 16` and the dimensionless context ratio fixed.
First train each arm alone. Mixed-scale bags are a separate experiment because
scale identity and interpolation artifacts can otherwise become shortcuts. The 4 mm representation
remains the primary endpoint even if a coarser context token helps downstream.

## Stage 4: ambiguity and soft matching

Before changing the hard loss, measure ambiguity with a descriptor and threshold registered before
the soft-loss results exist. The descriptor is independent of the potentially collapsed matching
teacher; use fixed morphology/intensity features plus a blinded audit on held-out development
subjects. If false negatives materially affect 4 mm targets, first ignore a narrow band of ambiguous
negatives while retaining the exact positive. A soft-positive loss is only tested as a paired
continuation from the same hard checkpoint, with frozen target descriptors and collapse diagnostics.

## Frozen MET evaluation

The encoder is frozen and modality-specific outputs remain separate. The primary scalar is the mean
per-modality macro one-versus-rest AUROC of four independently fitted affine probes for the locked
MET center-tissue labels. Each probe sees exactly one 4 mm encoder token: no raw pixels, coordinates,
token fusion, neighboring tokens, or trainable spatial context. Token extraction uses a deterministic
all-four-modality input bag. A leave-one-modality-out extraction is reported as a required sensitivity
analysis because pretraining hides one token at target locations.

Secondary endpoints are fixed before viewing test results:

- enhancing-tumor versus non-tumor enhancement discrimination where the negative-label construction
  is independently curated, subject-locked, and reviewed. BraTS masks alone do not label physiologic
  enhancement, and enhancement outside a tumor mask is not accepted as ground truth;
- a lightweight segmentation readout consisting only of normalization, one shared linear classifier
  per token, and fixed interpolation to the evaluation grid;
- fixed late fusion or concatenation of the four modality tokens, reported separately from the
  single-token primary endpoint;
- patient-level label-efficiency curves at 1, 5, 10, 25, and 100 percent of training subjects;
- cross-patient nearest-neighbor purity and retrieval, never within-patient retrieval.

Report random-encoder, raw-patch, patch-only, fixed-random-teacher, and coordinate-only baselines
alongside every frozen result. Full fine-tuning is secondary. Track token variance, covariance
effective rank, off-diagonal cosine similarity, teacher/student drift, and retrieval concentration
per modality and matching stratum throughout pretraining.

## Decision rule

Before the first real SSL run, baseline test-retest noise fixes a minimum meaningful improvement for
the single primary scalar and a hierarchical order for secondary claims. The confirmatory comparison
uses five paired seeds and a hierarchical interval that represents both seed and held-out-subject
variation. An arm advances only if the primary interval clears that margin, no pre-registered safety
endpoint materially regresses, collapse gates remain open, and coordinate-only or shuffled-source
controls do not reproduce the gain. Matching accuracy alone is never an advancement criterion.
