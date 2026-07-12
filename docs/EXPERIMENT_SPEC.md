# Experiment specification

## Scientific question

Can a vision transformer learn semantically meaningful small-footprint MRI tokens by matching a
contextual prediction from visible modalities to a position-blind patch target from a hidden
modality, and do those tokens transfer better than otherwise identical pixel-reconstruction tokens?

BraTS-MET is a downstream methodology testbed, not a constraint on permissible pretraining data.
The primary result is representation quality at a 4 mm physical footprint, not a maximally engineered
segmentation system.

## V0 representation contract

- The registered scale-matched arms are `(32 mm prism, 4 mm cube)` and
  `(64 mm prism, 8 mm cube)`, preserving an 8:1 prism-to-patch ratio.
- Both physical scales are sampled into a fixed `8 x 8 x 8` tensor before the shared
  patch stem, keeping model-visible shape and architecture constant across scales.
- A center is eligible only when every voxel in its complete 3D crop is valid non-background
  foreground in all four registered modalities.
- Modality-specific tokens remain separate throughout pretraining; there is no fused location token.
- One random foreground-centred prism is materialized per bag. Every source and target patch must
  have its complete physical footprint inside that same prism; no bag patch is drawn from the rest
  of the brain.
- Coordinates are physical millimeters relative to the materialized random prism anchor; subtracting
  any common translation from patches and anchor leaves pairwise RoPE phases unchanged.
- Each bag chooses one target modality D in a deterministic shuffled four-bag cycle. All 32 held
  targets in that bag are independently sampled D patches.
- The fixed-shape encoder bag contains 96 independently positioned patches: 30 from each of the
  three non-target modalities and six D patches. Tensor order is randomized and is never semantic.
- D source patches may be visible elsewhere inside the prism only when their physical footprints do
  not intersect a held D footprint. A/B/C patches may overlap or coincide with held D locations, but
  such co-location is allowed rather than required.
- V0 pretraining admits only cases with all four registered sequences. Missing-modality padding and
  modality dropout are explicit later experiments, not data-dependent missingness inside v0 bags.

## Blind teacher invariant

The EMA teacher is a function only of each clean normalized D patch tensor. Its API cannot accept
coordinates, anchors, modality IDs, patch sizes, scan statistics as separate features, neighboring
patches, or target indices. The target table is deterministically permuted independently of the query
table and paired only by explicit IDs. This permutation is an audit guard rather than a prerequisite:
without sequence-index features, a diagonal table still cannot create diagonal content similarity by
itself. The teacher preserves layout inside a patch; "blind" means blind to patch origin. The
normalization and resampling recipe remains part of the hashed data-generating process.

## First objective matrix

All arms use the same patient manifest, materialized patch plans, encoder initialization, optimizer
budget, seeds, and target locations. Comparisons that claim to isolate a loss also use identical
source evidence and visible-token counts.

1. Same-modality masked pixel reconstruction versus same-modality hard matching.
2. Cross-modality pixel reconstruction versus cross-modality conditional hard matching; this is the
   primary target-type comparison.
3. Cross-modality reconstruction plus matching as a secondary, pre-weighted arm.

Hard matching is the primary discriminative objective. Soft matching is not part of v0. Before any
soft-positive experiment, an offline ambiguity audit must show that semantically interchangeable
4 mm targets are a material false-negative problem. The safer first intervention is to ignore
ambiguous negatives while preserving one exact positive.

## Mandatory controls

- Blind-teacher and source/query permutation invariance.
- Common coordinate-shift invariance of spatial attention.
- Exact, fail-closed same-modality 3D patch non-intersection.
- Materialized or stateless patch schedules shared across objective arms.
- Source-content shuffle, source drop, and target-coordinate shuffle.
- No-coordinate, coordinate-only matching, independent re-anchoring, and fixed-random-teacher arms.
- Co-located-only, co-located-plus-context, and context-only source ablations.
- Random encoder, raw-pixel, patch-only, and coordinate-only baselines.
- A paired downward capacity arm (`256 x 8` versus base `384 x 12`) before any larger model; equal
  patch plans and token exposure, with an optimization check before interpreting a small-model loss.
- Per-modality target/student effective rank, variance, off-diagonal cosine, and EMA drift. Abort
  thresholds are evaluated only on an exact, subject-held-out fixed probe with at least 64 target
  patches per modality; stochastic training-batch statistics remain logging-only.

## Evaluation

The primary evaluation reads one frozen modality-specific 4 mm encoder token at a time with no raw
pixels, coordinates, fusion, or trainable spatial context. Fixed late fusion, cross-patient nearest
neighbors, label efficiency, and a normalization-plus-linear segmentation readout are secondary.
Enhancing-tumor versus physiologic-enhancement evaluation requires independently curated negative
labels; tumor masks alone do not provide them. Any full fine-tuning result is secondary.

All longitudinal visits and release duplicates share one canonical subject split. A versioned data
manifest and its digest are mandatory for every checkpoint. Warm starts are rejected if their
pretraining subjects or image digests overlap the locked final evaluation set.

## Launch provenance

Every real job must record and verify a literal `LAUNCH_SHA`, resolved configuration, data-manifest
SHA, split-manifest SHA, seed, and dependency-lock digest. Git checkout/fetch happens before `sbatch`;
the compute job only verifies the detached SHA and runs `.venv/bin/python`. Checkpoints are written
every 1,000 steps and W&B artifacts every 5,000 steps.
