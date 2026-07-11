# Experiment specification

## Scientific question

Can a vision transformer learn semantically meaningful small-footprint MRI tokens by matching a
contextual prediction from visible modalities to a position-blind patch target from a hidden
modality, and do those tokens transfer better than otherwise identical pixel-reconstruction tokens?

BraTS-MET is a downstream methodology testbed, not a constraint on permissible pretraining data.
The primary result is representation quality at a 4 mm physical footprint, not a maximally engineered
segmentation system.

## V0 representation contract

- The primary patch is a 4 x 4 x 4 mm isotropic cube; 8 x 8 x 8 mm is the first
  physical-scale ablation.
- Both physical scales are sampled into a fixed `16 x 16 x 16` tensor before the shared
  patch stem, keeping model-visible shape and architecture constant across scales.
- A center is eligible only when every voxel in its complete 3D crop is valid non-background
  foreground in all four registered modalities.
- Modality-specific tokens remain separate throughout pretraining; there is no fused location token.
- Coordinates are physical millimeters relative to the query-centroid gauge; subtracting any common
  anchor leaves the pairwise RoPE phases unchanged.
- Exactly one modality is hidden at each target location. Every other available modality may be
  visible at that location.
- The hidden target modality may be visible elsewhere in the bag only when its physical footprint
  does not intersect any target footprint.
- V0 pretraining admits only cases with all four registered sequences. Missing-modality padding and
  modality dropout are explicit later experiments, not data-dependent missingness inside v0 bags.

## Blind teacher invariant

The EMA teacher is a function only of the clean normalized target patch tensor. Its API cannot accept
coordinates, anchors, modality IDs, patch sizes, scan statistics as separate features, neighboring
patches, or target indices. It preserves spatial layout inside the patch; "blind" means blind to
patch origin. The normalization and resampling recipe that produced that tensor is still part of the
data-generating process and must be named, hashed, and ablated; patch-only API access does not make
scan-derived preprocessing disappear.

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
- Per-modality target/student effective rank, variance, off-diagonal cosine, and EMA drift with
  pre-registered abort thresholds.

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
