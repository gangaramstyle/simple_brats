# Historical xmodal leakage audit

Audit snapshot: 2026-07-11. The purpose of this document is to preserve what was learned from the
historical branches without carrying their implementation forward.

## What was sound

- Anchor-relative millimeter coordinates centered on a randomly sampled foreground point remove a
  fixed atlas origin. They do not hide relative anatomy or bag geometry: the anchor phase cancels in
  rotary query-key comparisons. Relative rotary position encoding is compatible with common-shift
  invariance.
- The mixed-v4 teacher did not receive target coordinates, target ordering, or surrounding tokens.
- Conditional within-bag, within-modality matching in mixed-v3/v4 is much harder to solve from
  modality or scale identity than the earlier global candidate loss.
- The context-only construction in mixed-v3 is a useful causal control for co-located evidence.

## Shortcuts or contamination found

### Geometric exclusion was approximate

The mixed-v4 sampler accepted target centers at Euclidean distance at least 8 mm. That does not
prove two square slabs are disjoint: diagonal offsets can pass the distance test while their
axis-aligned footprints still overlap. Its fallback could also return unconstrained targets. A
visible copy of the target modality at another position could therefore contain held target pixels.

The new sampler checks closed axis-aligned physical slabs directly, treats boundary contact as an
intersection, and fails the batch rather than relaxing the constraint.

### Some historical target paths could see the answer

In the main phased `forward_cross` path, the target prism was encoded before anchors were gathered.
Those apparent anchors had already attended held target tokens, and a target-series summary entered
the prediction queries. This makes the latent prediction path coordinate- and target-context-rich.
It is not a valid blind-target baseline and is not ported.

The older TCIA pixel-prediction path selected raw anchors before joint encoding and is a cleaner
pixel baseline. Its latent target path still has the above conceptual leakage risk.

### Candidate construction exposed nuisance labels

The mixed-v2 global loss pooled modalities and patch sizes, allowing the model to narrow candidates
from modality/scale identity instead of tissue semantics. Mixed-v3/v4 improved this with conditional
same-modality, fixed-size candidate sets. `simple_brats` makes bag and modality strata explicit in
the loss API and keeps scale fixed for the first comparison.

The historical band/optimal-transport experiment used segmentation masks to remove tumor voxels
from candidate band edges. That is label-informed candidate construction and is not self-supervised.

### Teacher inputs were not strictly patch-local

The v4 raw/z/CDF target variants were spatially blind, but z-score and CDF values depended on
full-scan statistics. The v0 network API accepts only a normalized clean patch tensor and no
statistics as separate features. Any per-scan normalization still changes that tensor using
scan-wide information, however, so its exact recipe and digest belong in provenance and its effect
must be tested explicitly.

### Prior probes were not clean unseen-subject estimates

Historical Phase 1/2 launchers left validation at its zero default, probe evaluation used a zero
holdout default, and a later warm start inherited those subjects. High enhancing-tumor probe scores
therefore cannot be read as held-out-patient generalization.

The live MET case split also hashed full visit IDs. In the audited snapshot, 72 canonical subject
stems crossed partitions and 76 nominal validation cases had another time point in training. The new
manifest groups visits by canonical subject and additionally rejects repeated image digests across
partitions and incompatible warm starts.

## Invariants carried into simple_brats

1. The teacher's network-level API is patch-only.
2. The target modality is absent at its target location.
3. Same-modality visible and target slabs are exactly non-intersecting.
4. Candidate strata are explicit bag plus modality IDs, never inferred from array order.
5. Modality-specific tokens remain separate during pretraining.
6. Relative geometry must pass a common-coordinate-shift test.
7. Final evaluation is locked at canonical-subject and image-digest level.

Historical revisions inspected: public `main` at `ba6d7f4`, `mixed-v2` at `82d7b71`, `mixed-v3`
at `1602b0e`, and `mixed-v4` at `9e7b238`.
