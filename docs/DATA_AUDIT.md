# BraTS-MET data audit

This log records dataset decisions made before any real-data SSL result was observed. Dataset
quarantines are immutable protocol inputs rather than ad hoc changes to a split.

## BraTS 2025 MET training release

- Raw manifest SHA-256:
  `03e2ca3f42d0ac8d3e14d99633d0bd61e04107286cd87e022bfe51d1aef68fff`
- Raw inventory: 1,296 cases and 810 canonical subjects.
- Initial seed-0 subject split failed the file-digest disjointness gate. This is a dataset finding,
  not a reason to weaken the gate.
- `BraTS-MET-00418` and `BraTS-MET-00507` have byte-identical `t1n`, `t1c`, `t2w`, `t2f`, and
  segmentation files under distinct canonical subject IDs.
- `BraTS-MET-00007` and `BraTS-MET-00019` have byte-identical `t1n`, `t1c`, and `t2f` files under
  distinct canonical subject IDs. Their `t2w` and segmentation files differ.
- Duplicate segmentation files confined to longitudinal visits of the same canonical subject do not
  cross a subject split and are not quarantined by this audit.

For the first clean protocol, all four members of the two ambiguous cross-subject components are
quarantined. Arbitrarily retaining one ID could preserve an incorrectly labeled or partially copied
case, while treating the IDs as one subject would leave uncertain identity in representation
learning. The decision can be revisited as a new manifest/filter pair after identity review; it is
not changed within an experiment family.

The canonical filter is
[`protocols/brats_met_2025_cross_subject_duplicate_quarantine.json`](../protocols/brats_met_2025_cross_subject_duplicate_quarantine.json),
with canonical payload SHA-256
`64294c244d5c87aeeb44f982d0739c94c7856e8459854add101640c2c20cdcfe`. Each exclusion cites only
cross-subject MRI file digests attached to that subject in the raw manifest. The filter removes every
visit for an excluded canonical subject and fails if its input manifest or evidence changes.

## Physical-grid audit

The release is not on one global array grid. For example, `BraTS-MET-00001-000` is
`240 x 240 x 155` at 1 mm spacing, while `BraTS-MET-00553-000` is `256 x 256 x 112` at
approximately `0.8594 x 0.8594 x 1.5` mm. Scanner origins also vary across cases. A global shape,
affine, or zero-origin rewrite is therefore not part of the protocol.

The locked extraction artifact is a case-grid manifest. It requires the four MRI modalities to
share a shape and numerically equivalent affines after lossless RAS reorientation within each case,
using fixed tolerances of `atol=1e-5` and `rtol=1e-6`, while allowing shape, spacing, orientation,
and origin to vary across cases. Every modality's native grid retains its real affine. A
deterministic axis-aligned 1 mm prepared grid is derived from all eight native voxel-cell boundary
corners, and its origin preserves the lower physical boundary. Patch plans store physical RAS-mm
centers; the model subtracts a per-bag anchor only at the positional-attention boundary.

Some MRI headers omit `xyzt_units`. An `unknown` unit is interpreted as millimetres only when at
least one companion MRI in the same case explicitly declares `mm` and all four RAS grids agree
within the pinned header tolerance. Every modality's declared unit is retained in the case-grid
manifest; explicit non-mm units and cases with no mm declaration fail the gate.
