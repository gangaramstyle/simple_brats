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
