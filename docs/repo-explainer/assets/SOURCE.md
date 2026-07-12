# MRI image source and derivative record

The five PNG files in this directory are small, display-only derivatives of one case from the
public [MedOtter BraTS 2023 GLI dataset](https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset)
on Hugging Face. The dataset card lists the release as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and describes T1-native (`t1n`),
post-contrast T1 (`t1c`), T2-weighted (`t2w`), and T2-FLAIR (`t2f`) NIfTIs resampled to 1 mm
isotropic resolution. For this downloaded case, we independently verified that all four images and
the segmentation have shape `240 x 240 x 155` and exactly equal NIfTI affines. That is a
header-level alignment check, not proof of image-content registration.

Attribution requested by the dataset card: **BraTS Organizers**, “The BraTS 2023 Challenge on Brain
Tumor Segmentation,” arXiv (2023). The linked dataset card and its Synapse homepage remain the
authoritative source for dataset provenance and access terms.

## Case and source files

- Case: `BraTS-GLI-00000-000`
- Source slice: axial array index `z=74`, selected because it had the largest nonzero segmentation
  area in this case
- `BraTS-GLI-00000-000-t1n.nii.gz` — SHA-256
  `e92b72ee221624c36cf89ac826deceeee7097f46dd66a5d218d18b7916ebd67d`
- `BraTS-GLI-00000-000-t1c.nii.gz` — SHA-256
  `735c27fd7a17b1702875837bcc843eedc88ced6ac2cb0e73cdf995e3e64ba82f`
- `BraTS-GLI-00000-000-t2w.nii.gz` — SHA-256
  `955ff59d053e87153bb7c809235743ec904817727ec02c630f3141e191d6f452`
- `BraTS-GLI-00000-000-t2f.nii.gz` — SHA-256
  `bb899a83627591e55cada00b2c6d5402199832b717c8b9f90bb550fe35d971ff`
- `BraTS-GLI-00000-000-seg.nii.gz` — SHA-256
  `5f74bef54e7c4eda1a8c329c1f73ef863460294525b82e99ac587fe72f10f6c6`

Each MRI slice was independently display-normalized using its nonzero-volume 0.5th and 99.5th
percentiles, clipped to that interval, mapped to `[0, 1]`, and displayed with gamma `0.82`. This is
only a visualization transform; it is not the repository's training normalization. The
segmentation derivative is a transparent contour overlay: numeric labels 1, 2, and 3 are rendered
in three distinct colors without assigning them semantic compartment names.

The original NIfTI files are intentionally kept under the git-ignored local path
`data/explainer/BraTS-GLI-00000-000/` for reproducible local visual work. They are not distributed
by this repository. Only the five 240 x 240 PNG derivatives are versioned here.
