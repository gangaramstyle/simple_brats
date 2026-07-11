from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from simple_brats.data.extraction import (
    CanonicalVolume,
    ExtractionError,
    ExtractionSpec,
    assert_interpolation_supports_disjoint,
    extract_patch,
    intersect_modality_foreground_support_masks,
    load_extraction_spec,
    load_nifti_ras,
    patch_interpolation_support,
    prepare_canonical_volume,
    resample_to_canonical_grid,
    save_extraction_spec,
    valid_patch_centers_mm,
)

IDENTITY_AFFINE = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _spec(shape: tuple[int, int, int] = (8, 8, 5)) -> ExtractionSpec:
    return ExtractionSpec(canonical_shape=shape, canonical_affine=IDENTITY_AFFINE)


def _data(shape: tuple[int, int, int] = (8, 8, 5)) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return values + 1.0


def _save(path: Path, data: np.ndarray, affine: np.ndarray | tuple[tuple[float, ...], ...]) -> None:
    image = nib.Nifti1Image(data, np.asarray(affine, dtype=np.float64))
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def test_extraction_spec_is_immutable_canonical_and_content_addressed() -> None:
    spec = _spec()
    decoded = json.loads(spec.to_json())
    round_trip = ExtractionSpec.from_dict(decoded)

    assert spec == round_trip
    assert spec.sha256 == round_trip.sha256
    assert spec.to_json() == json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    assert decoded["patch_source_shape"] == [4, 4, 1]
    assert decoded["patch_physical_extent_mm"] == [4.0, 4.0, 1.0]
    assert decoded["model_visible_shape"] == [16, 16, 1]
    with pytest.raises(FrozenInstanceError):
        spec.canonical_shape = (9, 9, 9)  # type: ignore[misc]


def test_extraction_spec_file_round_trip_is_strict_and_sha_bound(tmp_path: Path) -> None:
    spec = _spec()
    path = tmp_path / "extraction.json"
    save_extraction_spec(spec, path)

    assert path.read_bytes() == spec.to_json().encode("utf-8")
    assert load_extraction_spec(path, expected_sha256=spec.sha256) == spec

    with pytest.raises(ExtractionError, match="SHA mismatch"):
        load_extraction_spec(path, expected_sha256="0" * 64)


def test_extraction_spec_loader_rejects_duplicate_keys_and_non_finite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": "one", "schema": "two"}')
    with pytest.raises(ExtractionError, match="duplicate JSON object key"):
        load_extraction_spec(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value": NaN}')
    with pytest.raises(ExtractionError, match="non-finite JSON number"):
        load_extraction_spec(non_finite)


@pytest.mark.parametrize(
    "affine",
    [
        ((1.0, 0.1, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), *IDENTITY_AFFINE[2:]),
        ((-1.0, 0.0, 0.0, 0.0), *IDENTITY_AFFINE[1:]),
        ((2.0, 0.0, 0.0, 0.0), *IDENTITY_AFFINE[1:]),
    ],
)
def test_spec_rejects_oblique_non_ras_or_non_1mm_canonical_grid(
    affine: tuple[tuple[float, ...], ...],
) -> None:
    with pytest.raises(ExtractionError):
        ExtractionSpec(canonical_shape=(8, 8, 5), canonical_affine=affine)  # type: ignore[arg-type]


def test_loader_losslessly_reorients_lps_to_ras(tmp_path: Path) -> None:
    source = _data()
    lps_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 7.0],
            [0.0, -1.0, 0.0, 7.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "lps.nii.gz"
    _save(path, source, lps_affine)

    loaded = load_nifti_ras(path)

    assert loaded.source_orientation == ("L", "P", "S")
    assert tuple(nib.aff2axcodes(loaded.affine)) == ("R", "A", "S")
    np.testing.assert_array_equal(loaded.affine, np.eye(4))
    np.testing.assert_array_equal(loaded.data, np.flip(source, axis=(0, 1)))
    assert not loaded.data.flags.writeable
    assert not loaded.affine.flags.writeable


def test_loader_rejects_implicit_or_non_millimetre_spatial_units(tmp_path: Path) -> None:
    path = tmp_path / "unknown-units.nii.gz"
    nib.save(nib.Nifti1Image(_data(), np.eye(4)), path)

    with pytest.raises(ExtractionError, match="spatial units must be explicitly millimetres"):
        load_nifti_ras(path)


def test_equivalent_ras_and_lps_files_have_the_same_canonical_voxel_digest(
    tmp_path: Path,
) -> None:
    source = _data()
    lps_affine = np.array(
        [
            [-1.0, 0.0, 0.0, 7.0],
            [0.0, -1.0, 0.0, 7.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    lps_path = tmp_path / "lps.nii.gz"
    ras_path = tmp_path / "ras.nii.gz"
    _save(lps_path, source, lps_affine)
    _save(ras_path, np.flip(source, axis=(0, 1)), np.eye(4))

    lps = prepare_canonical_volume(lps_path, _spec())
    ras = prepare_canonical_volume(ras_path, _spec())

    assert lps.voxel_content_sha256 == ras.voxel_content_sha256
    assert lps.normalized_sha256 == ras.normalized_sha256
    np.testing.assert_array_equal(lps.data, ras.data)


def test_whole_volume_resampling_marks_padding_support_invalid(tmp_path: Path) -> None:
    path = tmp_path / "native.nii.gz"
    _save(path, _data(), np.eye(4))
    spec = _spec((10, 8, 5))

    resampled, valid = resample_to_canonical_grid(load_nifti_ras(path), spec, chunk_depth=2)

    assert resampled.shape == spec.canonical_shape
    assert valid.shape == spec.canonical_shape
    assert valid.dtype == np.bool_
    assert valid[:8].all()
    assert not valid[8:].any()
    assert not resampled[8:].any()


def test_whole_volume_resampling_discards_only_scanner_world_origin(tmp_path: Path) -> None:
    path = tmp_path / "translated-native.nii.gz"
    translated = np.eye(4)
    translated[:3, 3] = (-239.0, 239.0, 42.0)
    source = _data()
    _save(path, source, translated)

    resampled, valid = resample_to_canonical_grid(load_nifti_ras(path), _spec())

    np.testing.assert_array_equal(resampled, source)
    assert valid.all()


def test_foreground_mask_is_preserved_even_when_normalized_value_is_zero(
    tmp_path: Path,
) -> None:
    data = np.zeros((8, 8, 5), dtype=np.float32)
    flat = data.reshape(-1)
    flat[:159] = 1.0
    flat[159] = 2.0
    flat[160:319] = 3.0
    path = tmp_path / "mean-valued-foreground.nii.gz"
    _save(path, data, np.eye(4))

    volume = prepare_canonical_volume(path, _spec())

    mean_voxel = np.unravel_index(159, data.shape)
    assert volume.normalization_stats.mean == pytest.approx(2.0)
    assert volume.data[mean_voxel] == pytest.approx(0.0)
    assert volume.foreground_mask[mean_voxel]
    assert not volume.foreground_mask[np.unravel_index(319, data.shape)]
    assert not volume.foreground_mask.flags.writeable


def test_strict_four_modality_mask_yields_only_full_support_lattice_centers(
    tmp_path: Path,
) -> None:
    spec = _spec()
    volumes = {}
    for index, modality in enumerate(("t1n", "t1c", "t2w", "t2f")):
        data = _data() + index
        if modality == "t1c":
            data[0, 0, 0] = 0.0
        path = tmp_path / f"{modality}.nii.gz"
        _save(path, data, np.eye(4))
        volumes[modality] = prepare_canonical_volume(path, spec)

    shared = intersect_modality_foreground_support_masks(volumes, spec=spec)
    centers = valid_patch_centers_mm(spec, shared)

    assert not shared[0, 0, 0]
    assert centers.shape == (124, 3)
    assert not any(np.array_equal(center, (1.5, 1.5, 0.0)) for center in centers)
    assert any(np.array_equal(center, (1.5, 1.5, 1.0)) for center in centers)
    assert not shared.flags.writeable
    assert not centers.flags.writeable
    for center in centers:
        support = patch_interpolation_support(spec, center)
        slices = tuple(
            slice(start, stop)
            for start, stop in zip(support.start_ijk, support.stop_ijk, strict=True)
        )
        assert shared[slices].all()

    with pytest.raises(ExtractionError, match="exactly modalities"):
        intersect_modality_foreground_support_masks(
            {key: value for key, value in volumes.items() if key != "t2f"},
            spec=spec,
        )


def test_integer_crop_has_exact_physical_support_and_model_shape(tmp_path: Path) -> None:
    path = tmp_path / "native.nii.gz"
    _save(path, _data(), np.eye(4))
    spec = _spec()
    volume = prepare_canonical_volume(path, spec)

    patch = extract_patch(volume, (3.5, 3.5, 2.0), spec=spec)

    assert isinstance(volume, CanonicalVolume)
    assert patch.data.shape == (16, 16, 1)
    assert patch.support.start_ijk == (2, 2, 2)
    assert patch.support.stop_ijk == (6, 6, 3)
    assert patch.support.source_shape == (4, 4, 1)
    assert patch.support.lower_mm == (1.5, 1.5, 1.5)
    assert patch.support.upper_mm == (5.5, 5.5, 2.5)
    assert np.isfinite(patch.data).all()
    assert not patch.data.flags.writeable


def test_center_lattice_and_bounds_fail_closed() -> None:
    spec = _spec()

    with pytest.raises(ExtractionError, match="off the pinned canonical lattice"):
        patch_interpolation_support(spec, (3.0, 3.5, 2.0))
    with pytest.raises(ExtractionError, match="outside the canonical grid"):
        patch_interpolation_support(spec, (0.5, 0.5, 0.0))


def test_touching_interpolation_support_is_forbidden() -> None:
    spec = _spec((12, 12, 5))
    first = patch_interpolation_support(spec, (3.5, 3.5, 2.0))
    touching = patch_interpolation_support(spec, (7.5, 3.5, 2.0))
    separated = patch_interpolation_support(spec, (8.5, 3.5, 2.0))

    assert first.intersects(touching)
    assert not first.intersects(separated)
    with pytest.raises(ExtractionError, match="overlap or touch"):
        assert_interpolation_supports_disjoint(first, touching)
    assert_interpolation_supports_disjoint(first, separated)


def test_extractor_rejects_native_padding_and_forbidden_support(tmp_path: Path) -> None:
    path = tmp_path / "native.nii.gz"
    _save(path, _data(), np.eye(4))
    spec = _spec((10, 8, 5))
    volume = prepare_canonical_volume(path, spec)
    held_target = patch_interpolation_support(spec, (3.5, 3.5, 2.0))

    with pytest.raises(ExtractionError, match="padded native interpolation support"):
        extract_patch(volume, (7.5, 3.5, 2.0), spec=spec)
    with pytest.raises(ExtractionError, match="overlap or touch"):
        extract_patch(
            volume,
            (3.5, 3.5, 2.0),
            spec=spec,
            forbidden_supports=(held_target,),
        )


def test_nonfinite_voxels_and_bad_affines_are_rejected(tmp_path: Path) -> None:
    nonfinite = _data()
    nonfinite[2, 3, 1] = np.nan
    nonfinite_path = tmp_path / "nonfinite.nii.gz"
    _save(nonfinite_path, nonfinite, np.eye(4))
    with pytest.raises(ExtractionError, match="non-finite"):
        load_nifti_ras(nonfinite_path)

    near_singular_path = tmp_path / "near-singular.nii.gz"
    near_singular = np.diag((1.0, 1.0, 1e-10, 1.0))
    _save(near_singular_path, _data(), near_singular)
    with pytest.raises(ExtractionError, match="singular|ill-conditioned"):
        load_nifti_ras(near_singular_path)


def test_constant_or_empty_foreground_normalization_fails_closed(tmp_path: Path) -> None:
    for name, data in (
        ("empty", np.zeros((8, 8, 5), dtype=np.float32)),
        ("constant", np.ones((8, 8, 5), dtype=np.float32)),
    ):
        path = tmp_path / f"{name}.nii.gz"
        _save(path, data, np.eye(4))
        with pytest.raises(ExtractionError, match="foreground|standard deviation"):
            prepare_canonical_volume(path, _spec())
