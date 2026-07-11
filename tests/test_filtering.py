import hashlib
from dataclasses import FrozenInstanceError

import pytest

from simple_brats.data.filtering import (
    ManifestFilterError,
    ManifestFilterSpec,
    SubjectExclusion,
    apply_manifest_filter,
    load_manifest_filter_spec,
    save_manifest_filter_spec,
)
from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _case(
    case_id: str,
    *,
    file_digest: str,
    release: str = "BraTS2026",
) -> CaseRecord:
    return CaseRecord.create(
        source="BraTS-MET",
        release=release,
        case_id=case_id,
        files=(
            FileRecord(
                modality="t1c",
                path=f"{release}/{case_id}/{case_id}-t1c.nii.gz",
                sha256=file_digest,
            ),
        ),
    )


def _manifest() -> tuple[DatasetManifest, str]:
    duplicate = _digest("cross-subject duplicate t1c")
    return (
        DatasetManifest(
            cases=(
                _case("BraTS-MET-00001-001", file_digest=duplicate),
                _case(
                    "BraTS-MET-00001-002",
                    file_digest=_digest("subject one second visit"),
                ),
                _case("BraTS-MET-00002-001", file_digest=duplicate),
            )
        ),
        duplicate,
    )


def _spec(
    manifest: DatasetManifest,
    evidence: str,
    *,
    subject_id: str = "BraTS-MET-00002",
) -> ManifestFilterSpec:
    return ManifestFilterSpec(
        input_manifest_sha256=manifest.sha256,
        exclusions=(
            SubjectExclusion(
                subject_id=subject_id,
                reason="Duplicate t1c component; retain the longitudinal subject.",
                evidence_sha256=(evidence,),
            ),
        ),
    )


def test_apply_removes_every_visit_of_subject_and_returns_new_manifest() -> None:
    manifest, duplicate = _manifest()
    filtered = apply_manifest_filter(manifest, _spec(manifest, duplicate))

    assert filtered is not manifest
    assert filtered.subjects == ("BraTS-MET-00001",)
    assert [case.visit_id for case in filtered.cases] == ["001", "002"]
    assert filtered.sha256 != manifest.sha256


def test_spec_json_and_sha_are_canonical_across_input_order() -> None:
    manifest, duplicate = _manifest()
    second_digest = _digest("subject one second visit")
    first = ManifestFilterSpec(
        input_manifest_sha256=manifest.sha256,
        exclusions=(
            SubjectExclusion(
                "BraTS-MET-00002",
                "duplicate",
                (duplicate,),
            ),
            SubjectExclusion(
                "BraTS-MET-00001",
                "separate reviewed reason",
                (second_digest, duplicate),
            ),
        ),
    )
    second = ManifestFilterSpec(
        input_manifest_sha256=manifest.sha256,
        exclusions=(
            SubjectExclusion(
                "BraTS-MET-00001",
                "separate reviewed reason",
                (duplicate, second_digest),
            ),
            SubjectExclusion(
                "BraTS-MET-00002",
                "duplicate",
                (duplicate,),
            ),
        ),
    )

    assert first.to_json() == second.to_json()
    assert first.sha256 == second.sha256
    assert first.excluded_subject_ids == ("BraTS-MET-00001", "BraTS-MET-00002")


def test_filter_spec_round_trip_and_expected_sha(tmp_path) -> None:
    manifest, duplicate = _manifest()
    spec = _spec(manifest, duplicate)
    path = tmp_path / "quarantine.json"

    save_manifest_filter_spec(spec, path)
    assert path.read_bytes() == spec.to_json().encode("utf-8")
    assert load_manifest_filter_spec(path, expected_sha256=spec.sha256) == spec

    with pytest.raises(ManifestFilterError, match="manifest-filter SHA mismatch"):
        load_manifest_filter_spec(path, expected_sha256=_digest("wrong spec"))


def test_spec_and_exclusions_are_immutable() -> None:
    manifest, duplicate = _manifest()
    spec = _spec(manifest, duplicate)

    with pytest.raises(FrozenInstanceError):
        spec.input_manifest_sha256 = _digest("replacement")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        spec.exclusions[0].reason = "replacement"  # type: ignore[misc]


def test_duplicate_subject_exclusions_fail_closed() -> None:
    manifest, duplicate = _manifest()
    exclusion = SubjectExclusion("BraTS-MET-00002", "duplicate", (duplicate,))
    with pytest.raises(ManifestFilterError, match="at most once"):
        ManifestFilterSpec(
            input_manifest_sha256=manifest.sha256,
            exclusions=(exclusion, exclusion),
        )


def test_duplicate_or_missing_evidence_digests_fail_closed() -> None:
    duplicate = _digest("duplicate")
    with pytest.raises(ManifestFilterError, match="at least one"):
        SubjectExclusion("BraTS-MET-00001", "reason", ())
    with pytest.raises(ManifestFilterError, match="must be unique"):
        SubjectExclusion("BraTS-MET-00001", "reason", (duplicate, duplicate))


def test_filter_is_bound_to_exact_input_manifest_sha() -> None:
    manifest, duplicate = _manifest()
    spec = ManifestFilterSpec(
        input_manifest_sha256=_digest("different input manifest"),
        exclusions=(SubjectExclusion("BraTS-MET-00002", "duplicate", (duplicate,)),),
    )

    with pytest.raises(ManifestFilterError, match="input manifest SHA mismatch"):
        apply_manifest_filter(manifest, spec)


def test_unknown_excluded_subject_fails_closed() -> None:
    manifest, duplicate = _manifest()
    spec = _spec(manifest, duplicate, subject_id="BraTS-MET-99999")

    with pytest.raises(ManifestFilterError, match="absent from the input manifest"):
        apply_manifest_filter(manifest, spec)


def test_evidence_must_be_attached_to_the_exact_excluded_subject() -> None:
    manifest, _ = _manifest()
    evidence_on_other_subject = _digest("subject one second visit")
    spec = _spec(manifest, evidence_on_other_subject)

    with pytest.raises(ManifestFilterError, match="not attached to subject"):
        apply_manifest_filter(manifest, spec)


def test_filter_cannot_remove_every_case() -> None:
    evidence = _digest("only subject")
    manifest = DatasetManifest(cases=(_case("BraTS-MET-00001-001", file_digest=evidence),))

    with pytest.raises(ManifestFilterError, match="remove every case"):
        apply_manifest_filter(manifest, _spec(manifest, evidence, subject_id="BraTS-MET-00001"))


def test_empty_exclusion_list_is_a_valid_content_addressed_noop() -> None:
    manifest, _ = _manifest()
    spec = ManifestFilterSpec(input_manifest_sha256=manifest.sha256, exclusions=())

    filtered = apply_manifest_filter(manifest, spec)
    assert filtered == manifest
    assert filtered is not manifest


def test_loader_rejects_duplicate_json_keys_and_noncanonical_shapes(tmp_path) -> None:
    duplicate_key = tmp_path / "duplicate-key.json"
    duplicate_key.write_text(
        '{"schema_version":1,"schema_version":1,'
        f'"input_manifest_sha256":"{_digest("manifest")}","exclusions":[]}}'
    )
    with pytest.raises(ManifestFilterError, match="duplicate JSON object key"):
        load_manifest_filter_spec(duplicate_key)

    with pytest.raises(ManifestFilterError, match="lowercase SHA-256"):
        ManifestFilterSpec(
            input_manifest_sha256=_digest("manifest").upper(),
            exclusions=(),
        )
    with pytest.raises(ManifestFilterError, match="schema_version must be an integer"):
        ManifestFilterSpec(
            input_manifest_sha256=_digest("manifest"),
            exclusions=(),
            schema_version=True,
        )
