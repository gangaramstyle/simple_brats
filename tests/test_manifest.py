import hashlib
import tempfile
import unittest
from pathlib import Path

from simple_brats.data.manifest import (
    CaseRecord,
    DatasetManifest,
    FileRecord,
    IdentityError,
    ManifestError,
    canonicalize_case_identity,
    load_manifest,
    save_manifest,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def case(
    case_id: str,
    *,
    release: str = "BraTS2026",
    modalities=("t1n", "t1c"),
) -> CaseRecord:
    return CaseRecord.create(
        source="BraTS-MET",
        release=release,
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality=modality,
                path=f"{release}/{case_id}/{case_id}-{modality}.nii.gz",
                sha256=digest(f"{release}:{case_id}:{modality}"),
            )
            for modality in modalities
        ),
    )


class IdentityTests(unittest.TestCase):
    def test_met_case_is_split_at_reviewed_longitudinal_boundary(self):
        identity = canonicalize_case_identity("BraTS-MET-00730-001")
        self.assertEqual(identity.subject_id, "BraTS-MET-00730")
        self.assertEqual(identity.visit_id, "001")

    def test_unsupported_or_partial_identity_fails_closed(self):
        with self.assertRaises(IdentityError):
            canonicalize_case_identity("some-case-01")
        with self.assertRaises(IdentityError):
            canonicalize_case_identity("some-case-01", subject_id="some-subject")

    def test_explicit_metadata_supports_reviewed_external_identity(self):
        identity = canonicalize_case_identity(
            "external-case", subject_id="external-subject", visit_id="baseline"
        )
        self.assertEqual(identity.subject_id, "external-subject")
        self.assertEqual(identity.visit_id, "baseline")

    def test_canonical_met_identity_cannot_be_overridden(self):
        with self.assertRaisesRegex(IdentityError, "conflicts"):
            canonicalize_case_identity(
                "BraTS-MET-00730-001",
                subject_id="WRONG-SUBJECT",
                visit_id="001",
            )


class ManifestTests(unittest.TestCase):
    def test_manifest_json_and_sha_are_order_independent(self):
        early = case("BraTS-MET-00730-001", release="r1")
        later = case("BraTS-MET-00730-002", release="r2")
        first = DatasetManifest(cases=(later, early))
        second = DatasetManifest(cases=(early, later))
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.sha256, second.sha256)

    def test_modalities_must_exactly_match_file_records(self):
        with self.assertRaises(ManifestError):
            CaseRecord(
                source="BraTS-MET",
                release="r1",
                case_id="BraTS-MET-00730-001",
                subject_id="BraTS-MET-00730",
                visit_id="001",
                modalities=("t1n", "t1c"),
                files=(FileRecord("t1n", "one.nii.gz", digest("one")),),
            )

    def test_round_trip_preserves_canonical_sha(self):
        manifest = DatasetManifest(cases=(case("BraTS-MET-00730-001"),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            save_manifest(manifest, path)
            loaded = load_manifest(path, expected_sha256=manifest.sha256)
        self.assertEqual(loaded, manifest)
        self.assertEqual(loaded.sha256, manifest.sha256)

    def test_bad_digest_is_rejected(self):
        with self.assertRaises(ManifestError):
            FileRecord("t1n", "one.nii.gz", "not-a-digest")


if __name__ == "__main__":
    unittest.main()
