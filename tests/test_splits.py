import hashlib
import unittest

from simple_brats.data.manifest import CaseRecord, DatasetManifest, FileRecord
from simple_brats.data.splits import (
    SplitError,
    SplitFraction,
    SplitLeakageError,
    SplitManifest,
    SubjectAssignment,
    assert_digest_disjointness,
    assert_warm_start_compatible,
    create_subject_split,
    partition_cases,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def case(
    case_id: str,
    *,
    release: str,
    file_digest: str | None = None,
    subject_id: str | None = None,
    visit_id: str | None = None,
) -> CaseRecord:
    return CaseRecord.create(
        source="BraTS-MET",
        release=release,
        case_id=case_id,
        subject_id=subject_id,
        visit_id=visit_id,
        files=(
            FileRecord(
                modality="t1c",
                path=f"{release}/{case_id}-t1c.nii.gz",
                sha256=file_digest or digest(f"{release}:{case_id}"),
            ),
        ),
    )


class SplitTests(unittest.TestCase):
    def test_longitudinal_visits_and_releases_have_one_subject_assignment(self):
        first = case("BraTS-MET-00730-001", release="r1")
        second = case("BraTS-MET-00730-002", release="r2")
        other = case("BraTS-MET-00421-001", release="r2")
        manifest = DatasetManifest(cases=(first, second, other))
        split = create_subject_split(
            manifest,
            seed=0,
            fractions={"train": "0.5", "evaluation": "0.5"},
        )
        self.assertEqual(split.split_of(first.subject_id), split.split_of(second.subject_id))
        assigned = partition_cases(manifest, split)
        containing = [
            name
            for name, cases in assigned.items()
            if any(item.subject_id == first.subject_id for item in cases)
        ]
        self.assertEqual(len(containing), 1)
        self.assertEqual(
            sum(item.subject_id == first.subject_id for item in assigned[containing[0]]),
            2,
        )

    def test_assignment_is_deterministic_under_manifest_case_reordering(self):
        one = case("BraTS-MET-00730-001", release="r1")
        two = case("BraTS-MET-00421-001", release="r1")
        left = DatasetManifest(cases=(one, two))
        right = DatasetManifest(cases=(two, one))
        left_split = create_subject_split(left, seed=5, fractions={"train": "0.5", "test": "0.5"})
        right_split = create_subject_split(right, seed=5, fractions={"train": "0.5", "test": "0.5"})
        self.assertEqual(left.sha256, right.sha256)
        self.assertEqual(left_split.to_json(), right_split.to_json())

    def test_file_digest_crossing_partitions_is_rejected(self):
        duplicate = digest("same physical file")
        one = case("BraTS-MET-00730-001", release="r1", file_digest=duplicate)
        two = case("BraTS-MET-00421-001", release="r2", file_digest=duplicate)
        with self.assertRaises(SplitLeakageError):
            assert_digest_disjointness({"train": (one,), "test": (two,)})

    def test_split_is_bound_to_exact_manifest_sha(self):
        one = case("BraTS-MET-00730-001", release="r1")
        original = DatasetManifest(cases=(one,))
        split = SplitManifest(
            manifest_sha256=original.sha256,
            seed=0,
            fractions=(SplitFraction("train", "0.5"), SplitFraction("test", "0.5")),
            assignments=(SubjectAssignment(one.subject_id, "train"),),
        )
        changed = DatasetManifest(cases=(one, case("BraTS-MET-00421-001", release="r1")))
        with self.assertRaises(SplitError):
            partition_cases(changed, split)

    def test_generated_split_rejects_empty_declared_partition(self):
        manifest = DatasetManifest(cases=(case("BraTS-MET-00730-001", release="r1"),))
        with self.assertRaisesRegex(SplitError, "no cases"):
            create_subject_split(
                manifest,
                seed=0,
                fractions={"train": "0.5", "test": "0.5"},
            )

    def test_warm_start_rejects_subject_or_digest_overlap(self):
        warm_case = case("BraTS-MET-00730-001", release="pretrain")
        same_subject = case("BraTS-MET-00730-002", release="evaluation")
        warm = DatasetManifest(cases=(warm_case,))
        evaluation = DatasetManifest(cases=(same_subject,))
        with self.assertRaises(SplitLeakageError):
            assert_warm_start_compatible(warm, evaluation)

        same_bytes = case(
            "BraTS-MET-00421-001",
            release="evaluation",
            file_digest=next(iter(warm_case.file_digests)),
        )
        with self.assertRaises(SplitLeakageError):
            assert_warm_start_compatible(warm, DatasetManifest(cases=(same_bytes,)))

    def test_warm_start_accepts_disjoint_data(self):
        warm = DatasetManifest(cases=(case("BraTS-MET-00730-001", release="pretrain"),))
        evaluation = DatasetManifest(cases=(case("BraTS-MET-00421-001", release="evaluation"),))
        assert_warm_start_compatible(warm, evaluation)


if __name__ == "__main__":
    unittest.main()
