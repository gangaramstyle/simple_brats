from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from simple_brats.cohort_preflight import (
    CANDIDATE_POOL_SIZE,
    MAX_PLAN_ATTEMPTS,
    REGISTERED_CONFIG_SHA256,
    TARGET_COUNT,
    CohortPreflightError,
    WalltimeStop,
    _nearest_rank_summary,
    _publish_or_require,
    derive_first_occurrences,
)
from simple_brats.data.manifest import CaseRecord, FileRecord
from simple_brats.long_run import SubjectBalancedSchedule


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(subject: int, visit: int) -> CaseRecord:
    case_id = f"BraTS-MET-{subject:05d}-{visit:03d}"
    return CaseRecord.create(
        source="BraTS-MET",
        release="test",
        case_id=case_id,
        files=tuple(
            FileRecord(
                modality=modality,
                path=f"{case_id}/{case_id}-{modality}.nii.gz",
                sha256=_digest(f"{case_id}:{modality}"),
            )
            for modality in ("t1n", "t1c", "t2w", "t2f")
        ),
    )


def test_first_occurrence_is_each_cases_actual_earliest_bag_zero() -> None:
    cases = (
        _case(1, 0),
        _case(1, 1),
        _case(1, 2),
        _case(2, 0),
        _case(3, 0),
        _case(3, 1),
    )
    schedule = SubjectBalancedSchedule(cases, seed=0, bags_per_subject=8)

    occurrences = derive_first_occurrences(schedule)

    assert len(occurrences) == len(cases)
    for case in schedule.cases:
        key = (case.source, case.release, case.subject_id, case.visit_id, case.case_id)
        occurrence = occurrences[key]
        assignment = schedule.assignment_for_step(occurrence.absolute_step_index)
        assert assignment.case_index == occurrence.case_index
        assert assignment.subject_epoch == occurrence.subject_epoch
        assert assignment.bag_index == occurrence.bag_index == 0
        assert occurrence.absolute_step_index % schedule.bags_per_subject == 0
        for earlier in range(0, occurrence.absolute_step_index, schedule.bags_per_subject):
            assert schedule.assignment_for_step(earlier).case_index != occurrence.case_index


def test_registered_preflight_scientific_constants_are_exact() -> None:
    assert TARGET_COUNT == 32
    assert CANDIDATE_POOL_SIZE == 512
    assert MAX_PLAN_ATTEMPTS == 8
    assert REGISTERED_CONFIG_SHA256 == (
        "10396ae83b1b1c5fc9d710bbd3f9ccff6e720a48e4f86c9338f1d198af08b376"
    )


def test_nearest_rank_summary_has_explicit_minimum_and_quantiles() -> None:
    summary = _nearest_rank_summary([40, 10, 30, 20, 50])

    assert summary["minimum"] == 10
    assert summary["maximum"] == 50
    assert summary["mean"] == 30.0
    assert summary["quantile_method"] == "nearest_rank_v1"
    assert summary["quantiles"] == {
        "p00": 10,
        "p01": 10,
        "p05": 10,
        "p25": 20,
        "p50": 30,
        "p75": 40,
        "p95": 50,
        "p99": 50,
        "p100": 50,
    }


def test_atomic_contract_publication_is_restartable_but_conflicts_fail(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    value = {
        "schema": "test",
        "schema_version": 1,
        "value": 7,
        "tuple_normalized_by_canonical_json": (1, 2),
    }

    first_sha = _publish_or_require(path, value, "test contract")
    second_sha = _publish_or_require(path, value, "test contract")

    assert first_sha == second_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(CohortPreflightError, match="conflicts"):
        _publish_or_require(path, {**value, "value": 8}, "test contract")


def test_walltime_signal_flag_is_checked_between_cases() -> None:
    stop = WalltimeStop()
    assert not stop()

    stop.handle(10, None)

    assert stop()
    assert stop.signal_number == 10
