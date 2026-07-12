from __future__ import annotations

import inspect

import pytest
import torch

from simple_brats.evaluation.probes import (
    FrozenJointTable,
    FrozenTokenTable,
    ProbeEvaluationError,
    binary_metrics,
    cross_patient_knn,
    evaluate_frozen_tokens,
    evaluate_joint_frozen_features,
    fit_affine_probe,
    scalable_representation_stats,
)
from simple_brats.training.diagnostics import representation_stats


def _token_table() -> FrozenTokenTable:
    rows: list[list[float]] = []
    labels: list[int] = []
    modalities: list[int] = []
    subjects: list[str] = []
    partitions: list[str] = []
    sample_ids: list[str] = []
    index = 0
    for partition, partition_subjects in (
        ("probe_train", ("p0", "p1", "p2")),
        ("validation", ("v0", "v1")),
    ):
        for subject_index, subject in enumerate(partition_subjects):
            for modality in range(4):
                for label in (0, 0, 1, 1):
                    signed = 2.0 * label - 1.0
                    rows.append(
                        [
                            signed * 3.0 + 0.01 * subject_index,
                            signed,
                            float(modality),
                            float(index % 3) / 10.0,
                        ]
                    )
                    labels.append(label)
                    modalities.append(modality)
                    subjects.append(subject)
                    partitions.append(partition)
                    sample_ids.append(f"sample-{index}")
                    index += 1
    return FrozenTokenTable(
        features=torch.tensor(rows),
        labels=torch.tensor(labels),
        modality_ids=torch.tensor(modalities),
        subject_ids=tuple(subjects),
        partitions=tuple(partitions),
        sample_ids=tuple(sample_ids),
    )


def test_affine_probe_is_feature_and_label_only_and_separates() -> None:
    parameters = list(inspect.signature(fit_affine_probe).parameters)
    assert parameters == ["features", "labels", "l2_penalty"]
    features = torch.tensor([[-3.0, 1.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])

    probe = fit_affine_probe(features, labels, l2_penalty=0.1)
    metrics = binary_metrics(labels, probe.decision_function(features))

    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.average_precision == pytest.approx(1.0)
    assert metrics.balanced_accuracy == pytest.approx(1.0)


def test_scalable_stats_equal_reference_without_quadratic_cosine() -> None:
    torch.manual_seed(9)
    features = torch.randn(13, 7)
    expected = representation_stats(features)
    actual = scalable_representation_stats(features)

    assert actual.count == expected.count
    assert actual.variance == pytest.approx(expected.variance, rel=1e-5, abs=1e-7)
    assert actual.effective_rank == pytest.approx(expected.effective_rank, rel=1e-5)
    assert actual.off_diagonal_cosine == pytest.approx(
        expected.off_diagonal_cosine, rel=1e-5, abs=1e-7
    )


def test_scalable_stats_zero_rows_have_zero_pairwise_cosine() -> None:
    actual = scalable_representation_stats(torch.zeros(3, 2))

    assert actual.off_diagonal_cosine == 0.0


def test_cross_patient_knn_rejects_any_subject_overlap() -> None:
    bank = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1])
    with pytest.raises(ProbeEvaluationError, match="share subjects"):
        cross_patient_knn(
            bank,
            labels,
            ("same", "p1"),
            bank,
            labels,
            ("same", "v1"),
            neighbors=(1,),
        )


def test_full_evaluation_is_per_modality_and_validation_only() -> None:
    table = _token_table()
    report = evaluate_frozen_tokens(
        table,
        ordered_probe_train_subjects=("p0", "p1", "p2"),
        subject_budgets=(1, 3),
        l2_penalty=0.1,
        neighbors=(1, 2),
    )

    assert set(report["representation_by_modality"]) == {"t1n", "t1c", "t2w", "t2f"}
    assert set(report["subject_budget_reports"]) == {"1", "3"}
    for modality_report in report["subject_budget_reports"]["3"]["modalities"].values():
        assert modality_report["affine_probe"]["roc_auc"] == pytest.approx(1.0)
        assert modality_report["cross_patient_knn_retrieval"]["subject_overlap_count"] == 0
    assert report["contracts"]["coordinates"] == "absent"
    assert report["contracts"]["locked_test"] == "untouched"


def test_token_table_rejects_subject_crossing_probe_boundary() -> None:
    with pytest.raises(ProbeEvaluationError, match="cross probe_train/validation"):
        FrozenTokenTable(
            features=torch.randn(4, 3),
            labels=torch.tensor([0, 1, 0, 1]),
            modality_ids=torch.tensor([0, 0, 0, 0]),
            subject_ids=("same", "same", "same", "same"),
            partitions=("probe_train", "probe_train", "validation", "validation"),
            sample_ids=("0", "1", "2", "3"),
        )


def test_joint_feature_view_uses_same_held_out_probe_mechanics() -> None:
    table = _token_table()
    # The synthetic rows already contain a strong label axis.  Take one
    # canonical-modality row for each synthetic physical-location index.
    keep = table.modality_ids == 0
    joint = FrozenJointTable(
        features=torch.cat((table.features[keep], table.features[keep]), dim=1),
        labels=table.labels[keep],
        subject_ids=tuple(
            subject
            for subject, include in zip(table.subject_ids, keep.tolist(), strict=True)
            if include
        ),
        partitions=tuple(
            partition
            for partition, include in zip(table.partitions, keep.tolist(), strict=True)
            if include
        ),
        sample_ids=tuple(f"joint-{index}" for index in range(int(keep.sum()))),
    )
    report = evaluate_joint_frozen_features(
        joint,
        ordered_probe_train_subjects=("p0", "p1", "p2"),
        subject_budgets=(1, 3),
        l2_penalty=0.1,
        neighbors=(1, 2),
    )

    assert report["subject_budget_reports"]["3"]["affine_probe"]["roc_auc"] == pytest.approx(1.0)
    assert report["contracts"]["modality_order"] == ["t1n", "t1c", "t2w", "t2f"]
