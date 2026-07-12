"""Token-only probes and scalable representation diagnostics.

The public probe functions accept frozen feature vectors and binary labels.
They have no API for coordinates, image tensors, neighboring tokens, or
cross-modality fusion.  Every reported model is fit independently per MRI
modality, using labeled tokens from ``probe_train`` subjects, then evaluated
on globally disjoint ``validation`` subjects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from simple_brats.config import MODALITIES
from simple_brats.training.diagnostics import RepresentationStats


class ProbeEvaluationError(ValueError):
    """Frozen-token probe inputs violate the held-out evaluation contract."""


def _finite_matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2 or min(value.shape) <= 0:
        raise ProbeEvaluationError(f"{name} must be a non-empty rank-two tensor")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ProbeEvaluationError(f"{name} must contain only finite values")
    return result


def _binary_labels(value: Tensor, *, count: int, name: str = "labels") -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() != count:
        raise ProbeEvaluationError(f"{name} must have shape [{count}]")
    labels = value.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ProbeEvaluationError(f"{name} must contain only binary 0/1 values")
    return labels


@dataclass(frozen=True, slots=True, eq=False)
class FrozenTokenTable:
    """A deliberately narrow downstream view of already-extracted tokens.

    Raw patches and physical coordinates are absent by construction.  The
    metadata fields are used only to partition the feature table; the affine
    and kNN classifiers receive feature rows and labels alone.
    """

    features: Tensor
    labels: Tensor
    modality_ids: Tensor
    subject_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = _finite_matrix(self.features, "features").to(dtype=torch.float32)
        count = features.shape[0]
        labels = _binary_labels(self.labels, count=count)
        if (
            not isinstance(self.modality_ids, Tensor)
            or self.modality_ids.ndim != 1
            or self.modality_ids.numel() != count
        ):
            raise ProbeEvaluationError(f"modality_ids must have shape [{count}]")
        modality_ids = self.modality_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
        if not bool(((modality_ids >= 0) & (modality_ids < len(MODALITIES))).all()):
            raise ProbeEvaluationError("modality_ids must use the four canonical modalities")

        metadata = {
            "subject_ids": tuple(self.subject_ids),
            "partitions": tuple(self.partitions),
            "sample_ids": tuple(self.sample_ids),
        }
        for name, values in metadata.items():
            if len(values) != count or any(
                not isinstance(item, str) or not item or item != item.strip() for item in values
            ):
                raise ProbeEvaluationError(
                    f"{name} must contain {count} non-empty canonical strings"
                )
        if set(metadata["partitions"]) - {"probe_train", "validation"}:
            raise ProbeEvaluationError("partitions may contain only probe_train and validation")
        if len(set(metadata["sample_ids"])) != count:
            raise ProbeEvaluationError("sample_ids must be unique per modality token")

        probe_subjects = {
            subject
            for subject, partition in zip(
                metadata["subject_ids"], metadata["partitions"], strict=True
            )
            if partition == "probe_train"
        }
        validation_subjects = {
            subject
            for subject, partition in zip(
                metadata["subject_ids"], metadata["partitions"], strict=True
            )
            if partition == "validation"
        }
        if not probe_subjects or not validation_subjects:
            raise ProbeEvaluationError("both probe_train and validation subjects are required")
        overlap = probe_subjects & validation_subjects
        if overlap:
            raise ProbeEvaluationError(
                f"subjects cross probe_train/validation boundaries: {sorted(overlap)}"
            )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "modality_ids", modality_ids)
        for name, values in metadata.items():
            object.__setattr__(self, name, values)

    def mask(
        self,
        *,
        partition: str,
        modality_id: int,
        subjects: set[str] | frozenset[str] | None = None,
    ) -> Tensor:
        if partition not in {"probe_train", "validation"}:
            raise ProbeEvaluationError("unknown token-table partition")
        if isinstance(modality_id, bool) or modality_id not in range(len(MODALITIES)):
            raise ProbeEvaluationError("modality_id must be in [0, 4)")
        partition_mask = torch.tensor(
            [item == partition for item in self.partitions], dtype=torch.bool
        )
        result = partition_mask & (self.modality_ids == modality_id)
        if subjects is not None:
            result &= torch.tensor(
                [subject in subjects for subject in self.subject_ids], dtype=torch.bool
            )
        return result


@dataclass(frozen=True, slots=True, eq=False)
class FrozenJointTable:
    """One frozen fixed-order joint vector per physical patch location."""

    features: Tensor
    labels: Tensor
    subject_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = _finite_matrix(self.features, "features").to(dtype=torch.float32)
        count = features.shape[0]
        labels = _binary_labels(self.labels, count=count)
        subject_ids = tuple(self.subject_ids)
        partitions = tuple(self.partitions)
        sample_ids = tuple(self.sample_ids)
        for name, values in (
            ("subject_ids", subject_ids),
            ("partitions", partitions),
            ("sample_ids", sample_ids),
        ):
            if len(values) != count or any(
                not isinstance(item, str) or not item or item != item.strip() for item in values
            ):
                raise ProbeEvaluationError(
                    f"{name} must contain {count} non-empty canonical strings"
                )
        if set(partitions) - {"probe_train", "validation"}:
            raise ProbeEvaluationError("partitions may contain only probe_train and validation")
        if len(set(sample_ids)) != count:
            raise ProbeEvaluationError("joint sample_ids must be unique")
        probe_subjects = {
            subject
            for subject, partition in zip(subject_ids, partitions, strict=True)
            if partition == "probe_train"
        }
        validation_subjects = {
            subject
            for subject, partition in zip(subject_ids, partitions, strict=True)
            if partition == "validation"
        }
        if not probe_subjects or not validation_subjects:
            raise ProbeEvaluationError("both joint feature partitions are required")
        if probe_subjects & validation_subjects:
            raise ProbeEvaluationError("joint features cross subject partitions")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "subject_ids", subject_ids)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "sample_ids", sample_ids)

    def mask(
        self,
        *,
        partition: str,
        subjects: set[str] | frozenset[str] | None = None,
    ) -> Tensor:
        if partition not in {"probe_train", "validation"}:
            raise ProbeEvaluationError("unknown joint feature partition")
        result = torch.tensor([item == partition for item in self.partitions], dtype=torch.bool)
        if subjects is not None:
            result &= torch.tensor(
                [subject in subjects for subject in self.subject_ids], dtype=torch.bool
            )
        return result


@dataclass(frozen=True, slots=True)
class AffineProbe:
    """Closed-form, class-balanced ridge classifier fit on frozen tokens."""

    weight: Tensor
    bias: float
    feature_mean: Tensor
    feature_scale: Tensor
    l2_penalty: float

    def __post_init__(self) -> None:
        weight = _finite_matrix(self.weight.reshape(1, -1), "weight")[0]
        mean = _finite_matrix(self.feature_mean.reshape(1, -1), "feature_mean")[0]
        scale = _finite_matrix(self.feature_scale.reshape(1, -1), "feature_scale")[0]
        if weight.shape != mean.shape or weight.shape != scale.shape:
            raise ProbeEvaluationError("probe weight, mean, and scale must have equal width")
        if bool((scale <= 0).any()):
            raise ProbeEvaluationError("feature_scale must be strictly positive")
        if not math.isfinite(self.bias):
            raise ProbeEvaluationError("probe bias must be finite")
        if not math.isfinite(self.l2_penalty) or self.l2_penalty <= 0:
            raise ProbeEvaluationError("l2_penalty must be finite and positive")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)

    def decision_function(self, features: Tensor) -> Tensor:
        values = _finite_matrix(features, "features")
        if values.shape[1] != self.weight.numel():
            raise ProbeEvaluationError("feature width differs from the fitted affine probe")
        return ((values - self.feature_mean) / self.feature_scale) @ self.weight + self.bias


def fit_affine_probe(
    features: Tensor,
    labels: Tensor,
    *,
    l2_penalty: float = 1.0,
) -> AffineProbe:
    """Fit one deterministic affine ridge probe with train-only statistics.

    Positive and negative classes each receive total weight one half, so the
    fit cannot exploit an arbitrary sampling prevalence.  The intercept is
    unregularized.  No validation labels participate in fitting or threshold
    selection; the fixed decision threshold is zero.
    """

    values = _finite_matrix(features, "features")
    target = _binary_labels(labels, count=values.shape[0])
    if not math.isfinite(l2_penalty) or l2_penalty <= 0:
        raise ProbeEvaluationError("l2_penalty must be finite and positive")
    class_counts = torch.bincount(target, minlength=2)
    if int(class_counts.min()) == 0:
        raise ProbeEvaluationError("affine probe training requires both binary classes")

    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(torch.finfo(values.dtype).eps)
    standardized = (values - mean) / scale
    design = torch.cat((standardized, torch.ones((values.shape[0], 1), dtype=values.dtype)), dim=1)
    class_weight = torch.tensor(
        [0.5 / int(class_counts[0]), 0.5 / int(class_counts[1])], dtype=values.dtype
    )
    row_weight = class_weight[target]
    signed_target = target.to(dtype=values.dtype).mul(2.0).sub(1.0)
    gram = design.T @ (design * row_weight[:, None])
    penalty = torch.eye(design.shape[1], dtype=values.dtype) * float(l2_penalty)
    penalty[-1, -1] = 0.0
    right = design.T @ (signed_target * row_weight)
    try:
        coefficients = torch.linalg.solve(gram + penalty, right)
    except RuntimeError as error:
        raise ProbeEvaluationError("affine ridge system could not be solved") from error
    return AffineProbe(
        weight=coefficients[:-1],
        bias=float(coefficients[-1]),
        feature_mean=mean,
        feature_scale=scale,
        l2_penalty=float(l2_penalty),
    )


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    count: int
    positive_count: int
    prevalence: float
    roc_auc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "positive_count": self.positive_count,
            "prevalence": self.prevalence,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def _rank_metrics(labels: Tensor, scores: Tensor) -> tuple[float, float]:
    """Return tie-aware ROC AUC and grouped-threshold average precision."""

    order = torch.argsort(scores, descending=True, stable=True)
    ordered_labels = labels[order].to(dtype=torch.float64)
    ordered_scores = scores[order]
    _, counts = torch.unique_consecutive(ordered_scores, return_counts=True)
    ends = counts.cumsum(0) - 1
    true_positive = ordered_labels.cumsum(0)[ends]
    false_positive = (1.0 - ordered_labels).cumsum(0)[ends]
    positives = float(labels.sum())
    negatives = float(labels.numel() - labels.sum())
    tpr = torch.cat((torch.zeros(1, dtype=torch.float64), true_positive / positives))
    fpr = torch.cat((torch.zeros(1, dtype=torch.float64), false_positive / negatives))
    auc = float(torch.trapz(tpr, fpr))
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    previous_recall = torch.cat((torch.zeros(1, dtype=torch.float64), recall[:-1]))
    average_precision = float(((recall - previous_recall) * precision).sum())
    return auc, average_precision


def binary_metrics(labels: Tensor, scores: Tensor, *, threshold: float = 0.0) -> BinaryMetrics:
    if not isinstance(scores, Tensor) or scores.ndim != 1:
        raise ProbeEvaluationError("scores must be a rank-one tensor")
    score_values = scores.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(score_values).all()) or not math.isfinite(threshold):
        raise ProbeEvaluationError("scores and threshold must be finite")
    target = _binary_labels(labels, count=score_values.numel())
    counts = torch.bincount(target, minlength=2)
    if int(counts.min()) == 0:
        raise ProbeEvaluationError("binary metrics require both classes")
    auc, ap = _rank_metrics(target, score_values)
    predicted = score_values >= threshold
    positive = target == 1
    true_positive = int((predicted & positive).sum())
    true_negative = int((~predicted & ~positive).sum())
    sensitivity = true_positive / int(counts[1])
    specificity = true_negative / int(counts[0])
    return BinaryMetrics(
        count=target.numel(),
        positive_count=int(counts[1]),
        prevalence=float(target.float().mean()),
        roc_auc=auc,
        average_precision=ap,
        accuracy=(true_positive + true_negative) / target.numel(),
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
    )


def subject_macro_binary_metrics(
    labels: Tensor,
    scores: Tensor,
    subject_ids: Sequence[str],
    *,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Average complete binary metrics with equal weight per held-out subject."""

    if len(subject_ids) != labels.numel() or len(subject_ids) != scores.numel():
        raise ProbeEvaluationError("subject IDs must align with labels and scores")
    metrics: list[Mapping[str, object]] = []
    for subject in sorted(set(subject_ids)):
        mask = torch.tensor([item == subject for item in subject_ids], dtype=torch.bool)
        metrics.append(binary_metrics(labels[mask], scores[mask], threshold=threshold).to_dict())
    return _mean_dicts(metrics)


def scalable_representation_stats(features: Tensor) -> RepresentationStats:
    """Compute exact diagnostics without an O(sample_count squared) cosine matrix."""

    values = _finite_matrix(features, "features")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ProbeEvaluationError("representation diagnostics require at least 2x2 features")
    variance = float(values.var(dim=0, unbiased=False).mean())
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    effective_rank = float(
        torch.exp(
            -(probabilities * probabilities.clamp_min(torch.finfo(energy.dtype).eps).log()).sum()
        )
    )
    normalized = F.normalize(values, dim=1)
    count = values.shape[0]
    # sum_ij <x_i,x_j> = ||sum_i x_i||^2. F.normalize leaves a zero
    # row at zero, so subtract the measured diagonal instead of assuming
    # that every normalized row has unit norm.
    diagonal = normalized.square().sum(dim=1).sum()
    off_diagonal = float((normalized.sum(dim=0).square().sum() - diagonal) / (count * (count - 1)))
    return RepresentationStats(
        count=count,
        variance=max(variance, 0.0),
        effective_rank=min(max(effective_rank, 1.0), float(count)),
        off_diagonal_cosine=min(max(off_diagonal, -1.0), 1.0),
    )


def cross_patient_knn(
    bank_features: Tensor,
    bank_labels: Tensor,
    bank_subject_ids: Sequence[str],
    query_features: Tensor,
    query_labels: Tensor,
    query_subject_ids: Sequence[str],
    *,
    neighbors: Sequence[int] = (1, 5, 20),
    chunk_size: int = 512,
) -> dict[str, object]:
    """Evaluate cosine kNN/retrieval with a globally subject-disjoint bank."""

    bank = _finite_matrix(bank_features, "bank_features")
    query = _finite_matrix(query_features, "query_features")
    if bank.shape[1] != query.shape[1]:
        raise ProbeEvaluationError("bank/query feature widths differ")
    bank_target = _binary_labels(bank_labels, count=bank.shape[0], name="bank_labels")
    query_target = _binary_labels(query_labels, count=query.shape[0], name="query_labels")
    if len(bank_subject_ids) != bank.shape[0] or len(query_subject_ids) != query.shape[0]:
        raise ProbeEvaluationError("subject IDs must align with bank/query feature rows")
    overlap = set(bank_subject_ids) & set(query_subject_ids)
    if overlap:
        raise ProbeEvaluationError(f"kNN bank and queries share subjects: {sorted(overlap)}")
    requested = tuple(neighbors)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in requested)
        or max(requested) > bank.shape[0]
    ):
        raise ProbeEvaluationError("neighbors must be unique positive values within bank size")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ProbeEvaluationError("chunk_size must be a positive integer")

    bank_normalized = F.normalize(bank, dim=1)
    query_normalized = F.normalize(query, dim=1)
    maximum = max(requested)
    neighbor_labels: list[Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        similarities = query_normalized[start : start + chunk_size] @ bank_normalized.T
        indices = torch.topk(similarities, maximum, dim=1, largest=True, sorted=True).indices
        neighbor_labels.append(bank_target[indices])
    retrieved = torch.cat(neighbor_labels, dim=0)
    result: dict[str, object] = {
        "distance": "cosine",
        "bank_subject_count": len(set(bank_subject_ids)),
        "query_subject_count": len(set(query_subject_ids)),
        "subject_overlap_count": 0,
        "bank_count": bank.shape[0],
        "query_count": query.shape[0],
        "by_k": {},
    }
    by_k: dict[str, object] = {}
    for k in requested:
        selected = retrieved[:, :k]
        positive_fraction = selected.to(dtype=torch.float64).mean(dim=1)
        agreement = (selected == query_target[:, None]).to(dtype=torch.float64).mean(dim=1)
        positive_queries = query_target == 1
        negative_queries = ~positive_queries
        by_k[str(k)] = {
            "classification": binary_metrics(
                query_target, positive_fraction, threshold=0.5
            ).to_dict(),
            "classification_subject_macro": subject_macro_binary_metrics(
                query_target,
                positive_fraction,
                query_subject_ids,
                threshold=0.5,
            ),
            "retrieval_label_agreement": float(agreement.mean()),
            "retrieval_label_agreement_subject_macro": sum(
                float(
                    agreement[
                        torch.tensor(
                            [item == subject for item in query_subject_ids],
                            dtype=torch.bool,
                        )
                    ].mean()
                )
                for subject in sorted(set(query_subject_ids))
            )
            / len(set(query_subject_ids)),
            "positive_query_precision": float(selected[positive_queries].double().mean()),
            "negative_query_precision": float((1 - selected[negative_queries]).double().mean()),
        }
    result["by_k"] = by_k
    return result


def _mean_dicts(values: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not values:
        raise ProbeEvaluationError("cannot macro-average an empty metric collection")
    numeric_keys = {
        key
        for key in values[0]
        if all(isinstance(value.get(key), (int, float)) for value in values)
    }
    return {
        key: sum(float(value[key]) for value in values) / len(values)
        for key in sorted(numeric_keys)
    }


def evaluate_frozen_tokens(
    table: FrozenTokenTable,
    *,
    ordered_probe_train_subjects: Sequence[str],
    subject_budgets: Sequence[int],
    l2_penalty: float = 1.0,
    neighbors: Sequence[int] = (1, 5, 20),
) -> dict[str, object]:
    """Run per-modality affine, kNN, retrieval, and representation evaluation."""

    if not isinstance(table, FrozenTokenTable):
        raise TypeError("table must be a FrozenTokenTable")
    subject_order = tuple(ordered_probe_train_subjects)
    available = {
        subject
        for subject, partition in zip(table.subject_ids, table.partitions, strict=True)
        if partition == "probe_train"
    }
    if len(subject_order) != len(set(subject_order)) or set(subject_order) != available:
        raise ProbeEvaluationError(
            "ordered_probe_train_subjects must contain every probe-train subject exactly once"
        )
    budgets = tuple(subject_budgets)
    if (
        not budgets
        or tuple(sorted(set(budgets))) != budgets
        or any(
            isinstance(value, bool) or value <= 0 or value > len(subject_order) for value in budgets
        )
    ):
        raise ProbeEvaluationError(
            "subject_budgets must be sorted unique positive integers within the probe pool"
        )

    representation: dict[str, object] = {}
    for modality_id, modality in enumerate(MODALITIES):
        representation[modality] = {}
        for partition in ("probe_train", "validation"):
            mask = table.mask(partition=partition, modality_id=modality_id)
            representation[modality][partition] = scalable_representation_stats(
                table.features[mask]
            ).to_dict()

    budgets_report: dict[str, object] = {}
    for budget in budgets:
        selected_subjects = set(subject_order[:budget])
        modality_reports: dict[str, object] = {}
        affine_for_macro: list[Mapping[str, object]] = []
        knn_for_macro: dict[str, list[Mapping[str, object]]] = {str(k): [] for k in neighbors}
        for modality_id, modality in enumerate(MODALITIES):
            train_mask = table.mask(
                partition="probe_train",
                modality_id=modality_id,
                subjects=selected_subjects,
            )
            validation_mask = table.mask(partition="validation", modality_id=modality_id)
            probe = fit_affine_probe(
                table.features[train_mask], table.labels[train_mask], l2_penalty=l2_penalty
            )
            affine_metrics = binary_metrics(
                table.labels[validation_mask],
                probe.decision_function(table.features[validation_mask]),
                threshold=0.0,
            ).to_dict()
            train_subject_ids = [
                subject
                for subject, include in zip(table.subject_ids, train_mask.tolist(), strict=True)
                if include
            ]
            validation_subject_ids = [
                subject
                for subject, include in zip(
                    table.subject_ids, validation_mask.tolist(), strict=True
                )
                if include
            ]
            affine_subject_macro = subject_macro_binary_metrics(
                table.labels[validation_mask],
                probe.decision_function(table.features[validation_mask]),
                validation_subject_ids,
                threshold=0.0,
            )
            knn = cross_patient_knn(
                table.features[train_mask],
                table.labels[train_mask],
                train_subject_ids,
                table.features[validation_mask],
                table.labels[validation_mask],
                validation_subject_ids,
                neighbors=neighbors,
            )
            modality_reports[modality] = {
                "affine_probe": affine_metrics,
                "affine_probe_subject_macro": affine_subject_macro,
                "cross_patient_knn_retrieval": knn,
                "probe_train_token_count": int(train_mask.sum()),
                "validation_token_count": int(validation_mask.sum()),
            }
            affine_for_macro.append(affine_metrics)
            for k in neighbors:
                knn_for_macro[str(k)].append(knn["by_k"][str(k)]["classification"])
        budgets_report[str(budget)] = {
            "probe_train_subject_count": budget,
            "modalities": modality_reports,
            "macro_over_modalities": {
                "affine_probe": _mean_dicts(affine_for_macro),
                "knn_classification_by_k": {
                    key: _mean_dicts(values) for key, values in knn_for_macro.items()
                },
            },
        }
    return {
        "schema": "simple-brats.frozen-token-evaluation",
        "schema_version": 1,
        "representation_by_modality": representation,
        "subject_budget_reports": budgets_report,
        "contracts": {
            "classifier_input": "one frozen token vector only",
            "coordinates": "absent",
            "raw_pixels": "absent",
            "neighboring_tokens": "absent",
            "cross_modality_fusion": "absent",
            "probe_fit_partition": "probe_train_from_ssl_train_subjects",
            "report_partition": "subject_disjoint_validation",
            "locked_test": "untouched",
        },
    }


def evaluate_joint_frozen_features(
    table: FrozenJointTable,
    *,
    ordered_probe_train_subjects: Sequence[str],
    subject_budgets: Sequence[int],
    l2_penalty: float = 1.0,
    neighbors: Sequence[int] = (1, 5, 20),
) -> dict[str, object]:
    """Evaluate one fixed canonical four-modality representation per location."""

    if not isinstance(table, FrozenJointTable):
        raise TypeError("table must be a FrozenJointTable")
    subject_order = tuple(ordered_probe_train_subjects)
    available = {
        subject
        for subject, partition in zip(table.subject_ids, table.partitions, strict=True)
        if partition == "probe_train"
    }
    if len(subject_order) != len(set(subject_order)) or set(subject_order) != available:
        raise ProbeEvaluationError("joint probe subject order is incomplete or duplicated")
    budgets = tuple(subject_budgets)
    if (
        not budgets
        or tuple(sorted(set(budgets))) != budgets
        or any(
            isinstance(value, bool) or value <= 0 or value > len(subject_order) for value in budgets
        )
    ):
        raise ProbeEvaluationError("invalid joint feature subject budgets")
    representation = {
        partition: scalable_representation_stats(
            table.features[table.mask(partition=partition)]
        ).to_dict()
        for partition in ("probe_train", "validation")
    }
    budget_reports: dict[str, object] = {}
    for budget in budgets:
        subjects = set(subject_order[:budget])
        train_mask = table.mask(partition="probe_train", subjects=subjects)
        validation_mask = table.mask(partition="validation")
        probe = fit_affine_probe(
            table.features[train_mask], table.labels[train_mask], l2_penalty=l2_penalty
        )
        scores = probe.decision_function(table.features[validation_mask])
        validation_subject_ids = [
            subject
            for subject, include in zip(table.subject_ids, validation_mask.tolist(), strict=True)
            if include
        ]
        train_subject_ids = [
            subject
            for subject, include in zip(table.subject_ids, train_mask.tolist(), strict=True)
            if include
        ]
        budget_reports[str(budget)] = {
            "probe_train_subject_count": budget,
            "affine_probe": binary_metrics(
                table.labels[validation_mask], scores, threshold=0.0
            ).to_dict(),
            "affine_probe_subject_macro": subject_macro_binary_metrics(
                table.labels[validation_mask],
                scores,
                validation_subject_ids,
                threshold=0.0,
            ),
            "cross_patient_knn_retrieval": cross_patient_knn(
                table.features[train_mask],
                table.labels[train_mask],
                train_subject_ids,
                table.features[validation_mask],
                table.labels[validation_mask],
                validation_subject_ids,
                neighbors=neighbors,
            ),
        }
    return {
        "schema": "simple-brats.frozen-joint-feature-evaluation",
        "schema_version": 1,
        "representation": representation,
        "subject_budget_reports": budget_reports,
        "contracts": {
            "coordinates": "identically_zero_and_not_classifier_input",
            "spatial_context": "absent",
            "modality_order": list(MODALITIES),
            "probe_fit_partition": "probe_train_from_ssl_train_subjects",
            "report_partition": "subject_disjoint_validation",
            "locked_test": "untouched",
        },
    }
