import pytest
import torch

from simple_brats.training import (
    CollapseThresholds,
    RepresentationStats,
    collapse_reasons,
    representation_stats,
    stats_by_modality,
)


def test_stats_are_computed_separately_by_modality() -> None:
    torch.manual_seed(101)
    features = torch.randn(2, 8, 12)
    modality_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]]).expand(2, -1)
    grouped = stats_by_modality(features, modality_ids)
    assert set(grouped) == {0, 1, 2, 3}
    assert all(stats.count == 4 for stats in grouped.values())


def test_collapse_reasons_compare_to_locked_reference() -> None:
    torch.manual_seed(102)
    reference = representation_stats(torch.randn(16, 8))
    collapsed = representation_stats(torch.ones(16, 8) + 1e-5 * torch.randn(16, 8))
    thresholds = CollapseThresholds(
        minimum_variance_ratio=0.1,
        minimum_effective_rank_ratio=0.5,
        maximum_off_diagonal_cosine=0.9,
    )
    reasons = collapse_reasons(collapsed, reference, thresholds)
    assert "variance_ratio" in reasons
    assert "off_diagonal_cosine" in reasons


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"count": 1}, "count must be at least two"),
        ({"variance": -0.1}, "variance must be non-negative"),
        ({"variance": float("nan")}, "variance must be finite"),
        ({"effective_rank": 0.9}, "effective_rank must lie"),
        ({"effective_rank": 9.0}, "effective_rank must lie"),
        ({"off_diagonal_cosine": float("inf")}, "off_diagonal_cosine must be finite"),
        ({"off_diagonal_cosine": 1.1}, "off_diagonal_cosine must lie"),
    ],
)
def test_representation_stats_reject_invalid_fields(updates, error) -> None:
    values = {
        "count": 8,
        "variance": 1.0,
        "effective_rank": 4.0,
        "off_diagonal_cosine": 0.0,
    }
    values.update(updates)
    with pytest.raises((TypeError, ValueError), match=error):
        RepresentationStats(**values)


def test_collapse_thresholds_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="minimum_variance_ratio must be finite"):
        CollapseThresholds(
            minimum_variance_ratio=float("nan"),
            minimum_effective_rank_ratio=0.5,
            maximum_off_diagonal_cosine=0.9,
        )
