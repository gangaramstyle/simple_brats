import torch

from simple_brats.training import (
    CollapseThresholds,
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
