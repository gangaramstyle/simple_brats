import pytest
import torch
import torch.nn.functional as F

from simple_brats.objectives import hard_symmetric_info_nce


def _metadata() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Four independent candidate groups, each containing two positions.
    bag_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    modality_ids = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    pair_ids = torch.tensor([10, 11, 20, 21, 10, 11, 20, 21])
    return bag_ids, modality_ids, pair_ids


def test_loss_is_invariant_to_independent_target_permutation() -> None:
    torch.manual_seed(30)
    predictions = F.normalize(torch.randn(8, 16), dim=-1)
    targets = predictions.clone()
    bag_ids, modality_ids, pair_ids = _metadata()

    reference = hard_symmetric_info_nce(
        predictions,
        targets,
        prediction_bag_ids=bag_ids,
        prediction_modality_ids=modality_ids,
        prediction_pair_ids=pair_ids,
        target_bag_ids=bag_ids,
        target_modality_ids=modality_ids,
        target_pair_ids=pair_ids,
    )
    permutation = torch.tensor([6, 1, 4, 3, 0, 7, 2, 5])
    permuted = hard_symmetric_info_nce(
        predictions,
        targets[permutation],
        prediction_bag_ids=bag_ids,
        prediction_modality_ids=modality_ids,
        prediction_pair_ids=pair_ids,
        target_bag_ids=bag_ids[permutation],
        target_modality_ids=modality_ids[permutation],
        target_pair_ids=pair_ids[permutation],
    )

    torch.testing.assert_close(permuted.loss, reference.loss)
    torch.testing.assert_close(permuted.accuracy, reference.accuracy)
    torch.testing.assert_close(permuted.chance, reference.chance)
    assert reference.accuracy.item() == 1.0
    assert reference.chance.item() == 0.5
    assert reference.num_groups == 4


def test_loss_is_invariant_to_consistent_pair_permutation() -> None:
    torch.manual_seed(31)
    predictions = torch.randn(8, 12)
    targets = torch.randn(8, 12)
    bag_ids, modality_ids, pair_ids = _metadata()
    reference = hard_symmetric_info_nce(
        predictions,
        targets,
        prediction_bag_ids=bag_ids,
        prediction_modality_ids=modality_ids,
        prediction_pair_ids=pair_ids,
        target_bag_ids=bag_ids,
        target_modality_ids=modality_ids,
        target_pair_ids=pair_ids,
    )
    permutation = torch.tensor([7, 3, 5, 0, 6, 2, 1, 4])
    permuted = hard_symmetric_info_nce(
        predictions[permutation],
        targets[permutation],
        prediction_bag_ids=bag_ids[permutation],
        prediction_modality_ids=modality_ids[permutation],
        prediction_pair_ids=pair_ids[permutation],
        target_bag_ids=bag_ids[permutation],
        target_modality_ids=modality_ids[permutation],
        target_pair_ids=pair_ids[permutation],
    )
    torch.testing.assert_close(permuted.loss, reference.loss)
    torch.testing.assert_close(permuted.accuracy, reference.accuracy)


def test_duplicate_positive_identity_fails_closed() -> None:
    predictions = torch.randn(2, 4)
    targets = torch.randn(2, 4)
    with pytest.raises(ValueError, match="exactly one"):
        hard_symmetric_info_nce(
            predictions,
            targets,
            prediction_bag_ids=torch.zeros(2, dtype=torch.long),
            prediction_modality_ids=torch.zeros(2, dtype=torch.long),
            prediction_pair_ids=torch.tensor([0, 0]),
            target_bag_ids=torch.zeros(2, dtype=torch.long),
            target_modality_ids=torch.zeros(2, dtype=torch.long),
            target_pair_ids=torch.tensor([0, 0]),
        )


def test_singleton_candidate_group_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least two alternatives"):
        hard_symmetric_info_nce(
            torch.randn(1, 4),
            torch.randn(1, 4),
            prediction_bag_ids=torch.zeros(1, dtype=torch.long),
            prediction_modality_ids=torch.zeros(1, dtype=torch.long),
        )
