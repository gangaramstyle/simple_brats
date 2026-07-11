"""Hard conditional patch matching with explicit candidate identities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class MatchingOutput:
    """Loss and retrieval diagnostics for symmetric conditional InfoNCE."""

    loss: Tensor
    accuracy: Tensor
    chance: Tensor
    forward_accuracy: Tensor
    backward_accuracy: Tensor
    num_pairs: int
    num_groups: int

    def as_dict(self) -> dict[str, Tensor | int]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "chance": self.chance,
            "forward_accuracy": self.forward_accuracy,
            "backward_accuracy": self.backward_accuracy,
            "num_pairs": self.num_pairs,
            "num_groups": self.num_groups,
        }


def _flatten_features(features: Tensor, name: str) -> Tensor:
    if features.ndim < 2:
        raise ValueError(f"{name} must have at least two dimensions")
    if features.shape[-1] <= 0:
        raise ValueError(f"{name} must have a non-empty embedding dimension")
    return features.reshape(-1, features.shape[-1])


def _flatten_metadata(metadata: Tensor, expected: int, name: str, device: torch.device) -> Tensor:
    if not isinstance(metadata, Tensor):
        raise TypeError(f"{name} must be a tensor")
    metadata = metadata.reshape(-1).to(device=device)
    if metadata.numel() != expected:
        raise ValueError(f"{name} must contain {expected} entries, got {metadata.numel()}")
    if metadata.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must contain integer IDs")
    return metadata.to(torch.int64)


def hard_symmetric_info_nce(
    predictions: Tensor,
    targets: Tensor,
    *,
    prediction_bag_ids: Tensor,
    prediction_modality_ids: Tensor,
    prediction_pair_ids: Tensor | None = None,
    target_bag_ids: Tensor | None = None,
    target_modality_ids: Tensor | None = None,
    target_pair_ids: Tensor | None = None,
    prediction_scale_ids: Tensor | None = None,
    target_scale_ids: Tensor | None = None,
    temperature: float = 0.07,
) -> MatchingOutput:
    """Compute hard symmetric InfoNCE within explicit conditional groups.

    Candidate groups are ``(bag ID, target modality ID)``.  If scale IDs are
    supplied, scale becomes an additional group key, guaranteeing that a
    query never compares against a target from another physical footprint.

    ``pair_ids`` identify the one hard positive within a group.  Supplying
    distinct prediction/target metadata makes the objective invariant to an
    independent permutation of the target table.  When pair IDs and target
    metadata are omitted, aligned prediction/target order is assumed as a
    convenience for simple batches.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    predictions = _flatten_features(predictions, "predictions")
    targets = _flatten_features(targets, "targets")
    if predictions.shape[-1] != targets.shape[-1]:
        raise ValueError("predictions and targets must have the same embedding dimension")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if predictions.shape[0] == 0 or targets.shape[0] == 0:
        raise ValueError("matching requires at least one prediction and target")
    device = predictions.device
    n_predictions, n_targets = predictions.shape[0], targets.shape[0]

    prediction_bag_ids = _flatten_metadata(
        prediction_bag_ids, n_predictions, "prediction_bag_ids", device
    )
    prediction_modality_ids = _flatten_metadata(
        prediction_modality_ids, n_predictions, "prediction_modality_ids", device
    )

    aligned_default = n_predictions == n_targets
    if target_bag_ids is None:
        if not aligned_default:
            raise ValueError("target_bag_ids are required when prediction and target counts differ")
        target_bag_ids = prediction_bag_ids
    else:
        target_bag_ids = _flatten_metadata(target_bag_ids, n_targets, "target_bag_ids", device)
    if target_modality_ids is None:
        if not aligned_default:
            raise ValueError(
                "target_modality_ids are required when prediction and target counts differ"
            )
        target_modality_ids = prediction_modality_ids
    else:
        target_modality_ids = _flatten_metadata(
            target_modality_ids, n_targets, "target_modality_ids", device
        )

    if prediction_pair_ids is None:
        if not aligned_default:
            raise ValueError(
                "prediction_pair_ids are required when prediction and target counts differ"
            )
        prediction_pair_ids = torch.arange(n_predictions, device=device)
    else:
        prediction_pair_ids = _flatten_metadata(
            prediction_pair_ids, n_predictions, "prediction_pair_ids", device
        )
    if target_pair_ids is None:
        if not aligned_default:
            raise ValueError(
                "target_pair_ids are required when prediction and target counts differ"
            )
        target_pair_ids = prediction_pair_ids
    else:
        target_pair_ids = _flatten_metadata(target_pair_ids, n_targets, "target_pair_ids", device)

    if (prediction_scale_ids is None) != (target_scale_ids is None):
        if target_scale_ids is None and aligned_default:
            target_scale_ids = prediction_scale_ids
        else:
            raise ValueError("prediction_scale_ids and target_scale_ids must be supplied together")
    group_columns_prediction = [prediction_bag_ids, prediction_modality_ids]
    group_columns_target = [target_bag_ids, target_modality_ids]
    if prediction_scale_ids is not None:
        prediction_scale_ids = _flatten_metadata(
            prediction_scale_ids, n_predictions, "prediction_scale_ids", device
        )
        target_scale_ids = _flatten_metadata(
            target_scale_ids,
            n_targets,
            "target_scale_ids",
            device,  # type: ignore[arg-type]
        )
        group_columns_prediction.append(prediction_scale_ids)
        group_columns_target.append(target_scale_ids)

    prediction_keys = torch.stack(group_columns_prediction, dim=1)
    target_keys = torch.stack(group_columns_target, dim=1)
    all_keys = torch.unique(torch.cat((prediction_keys, target_keys), dim=0), dim=0)

    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    forward_loss_sum = predictions.new_zeros(())
    backward_loss_sum = predictions.new_zeros(())
    forward_correct = predictions.new_zeros(())
    backward_correct = predictions.new_zeros(())
    total_pairs = 0
    total_groups = 0

    for key in all_keys:
        prediction_indices = torch.nonzero(
            (prediction_keys == key).all(dim=1), as_tuple=False
        ).squeeze(1)
        target_indices = torch.nonzero((target_keys == key).all(dim=1), as_tuple=False).squeeze(1)
        if prediction_indices.numel() == 0 or target_indices.numel() == 0:
            raise ValueError(f"candidate group {key.tolist()} is missing predictions or targets")
        if prediction_indices.numel() < 2 or target_indices.numel() < 2:
            raise ValueError(
                f"candidate group {key.tolist()} must contain at least two alternatives"
            )

        group_prediction_pair_ids = prediction_pair_ids[prediction_indices]
        group_target_pair_ids = target_pair_ids[target_indices]
        positive_matrix = group_prediction_pair_ids[:, None] == group_target_pair_ids[None, :]
        if not bool((positive_matrix.sum(dim=1) == 1).all()):
            raise ValueError(
                f"every prediction in group {key.tolist()} must have exactly one target"
            )
        if not bool((positive_matrix.sum(dim=0) == 1).all()):
            raise ValueError(
                f"every target in group {key.tolist()} must have exactly one prediction"
            )

        group_predictions = predictions[prediction_indices]
        group_targets = targets[target_indices]
        logits = torch.matmul(group_predictions, group_targets.transpose(0, 1)) / temperature
        forward_labels = positive_matrix.to(torch.int64).argmax(dim=1)
        backward_labels = positive_matrix.transpose(0, 1).to(torch.int64).argmax(dim=1)
        forward_loss_sum = forward_loss_sum + F.cross_entropy(
            logits, forward_labels, reduction="sum"
        )
        backward_loss_sum = backward_loss_sum + F.cross_entropy(
            logits.transpose(0, 1), backward_labels, reduction="sum"
        )
        with torch.no_grad():
            forward_correct = forward_correct + (logits.argmax(dim=1) == forward_labels).sum()
            backward_correct = (
                backward_correct + (logits.transpose(0, 1).argmax(dim=1) == backward_labels).sum()
            )
        total_pairs += prediction_indices.numel()
        total_groups += 1

    if total_pairs != n_predictions or total_pairs != n_targets:
        raise ValueError("matching requires a one-to-one prediction/target pair set")

    denominator = float(total_pairs)
    loss = 0.5 * (forward_loss_sum + backward_loss_sum) / denominator
    forward_accuracy = forward_correct / denominator
    backward_accuracy = backward_correct / denominator
    accuracy = 0.5 * (forward_accuracy + backward_accuracy)
    # For a group of k candidates, each of its k queries has random accuracy
    # 1/k.  Thus every group contributes one expected correct retrieval.
    chance = predictions.new_tensor(total_groups / denominator)
    return MatchingOutput(
        loss=loss,
        accuracy=accuracy,
        chance=chance,
        forward_accuracy=forward_accuracy,
        backward_accuracy=backward_accuracy,
        num_pairs=total_pairs,
        num_groups=total_groups,
    )


class HardSymmetricInfoNCE(nn.Module):
    """Module wrapper around :func:`hard_symmetric_info_nce`."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        *,
        prediction_bag_ids: Tensor,
        prediction_modality_ids: Tensor,
        prediction_pair_ids: Tensor | None = None,
        target_bag_ids: Tensor | None = None,
        target_modality_ids: Tensor | None = None,
        target_pair_ids: Tensor | None = None,
        prediction_scale_ids: Tensor | None = None,
        target_scale_ids: Tensor | None = None,
    ) -> MatchingOutput:
        return hard_symmetric_info_nce(
            predictions,
            targets,
            prediction_bag_ids=prediction_bag_ids,
            prediction_modality_ids=prediction_modality_ids,
            prediction_pair_ids=prediction_pair_ids,
            target_bag_ids=target_bag_ids,
            target_modality_ids=target_modality_ids,
            target_pair_ids=target_pair_ids,
            prediction_scale_ids=prediction_scale_ids,
            target_scale_ids=target_scale_ids,
            temperature=self.temperature,
        )
