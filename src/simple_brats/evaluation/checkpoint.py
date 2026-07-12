"""Extract singleton online-encoder tokens from a runner-v3 checkpoint."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from simple_brats.config import MODALITIES, ExperimentConfig
from simple_brats.data.case_grids import CaseGridManifest
from simple_brats.data.extraction import patch_interpolation_support
from simple_brats.data.manifest import CaseRecord, DatasetManifest, sha256_file
from simple_brats.data.pipeline import CachedNiftiPatchExtractor
from simple_brats.data.splits import SplitManifest, partition_cases, validate_split
from simple_brats.long_run import SubjectBalancedSchedule
from simple_brats.sampling import SlabGeometry
from simple_brats.training.matching import build_matching_system

from .patches import EvaluationPatchManifest, EvaluationPatchRecord
from .probes import (
    FrozenJointTable,
    FrozenTokenTable,
    evaluate_frozen_tokens,
    evaluate_joint_frozen_features,
)

RUNNER_CHECKPOINT_CONTAINER_SCHEMA_VERSION = 1
RUNNER_CHECKPOINT_STATE_SCHEMA_VERSION = 3


class CheckpointEvaluationError(ValueError):
    """Checkpoint or extracted-token provenance is incompatible with evaluation."""


def configure_deterministic_evaluation_runtime(
    device: torch.device,
) -> dict[str, object]:
    """Fail closed on the deterministic policy used for checkpoint comparisons."""

    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise CheckpointEvaluationError(
            "CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python starts"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.are_deterministic_algorithms_enabled():
        raise CheckpointEvaluationError("Torch deterministic algorithms were not retained")
    return {
        "schema": "simple-brats.deterministic-checkpoint-evaluation-runtime",
        "schema_version": 1,
        "device_type": device.type,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": (":4096:8" if device.type == "cuda" else "not_applicable"),
    }


class SingletonOnlineTokenEncoder(nn.Module):
    """Patch/modality-only view of the trained online contextual encoder.

    Each invocation forms independent one-token bags.  The fixed relative
    coordinate is identically zero and is created internally, so neither the
    caller nor the downstream classifier can provide position or context.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, patches: Tensor, modality_ids: Tensor) -> Tensor:
        if patches.ndim != 4:
            raise CheckpointEvaluationError("singleton patches must have shape [samples, D, H, W]")
        if modality_ids.ndim != 1 or modality_ids.numel() != patches.shape[0]:
            raise CheckpointEvaluationError("modality IDs must align with singleton patches")
        batch = patches.shape[0]
        coordinates = torch.zeros((batch, 1, 3), dtype=patches.dtype, device=patches.device)
        anchors = torch.zeros((batch, 3), dtype=patches.dtype, device=patches.device)
        output = self.encoder(
            patches[:, None],
            modality_ids[:, None],
            coordinates,
            anchors,
        )
        if output.ndim != 3 or output.shape[:2] != (batch, 1):
            raise CheckpointEvaluationError("online encoder violated singleton token shape")
        return output[:, 0]


class ColocatedFourModalityTokenEncoder(nn.Module):
    """Encode a co-located canonical four-token bag and concatenate its outputs.

    The online encoder may use learned cross-modality attention, which is the
    intended downstream joint view.  Coordinates and anchors are fixed to zero
    internally, no neighboring locations are present, and the four output
    tokens remain separate until a fixed-order concatenation after encoding.
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, patches: Tensor) -> Tensor:
        if patches.ndim != 5 or patches.shape[1] != len(MODALITIES):
            raise CheckpointEvaluationError(
                "co-located patches must have shape [samples, 4, D, H, W]"
            )
        batch = patches.shape[0]
        modality_ids = torch.arange(len(MODALITIES), dtype=torch.long, device=patches.device)[
            None
        ].expand(batch, -1)
        coordinates = torch.zeros(
            (batch, len(MODALITIES), 3), dtype=patches.dtype, device=patches.device
        )
        anchors = torch.zeros((batch, 3), dtype=patches.dtype, device=patches.device)
        output = self.encoder(patches, modality_ids, coordinates, anchors)
        if output.ndim != 3 or output.shape[:2] != (batch, len(MODALITIES)):
            raise CheckpointEvaluationError("online encoder violated co-located token shape")
        return output.flatten(1)


@dataclass(frozen=True, slots=True)
class LoadedCheckpointEncoder:
    encoder: SingletonOnlineTokenEncoder
    step: int
    checkpoint_sha256: str
    provenance: Mapping[str, object]
    consumed_ssl_train_subject_count: int
    total_ssl_train_subject_count: int
    deterministic_runtime: Mapping[str, object]

    @property
    def complete_ssl_train_subject_coverage(self) -> bool:
        return self.consumed_ssl_train_subject_count == self.total_ssl_train_subject_count


def _required_provenance_sha(provenance: Mapping[str, object], name: str, expected: str) -> None:
    if provenance.get(name) != expected:
        raise CheckpointEvaluationError(
            f"checkpoint {name} differs from the held-out evaluation input"
        )


def _declared_train_subjects(provenance: Mapping[str, object]) -> tuple[str, ...]:
    selected_train_subjects = provenance.get("selected_train_subject_ids")
    selected_subjects = provenance.get("selected_subject_ids")
    if selected_train_subjects is not None and selected_subjects is not None:
        if selected_train_subjects != selected_subjects:
            raise CheckpointEvaluationError(
                "checkpoint subject-consumption provenance fields disagree"
            )
    declared = selected_train_subjects if selected_train_subjects is not None else selected_subjects
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(subject, str) or not subject for subject in declared)
        or len(set(declared)) != len(declared)
    ):
        raise CheckpointEvaluationError(
            "checkpoint must enumerate unique subjects in its SSL training schedule"
        )
    return tuple(declared)


def _subjects_exposed_by_checkpoint(
    provenance: Mapping[str, object],
    *,
    step: int,
    train_cases: Sequence[CaseRecord],
    config: ExperimentConfig,
    declared_subjects: tuple[str, ...],
) -> set[str]:
    """Derive actual prefix exposure instead of trusting a static cohort declaration."""

    schema = provenance.get("schema")
    schedule_record = provenance.get("schedule")
    if schema == "simple-brats.long-real-matching":
        if not isinstance(schedule_record, Mapping):
            raise CheckpointEvaluationError("long checkpoint has no subject schedule record")
        bags = schedule_record.get("bags_per_subject")
        schedule_sha = schedule_record.get("subject_schedule_sha256")
        if isinstance(bags, bool) or not isinstance(bags, int) or bags <= 0:
            raise CheckpointEvaluationError("long checkpoint has an invalid bags-per-subject value")
        schedule = SubjectBalancedSchedule(train_cases, seed=config.seed, bags_per_subject=bags)
        if schedule.sha256 != schedule_sha:
            raise CheckpointEvaluationError(
                "long checkpoint subject schedule digest is inconsistent"
            )
        if set(declared_subjects) != set(schedule.subject_ids):
            raise CheckpointEvaluationError(
                "long checkpoint scheduled-subject declaration is incomplete"
            )
        completed_blocks = (step + bags - 1) // bags
        return {
            schedule.assignment_for_step(block * bags).subject_id
            for block in range(completed_blocks)
        }

    if schema == "simple-brats.short-real-matching":
        if not isinstance(schedule_record, Mapping):
            raise CheckpointEvaluationError("short checkpoint has no case schedule record")
        bags = schedule_record.get("bags_per_case")
        case_ids = provenance.get("selected_train_case_ids")
        if (
            isinstance(bags, bool)
            or not isinstance(bags, int)
            or bags <= 0
            or not isinstance(case_ids, list)
            or not case_ids
            or any(not isinstance(case_id, str) for case_id in case_ids)
            or len(set(case_ids)) != len(case_ids)
        ):
            raise CheckpointEvaluationError("short checkpoint has an invalid case schedule")
        cases_by_id: dict[str, CaseRecord] = {}
        for case in train_cases:
            if case.case_id in cases_by_id:
                raise CheckpointEvaluationError("train case IDs are not globally unique")
            cases_by_id[case.case_id] = case
        try:
            scheduled_cases = tuple(cases_by_id[case_id] for case_id in case_ids)
        except KeyError as error:
            raise CheckpointEvaluationError(
                "short checkpoint scheduled a case outside the train split"
            ) from error
        if set(declared_subjects) != {case.subject_id for case in scheduled_cases}:
            raise CheckpointEvaluationError(
                "short checkpoint subject and case schedule declarations disagree"
            )
        completed_blocks = (step + bags - 1) // bags
        return {
            scheduled_cases[block % len(scheduled_cases)].subject_id
            for block in range(completed_blocks)
        }

    # Compatibility for older checkpoint schemas. New registered long/short
    # runs always use one of the exact step-aware paths above.
    return set(declared_subjects)


def load_online_encoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    split: SplitManifest,
    case_grids: CaseGridManifest,
    evaluation_patches: EvaluationPatchManifest,
    device: str | torch.device,
    require_all_ssl_train_subjects: bool = True,
) -> LoadedCheckpointEncoder:
    """Strictly load runner-v3 state and return only ``system.encoder``."""

    if not isinstance(require_all_ssl_train_subjects, bool):
        raise TypeError("require_all_ssl_train_subjects must be boolean")
    resolved_device = torch.device(device)
    deterministic_runtime = configure_deterministic_evaluation_runtime(resolved_device)
    validate_split(manifest, split)
    case_grids.validate_manifest(manifest)
    if evaluation_patches.data_manifest_sha256 != manifest.sha256:
        raise CheckpointEvaluationError("evaluation patches use a different data manifest")
    if evaluation_patches.subject_split_sha256 != split.sha256:
        raise CheckpointEvaluationError("evaluation patches use a different subject split")
    if evaluation_patches.case_grid_manifest_sha256 != case_grids.sha256:
        raise CheckpointEvaluationError("evaluation patches use a different case-grid manifest")
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != RUNNER_CHECKPOINT_CONTAINER_SCHEMA_VERSION
    ):
        raise CheckpointEvaluationError("unsupported runner checkpoint container")
    state = payload.get("state")
    provenance = payload.get("metadata")
    if (
        not isinstance(state, Mapping)
        or state.get("runner_schema_version") != RUNNER_CHECKPOINT_STATE_SCHEMA_VERSION
    ):
        raise CheckpointEvaluationError("evaluation requires runner state schema version 3")
    if not isinstance(provenance, Mapping) or state.get("provenance") != provenance:
        raise CheckpointEvaluationError("checkpoint provenance records are missing or inconsistent")
    step = state.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or payload.get("step") != step
    ):
        raise CheckpointEvaluationError("checkpoint step is invalid or inconsistent")
    _required_provenance_sha(provenance, "manifest_sha256", manifest.sha256)
    _required_provenance_sha(provenance, "split_sha256", split.sha256)
    _required_provenance_sha(provenance, "case_grid_manifest_sha256", case_grids.sha256)
    _required_provenance_sha(provenance, "config_sha256", config.sha256)

    partitions = partition_cases(manifest, split)
    train_subjects = {case.subject_id for case in partitions["train"]}
    validation_subjects = {case.subject_id for case in partitions["validation"]}
    test_subjects = {case.subject_id for case in partitions["test"]}
    declared_subjects = _declared_train_subjects(provenance)
    declared_subject_set = set(declared_subjects)
    if declared_subject_set - train_subjects:
        raise CheckpointEvaluationError("checkpoint consumed subjects outside the train split")
    if declared_subject_set & (validation_subjects | test_subjects):
        raise CheckpointEvaluationError("checkpoint consumed validation or locked-test subjects")
    consumed_subjects = _subjects_exposed_by_checkpoint(
        provenance,
        step=step,
        train_cases=partitions["train"],
        config=config,
        declared_subjects=declared_subjects,
    )
    if not consumed_subjects or consumed_subjects - declared_subject_set:
        raise CheckpointEvaluationError("checkpoint exposure prefix is invalid")
    if require_all_ssl_train_subjects and consumed_subjects != train_subjects:
        missing = len(train_subjects - consumed_subjects)
        raise CheckpointEvaluationError(
            f"representation evaluation requires all SSL-train subjects; {missing} are missing"
        )
    if set(evaluation_patches.probe_train_subjects) - train_subjects:
        raise CheckpointEvaluationError("probe-train labels are outside the SSL train split")
    if set(evaluation_patches.validation_subjects) != validation_subjects:
        raise CheckpointEvaluationError(
            "evaluation patch manifest must cover every validation subject exactly"
        )

    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise CheckpointEvaluationError("checkpoint contains no model state mapping")
    system = build_matching_system(config)
    try:
        system.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise CheckpointEvaluationError("checkpoint model state differs from config") from error
    # The EMA target and predictor remain in the discarded system; only the
    # online contextual encoder is reachable through this evaluation object.
    online_encoder = system.encoder.to(resolved_device).eval().requires_grad_(False)
    return LoadedCheckpointEncoder(
        encoder=SingletonOnlineTokenEncoder(online_encoder),
        step=step,
        checkpoint_sha256=sha256_file(path),
        provenance=dict(provenance),
        consumed_ssl_train_subject_count=len(consumed_subjects),
        total_ssl_train_subject_count=len(train_subjects),
        deterministic_runtime=deterministic_runtime,
    )


def build_random_online_encoder(
    config: ExperimentConfig,
    *,
    seed: int,
    device: str | torch.device,
) -> SingletonOnlineTokenEncoder:
    """Create the architecture-matched, deterministic random-encoder control."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CheckpointEvaluationError("random encoder seed must be non-negative")
    resolved_device = torch.device(device)
    configure_deterministic_evaluation_runtime(resolved_device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        system = build_matching_system(config)
    return SingletonOnlineTokenEncoder(system.encoder.to(resolved_device))


def _case_lookup(manifest: DatasetManifest) -> dict[tuple[str, str, str], CaseRecord]:
    return {case.key: case for case in manifest.cases}


def _extract_raw_crop(
    extractor: CachedNiftiPatchExtractor,
    case: CaseRecord,
    modality: str,
    center_mm: tuple[float, float, float],
) -> Tensor:
    volume = extractor.canonical_volumes_for_case(case)[modality]
    support = patch_interpolation_support(extractor.extraction_spec, center_mm)
    slices = tuple(
        slice(start, stop) for start, stop in zip(support.start_ijk, support.stop_ijk, strict=True)
    )
    crop = volume.data[slices]
    foreground = volume.foreground_mask[slices]
    if crop.shape != (4, 4, 4) or not bool(foreground.all()) or not np.isfinite(crop).all():
        raise CheckpointEvaluationError("raw 4mm control crop violated foreground geometry")
    return torch.from_numpy(np.array(crop, dtype=np.float32, order="C")).flatten()


def _encode_pending(
    patches: Sequence[Tensor],
    modality_ids: Sequence[int],
    *,
    trained_encoder: SingletonOnlineTokenEncoder,
    random_encoder: SingletonOnlineTokenEncoder,
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    values = torch.stack(tuple(patches)).to(device)
    ids = torch.tensor(tuple(modality_ids), dtype=torch.long, device=device)
    return (
        tuple(trained_encoder(values, ids).cpu().unbind()),
        tuple(random_encoder(values, ids).cpu().unbind()),
    )


def _encode_joint_pending(
    patches: Sequence[Tensor],
    *,
    trained_encoder: ColocatedFourModalityTokenEncoder,
    random_encoder: ColocatedFourModalityTokenEncoder,
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    values = torch.stack(tuple(patches)).to(device)
    return (
        tuple(trained_encoder(values).cpu().unbind()),
        tuple(random_encoder(values).cpu().unbind()),
    )


def extract_evaluation_feature_tables(
    *,
    data_root: str | Path,
    manifest: DatasetManifest,
    case_grids: CaseGridManifest,
    evaluation_patches: EvaluationPatchManifest,
    trained_encoder: SingletonOnlineTokenEncoder,
    random_encoder: SingletonOnlineTokenEncoder,
    device: str | torch.device,
    batch_size: int = 256,
) -> dict[str, FrozenTokenTable | FrozenJointTable]:
    """Extract primary tokens plus explicitly secondary random/raw controls."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise CheckpointEvaluationError("batch_size must be a positive integer")
    if evaluation_patches.data_manifest_sha256 != manifest.sha256:
        raise CheckpointEvaluationError("evaluation patches differ from the data manifest")
    case_grids.validate_manifest(manifest)
    cases = _case_lookup(manifest)
    records_by_case: dict[tuple[str, str, str], list[EvaluationPatchRecord]] = defaultdict(list)
    for record in evaluation_patches.records:
        records_by_case[(record.source, record.release, record.case_id)].append(record)

    trained_features: list[Tensor] = []
    random_features: list[Tensor] = []
    raw_features: list[Tensor] = []
    trained_joint_features: list[Tensor] = []
    random_joint_features: list[Tensor] = []
    raw_joint_features: list[Tensor] = []
    labels: list[int] = []
    modality_ids: list[int] = []
    subject_ids: list[str] = []
    partitions: list[str] = []
    sample_ids: list[str] = []
    joint_labels: list[int] = []
    joint_subject_ids: list[str] = []
    joint_partitions: list[str] = []
    joint_sample_ids: list[str] = []
    resolved_device = torch.device(device)
    trained_joint_encoder = ColocatedFourModalityTokenEncoder(trained_encoder.encoder)
    random_joint_encoder = ColocatedFourModalityTokenEncoder(random_encoder.encoder)
    for key in sorted(records_by_case):
        case = cases.get(key)
        if case is None:
            raise CheckpointEvaluationError("evaluation record case is absent from the manifest")
        spec = case_grids.extraction_spec_for_case(
            case, patch_config=evaluation_patches.patch_config
        )
        extractor = CachedNiftiPatchExtractor(
            data_root=data_root,
            manifest=manifest,
            data_manifest_sha256=manifest.sha256,
            extraction_spec=spec,
        )
        files = {file.modality: file for file in case.files}
        pending_patches: list[Tensor] = []
        pending_modalities: list[int] = []
        pending_joint_patches: list[Tensor] = []

        for record in sorted(records_by_case[key], key=lambda item: item.sample_id):
            location_patches: list[Tensor] = []
            location_raw: list[Tensor] = []
            for modality_id, modality in enumerate(MODALITIES):
                file = files[modality]
                patch = extractor(
                    path=file.path,
                    file_sha256=file.sha256,
                    modality=modality,
                    center_mm=record.center_mm,
                    geometry=SlabGeometry(
                        in_plane_footprint_mm=4.0,
                        thin_extent_mm=4.0,
                        model_shape=(16, 16, 16),
                    ),
                )
                pending_patches.append(patch)
                pending_modalities.append(modality_id)
                raw = _extract_raw_crop(extractor, case, modality, record.center_mm)
                raw_features.append(raw)
                location_patches.append(patch)
                location_raw.append(raw)
                labels.append(record.label)
                modality_ids.append(modality_id)
                subject_ids.append(record.subject_id)
                partitions.append(record.partition)
                sample_ids.append(f"{record.sample_id}:{modality}")
                if len(pending_patches) == batch_size:
                    trained, random = _encode_pending(
                        pending_patches,
                        pending_modalities,
                        trained_encoder=trained_encoder,
                        random_encoder=random_encoder,
                        device=resolved_device,
                    )
                    trained_features.extend(trained)
                    random_features.extend(random)
                    pending_patches.clear()
                    pending_modalities.clear()
            pending_joint_patches.append(torch.stack(location_patches))
            raw_joint_features.append(torch.cat(location_raw))
            joint_labels.append(record.label)
            joint_subject_ids.append(record.subject_id)
            joint_partitions.append(record.partition)
            joint_sample_ids.append(record.sample_id)
            if len(pending_joint_patches) == batch_size:
                trained, random = _encode_joint_pending(
                    pending_joint_patches,
                    trained_encoder=trained_joint_encoder,
                    random_encoder=random_joint_encoder,
                    device=resolved_device,
                )
                trained_joint_features.extend(trained)
                random_joint_features.extend(random)
                pending_joint_patches.clear()
        if pending_patches:
            trained, random = _encode_pending(
                pending_patches,
                pending_modalities,
                trained_encoder=trained_encoder,
                random_encoder=random_encoder,
                device=resolved_device,
            )
            trained_features.extend(trained)
            random_features.extend(random)
        if pending_joint_patches:
            trained, random = _encode_joint_pending(
                pending_joint_patches,
                trained_encoder=trained_joint_encoder,
                random_encoder=random_joint_encoder,
                device=resolved_device,
            )
            trained_joint_features.extend(trained)
            random_joint_features.extend(random)

    metadata = {
        "labels": torch.tensor(labels, dtype=torch.long),
        "modality_ids": torch.tensor(modality_ids, dtype=torch.long),
        "subject_ids": tuple(subject_ids),
        "partitions": tuple(partitions),
        "sample_ids": tuple(sample_ids),
    }
    joint_metadata = {
        "labels": torch.tensor(joint_labels, dtype=torch.long),
        "subject_ids": tuple(joint_subject_ids),
        "partitions": tuple(joint_partitions),
        "sample_ids": tuple(joint_sample_ids),
    }
    return {
        "primary_trained_online_singleton_token": FrozenTokenTable(
            features=torch.stack(trained_features), **metadata
        ),
        "control_random_online_singleton_token": FrozenTokenTable(
            features=torch.stack(random_features), **metadata
        ),
        "control_raw_normalized_4x4x4": FrozenTokenTable(
            features=torch.stack(raw_features), **metadata
        ),
        "primary_trained_online_colocated_four_token_concat": FrozenJointTable(
            features=torch.stack(trained_joint_features), **joint_metadata
        ),
        "control_random_online_colocated_four_token_concat": FrozenJointTable(
            features=torch.stack(random_joint_features), **joint_metadata
        ),
        "control_raw_normalized_four_modality_concat": FrozenJointTable(
            features=torch.stack(raw_joint_features), **joint_metadata
        ),
    }


def evaluate_checkpoint_feature_tables(
    tables: Mapping[str, FrozenTokenTable | FrozenJointTable],
    *,
    evaluation_patches: EvaluationPatchManifest,
    subject_budgets: Sequence[int],
    l2_penalty: float = 1.0,
    neighbors: Sequence[int] = (1, 5, 20),
) -> dict[str, object]:
    expected = {
        "primary_trained_online_singleton_token",
        "control_random_online_singleton_token",
        "control_raw_normalized_4x4x4",
        "primary_trained_online_colocated_four_token_concat",
        "control_random_online_colocated_four_token_concat",
        "control_raw_normalized_four_modality_concat",
    }
    if set(tables) != expected:
        raise CheckpointEvaluationError("feature tables must contain primary and both controls")
    singleton_names = {
        "primary_trained_online_singleton_token",
        "control_random_online_singleton_token",
        "control_raw_normalized_4x4x4",
    }
    singleton_reports = {
        name: evaluate_frozen_tokens(
            tables[name],  # type: ignore[arg-type]
            ordered_probe_train_subjects=evaluation_patches.probe_train_subjects,
            subject_budgets=subject_budgets,
            l2_penalty=l2_penalty,
            neighbors=neighbors,
        )
        for name in singleton_names
    }
    joint_names = {
        "primary_trained_online_colocated_four_token_concat",
        "control_random_online_colocated_four_token_concat",
        "control_raw_normalized_four_modality_concat",
    }
    joint_reports = {
        name: evaluate_joint_frozen_features(
            tables[name],  # type: ignore[arg-type]
            ordered_probe_train_subjects=evaluation_patches.probe_train_subjects,
            subject_budgets=subject_budgets,
            l2_penalty=l2_penalty,
            neighbors=neighbors,
        )
        for name in joint_names
    }
    return {
        "schema": "simple-brats.checkpoint-representation-evaluation",
        "schema_version": 1,
        "primary": {
            "joint_colocated_four_token_concat": joint_reports[
                "primary_trained_online_colocated_four_token_concat"
            ],
            "singleton_modality_isolation": singleton_reports[
                "primary_trained_online_singleton_token"
            ],
        },
        "controls": {
            "random_architecture_matched_encoder": {
                "joint_colocated_four_token_concat": joint_reports[
                    "control_random_online_colocated_four_token_concat"
                ],
                "singleton_modality_isolation": singleton_reports[
                    "control_random_online_singleton_token"
                ],
            },
            "raw_normalized_4x4x4_affine_knn": {
                "canonical_four_modality_concat": joint_reports[
                    "control_raw_normalized_four_modality_concat"
                ],
                "per_modality": singleton_reports["control_raw_normalized_4x4x4"],
            },
        },
        "interpretation_contract": (
            "controls are secondary baselines and are never inputs to the primary token probe"
        ),
    }


__all__ = [
    "RUNNER_CHECKPOINT_CONTAINER_SCHEMA_VERSION",
    "RUNNER_CHECKPOINT_STATE_SCHEMA_VERSION",
    "CheckpointEvaluationError",
    "ColocatedFourModalityTokenEncoder",
    "LoadedCheckpointEncoder",
    "SingletonOnlineTokenEncoder",
    "build_random_online_encoder",
    "configure_deterministic_evaluation_runtime",
    "evaluate_checkpoint_feature_tables",
    "extract_evaluation_feature_tables",
    "load_online_encoder_checkpoint",
]
