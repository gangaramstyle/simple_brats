"""End-to-end training contracts for the initial matching task."""

from .checkpoints import (
    ArtifactSink,
    CheckpointManager,
    CheckpointPolicy,
    WandbArtifactSink,
)
from .diagnostics import (
    CollapseThresholds,
    RepresentationStats,
    collapse_reasons,
    representation_stats,
    stats_by_modality,
)
from .matching import (
    CrossModalMatchingSystem,
    MatchingBatch,
    MatchingStepOutput,
    build_matching_system,
    optimizer_parameter_groups,
    validate_matching_batch,
)
from .runner import (
    RepresentationCollapseError,
    StepMetrics,
    TrainingResult,
    TrainingRunnerError,
    run_matching_training,
)
from .synthetic import make_synthetic_matching_batch, run_synthetic_smoke

__all__ = [
    "CrossModalMatchingSystem",
    "CollapseThresholds",
    "ArtifactSink",
    "CheckpointManager",
    "CheckpointPolicy",
    "MatchingBatch",
    "MatchingStepOutput",
    "RepresentationStats",
    "RepresentationCollapseError",
    "StepMetrics",
    "TrainingResult",
    "TrainingRunnerError",
    "WandbArtifactSink",
    "build_matching_system",
    "collapse_reasons",
    "make_synthetic_matching_batch",
    "optimizer_parameter_groups",
    "representation_stats",
    "run_synthetic_smoke",
    "run_matching_training",
    "stats_by_modality",
    "validate_matching_batch",
]
