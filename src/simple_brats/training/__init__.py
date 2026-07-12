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
    PREDICTION_DIAGNOSTIC_STREAM,
    TEACHER_TARGET_DIAGNOSTIC_STREAM,
    TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM,
    FixedTargetPatchProbe,
    RepresentationCollapseError,
    StepMetrics,
    TrainingResult,
    TrainingRunnerError,
    preserve_runner_rng_state,
    run_matching_training,
)
from .runtime import (
    TrainingRuntimeError,
    TrainingRuntimePolicy,
    apply_model_runtime,
    build_adamw_optimizer,
    configure_training_runtime,
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
    "FixedTargetPatchProbe",
    "RepresentationStats",
    "RepresentationCollapseError",
    "PREDICTION_DIAGNOSTIC_STREAM",
    "StepMetrics",
    "TrainingResult",
    "TrainingRunnerError",
    "TrainingRuntimeError",
    "TrainingRuntimePolicy",
    "preserve_runner_rng_state",
    "TEACHER_TARGET_DIAGNOSTIC_STREAM",
    "TRAINING_TEACHER_TARGET_DIAGNOSTIC_STREAM",
    "WandbArtifactSink",
    "build_matching_system",
    "apply_model_runtime",
    "build_adamw_optimizer",
    "collapse_reasons",
    "configure_training_runtime",
    "make_synthetic_matching_batch",
    "optimizer_parameter_groups",
    "representation_stats",
    "run_synthetic_smoke",
    "run_matching_training",
    "stats_by_modality",
    "validate_matching_batch",
]
