"""Model components for cross-modality patch representation learning."""

from .encoder import ConvPatchStem, CrossModalEncoder, EncoderConfig
from .predictor import TargetModalityPredictor
from .rope import MillimetreRoPE, anchor_relative_coordinates, apply_rotary, build_mm_rope
from .teacher import BlindPatchTeacher, EMATeacher, EncoderStemPatchTeacher, update_ema_

__all__ = [
    "BlindPatchTeacher",
    "ConvPatchStem",
    "CrossModalEncoder",
    "EMATeacher",
    "EncoderStemPatchTeacher",
    "EncoderConfig",
    "MillimetreRoPE",
    "TargetModalityPredictor",
    "anchor_relative_coordinates",
    "apply_rotary",
    "build_mm_rope",
    "update_ema_",
]
