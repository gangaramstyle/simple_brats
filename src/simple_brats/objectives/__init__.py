"""Self-supervised training objectives."""

from .matching import HardSymmetricInfoNCE, MatchingOutput, hard_symmetric_info_nce

__all__ = ["HardSymmetricInfoNCE", "MatchingOutput", "hard_symmetric_info_nce"]
