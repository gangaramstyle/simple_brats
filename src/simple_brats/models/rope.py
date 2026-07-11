"""Rotary position encoding for physical MRI coordinates.

The coordinates supplied to the model are millimetres relative to a random
bag anchor.  RoPE is the *only* place spatial coordinates enter the encoder or
predictor: no absolute-coordinate or sequence-index embedding is added to a
token.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def anchor_relative_coordinates(coordinates_mm: Tensor, anchor_mm: Tensor) -> Tensor:
    """Express ``coordinates_mm`` relative to one anchor per batch item.

    Args:
        coordinates_mm: Tensor shaped ``[batch, tokens, axes]``.
        anchor_mm: Tensor shaped ``[batch, axes]`` or ``[batch, 1, axes]``.

    A common translation of the coordinates and anchor therefore leaves the
    returned coordinates unchanged.
    """

    if coordinates_mm.ndim != 3:
        raise ValueError(
            "coordinates_mm must have shape [batch, tokens, axes], "
            f"got {tuple(coordinates_mm.shape)}"
        )
    if anchor_mm.ndim == 2:
        anchor_mm = anchor_mm[:, None, :]
    if anchor_mm.ndim != 3 or anchor_mm.shape[1] != 1:
        raise ValueError(
            "anchor_mm must have shape [batch, axes] or [batch, 1, axes], "
            f"got {tuple(anchor_mm.shape)}"
        )
    if (
        anchor_mm.shape[0] != coordinates_mm.shape[0]
        or anchor_mm.shape[-1] != coordinates_mm.shape[-1]
    ):
        raise ValueError("coordinates_mm and anchor_mm must agree on batch size and axes")
    return coordinates_mm - anchor_mm.to(device=coordinates_mm.device, dtype=coordinates_mm.dtype)


def build_mm_rope(
    coordinates_mm: Tensor,
    head_dim: int,
    *,
    min_wavelength_mm: float = 2.0,
    max_wavelength_mm: float = 1024.0,
) -> tuple[Tensor, Tensor]:
    """Build cosine and sine rotations for arbitrary physical coordinates.

    Rotary pairs are assigned round-robin to coordinate axes.  This permits
    useful head sizes such as 32, which are even but not divisible by six.
    Wavelengths are log-spaced independently within each axis.
    """

    if coordinates_mm.ndim != 3:
        raise ValueError("coordinates_mm must have shape [batch, tokens, axes]")
    if not coordinates_mm.is_floating_point():
        raise TypeError("coordinates_mm must be floating point")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
    if min_wavelength_mm <= 0 or max_wavelength_mm < min_wavelength_mm:
        raise ValueError("wavelengths must satisfy 0 < min_wavelength_mm <= max_wavelength_mm")

    n_axes = coordinates_mm.shape[-1]
    if n_axes <= 0:
        raise ValueError("coordinates_mm must contain at least one axis")

    n_pairs = head_dim // 2
    device = coordinates_mm.device
    # Pair 0 rotates with axis 0, pair 1 with axis 1, and so on.  When
    # n_pairs is not divisible by n_axes, the first axes get one extra pair.
    pair_index = torch.arange(n_pairs, device=device)
    axis_index = pair_index.remainder(n_axes)
    rank_within_axis = torch.div(pair_index, n_axes, rounding_mode="floor")
    pair_counts = torch.bincount(axis_index, minlength=n_axes)
    denominator = (pair_counts[axis_index] - 1).clamp_min(1)
    fraction = rank_within_axis.to(torch.float32) / denominator.to(torch.float32)
    wavelengths = min_wavelength_mm * (max_wavelength_mm / min_wavelength_mm) ** fraction
    angular_frequency = (2.0 * math.pi / wavelengths).to(dtype=coordinates_mm.dtype)

    selected_coordinates = coordinates_mm[..., axis_index]
    angles = selected_coordinates * angular_frequency
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Apply interleaved rotary pairs to ``x``.

    ``x`` is normally ``[batch, heads, tokens, head_dim]`` while cosine and
    sine are ``[batch, tokens, head_dim / 2]``.  The head dimension is
    broadcast without allocating a copy.
    """

    if x.shape[-1] % 2:
        raise ValueError("the final dimension of x must be even")
    if cosine.shape != sine.shape or cosine.shape[-1] * 2 != x.shape[-1]:
        raise ValueError("cosine and sine must match x's rotary-pair dimensions")
    if x.ndim != 4 or cosine.ndim != 3:
        raise ValueError(
            "expected x [batch, heads, tokens, head_dim] and rotations "
            "[batch, tokens, head_dim / 2]"
        )
    if cosine.shape[0] != x.shape[0] or cosine.shape[1] != x.shape[2]:
        raise ValueError("rotations must agree with x on batch and token dimensions")

    cosine = cosine[:, None].to(device=x.device, dtype=x.dtype)
    sine = sine[:, None].to(device=x.device, dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class MillimetreRoPE(nn.Module):
    """Stateless module form of :func:`build_mm_rope` + :func:`apply_rotary`."""

    def __init__(
        self,
        head_dim: int,
        *,
        min_wavelength_mm: float = 2.0,
        max_wavelength_mm: float = 1024.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        self.head_dim = head_dim
        self.min_wavelength_mm = min_wavelength_mm
        self.max_wavelength_mm = max_wavelength_mm

    def forward(self, x: Tensor, coordinates_mm: Tensor) -> Tensor:
        cosine, sine = build_mm_rope(
            coordinates_mm,
            self.head_dim,
            min_wavelength_mm=self.min_wavelength_mm,
            max_wavelength_mm=self.max_wavelength_mm,
        )
        return apply_rotary(x, cosine, sine)
