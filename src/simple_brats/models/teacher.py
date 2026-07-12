"""Position-blind patch target encoder and its exponential-moving average."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .encoder import CrossModalEncoder


class BlindPatchTeacher(nn.Module):
    """Encode each clean target patch independently using patch pixels only.

    The public ``forward`` method intentionally accepts exactly one value: a
    patch tensor.  It has no route for coordinates, modality/scale identity,
    scan statistics, target indices, neighbouring patches, or source context.
    A full-footprint Conv3d (rather than spatial averaging) retains learned
    within-patch geometry before the projection MLP.
    """

    def __init__(
        self,
        embedding_dim: int = 192,
        *,
        hidden_dim: int | None = None,
        in_channels: int = 1,
        patch_shape: tuple[int, int, int] = (8, 8, 8),
    ) -> None:
        super().__init__()
        if len(patch_shape) != 3 or any(size <= 0 for size in patch_shape):
            raise ValueError("patch_shape must contain three positive dimensions")
        hidden_dim = hidden_dim or embedding_dim
        self.in_channels = in_channels
        self.patch_shape = tuple(patch_shape)
        self.geometry_encoder = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=self.patch_shape,
            stride=self.patch_shape,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, embedding_dim),
        )

    def forward(self, patches: Tensor) -> Tensor:
        """Encode ``patches`` shaped ``[B, N, (C), *patch_shape]``."""

        if patches.ndim == 5:
            batch, n_patches = patches.shape[:2]
            expected = (batch, n_patches, *self.patch_shape)
            if tuple(patches.shape) != expected:
                raise ValueError(f"patches must have shape {expected}, got {tuple(patches.shape)}")
            patches = patches.unsqueeze(2)
        elif patches.ndim == 6:
            batch, n_patches = patches.shape[:2]
            expected = (batch, n_patches, self.in_channels, *self.patch_shape)
            if tuple(patches.shape) != expected:
                raise ValueError(f"patches must have shape {expected}, got {tuple(patches.shape)}")
        else:
            raise ValueError("patches must have shape [batch, patches, (channels), D, H, W]")

        features = self.geometry_encoder(
            patches.reshape(batch * n_patches, self.in_channels, *self.patch_shape)
        ).flatten(1)
        return self.projection(features).reshape(batch, n_patches, -1)


class EncoderStemPatchTeacher(nn.Module):
    """Patch-only view of the encoder weights used by the online student.

    The convolution is the *same module object* as the contextual encoder's
    content stem and is therefore trained through the source path.  Target
    normalization is fixed and non-affine.  :class:`EMATeacher` takes a
    detached copy for target construction.  Modality embeddings and all
    contextual transformer blocks are deliberately absent.
    """

    def __init__(self, geometry_encoder: nn.Conv3d, output_norm: nn.LayerNorm) -> None:
        super().__init__()
        if not isinstance(geometry_encoder, nn.Conv3d):
            raise TypeError("geometry_encoder must be a Conv3d patch projection")
        if not isinstance(output_norm, nn.LayerNorm):
            raise TypeError("output_norm must be a LayerNorm")
        kernel_size = tuple(int(value) for value in geometry_encoder.kernel_size)
        if len(kernel_size) != 3:
            raise ValueError("geometry_encoder must have a three-dimensional kernel")
        if tuple(geometry_encoder.stride) != kernel_size:
            raise ValueError("geometry_encoder must emit exactly one token per patch")
        if tuple(output_norm.normalized_shape) != (geometry_encoder.out_channels,):
            raise ValueError("output_norm width must match geometry_encoder output channels")
        self.geometry_encoder = geometry_encoder
        self.output_norm = output_norm
        self.in_channels = geometry_encoder.in_channels
        self.patch_shape = kernel_size

    @classmethod
    def from_encoder(cls, encoder: CrossModalEncoder) -> EncoderStemPatchTeacher:
        # Target normalization is deliberately non-affine.  Reusing the
        # contextual output LayerNorm's learned scale/bias would create a
        # second co-adapting route toward constant target embeddings.
        target_norm = nn.LayerNorm(encoder.config.embed_dim, elementwise_affine=False)
        return cls(encoder.patch_stem.projection, target_norm)

    def forward(self, patches: Tensor) -> Tensor:
        """Encode patches with no argument other than their normalized pixels."""

        if patches.ndim == 5:
            batch, n_patches = patches.shape[:2]
            expected = (batch, n_patches, *self.patch_shape)
            if tuple(patches.shape) != expected:
                raise ValueError(f"patches must have shape {expected}, got {tuple(patches.shape)}")
            patches = patches.unsqueeze(2)
        elif patches.ndim == 6:
            batch, n_patches = patches.shape[:2]
            expected = (batch, n_patches, self.in_channels, *self.patch_shape)
            if tuple(patches.shape) != expected:
                raise ValueError(f"patches must have shape {expected}, got {tuple(patches.shape)}")
        else:
            raise ValueError("patches must have shape [batch, patches, (channels), D, H, W]")

        features = self.geometry_encoder(
            patches.reshape(batch * n_patches, self.in_channels, *self.patch_shape)
        ).flatten(1)
        return self.output_norm(features).reshape(batch, n_patches, -1)


@torch.no_grad()
def update_ema_(target: nn.Module, online: nn.Module, momentum: float) -> None:
    """Update ``target`` parameters/buffers from ``online`` in place."""

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("momentum must lie in [0, 1]")
    target_parameters = dict(target.named_parameters())
    online_parameters = dict(online.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise ValueError("target and online models must have identical parameter names")
    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name]
        if target_parameter.shape != online_parameter.shape:
            raise ValueError(f"parameter shape mismatch for {name}")
        target_parameter.lerp_(online_parameter.detach().to(target_parameter), 1.0 - momentum)

    target_buffers = dict(target.named_buffers())
    online_buffers = dict(online.named_buffers())
    if target_buffers.keys() != online_buffers.keys():
        raise ValueError("target and online models must have identical buffer names")
    for name, target_buffer in target_buffers.items():
        online_buffer = online_buffers[name].detach().to(target_buffer)
        if target_buffer.is_floating_point():
            target_buffer.lerp_(online_buffer, 1.0 - momentum)
        else:
            target_buffer.copy_(online_buffer)


class EMATeacher(nn.Module):
    """Frozen EMA copy of an online :class:`BlindPatchTeacher`."""

    def __init__(self, online_teacher: nn.Module, momentum: float = 0.996) -> None:
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must lie in [0, 1]")
        self.momentum = momentum
        self.teacher = copy.deepcopy(online_teacher)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True) -> EMATeacher:
        """Keep the target network in evaluation mode when its parent trains."""

        super().train(False)
        return self

    @torch.no_grad()
    def update(self, online_teacher: nn.Module, momentum: float | None = None) -> None:
        update_ema_(self.teacher, online_teacher, self.momentum if momentum is None else momentum)
        self.num_updates.add_(1)

    @torch.no_grad()
    def forward(self, patches: Tensor) -> Tensor:
        return self.teacher(patches)
