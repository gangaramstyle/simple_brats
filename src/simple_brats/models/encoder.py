"""A small physical-coordinate ViT encoder for multimodal MRI patch bags."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import MillimetreRoPE, anchor_relative_coordinates


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for :class:`CrossModalEncoder`.

    ``patch_shape`` follows PyTorch's Conv3d spatial order.  The default keeps
    the project's established ``16 x 16 x 1`` slab representation.
    """

    num_modalities: int = 4
    in_channels: int = 1
    patch_shape: tuple[int, int, int] = (16, 16, 1)
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    min_wavelength_mm: float = 2.0
    max_wavelength_mm: float = 1024.0


class ConvPatchStem(nn.Module):
    """Shared patch tokenizer plus an explicit modality identity embedding.

    The same Conv3d weights process every modality.  Modality identity is
    injected only after content tokenization, and every input patch remains a
    separate token; this module performs no location-level token fusion.
    """

    def __init__(
        self,
        *,
        num_modalities: int,
        embed_dim: int,
        patch_shape: tuple[int, int, int] = (16, 16, 1),
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("num_modalities must be positive")
        if len(patch_shape) != 3 or any(size <= 0 for size in patch_shape):
            raise ValueError("patch_shape must contain three positive dimensions")
        self.num_modalities = num_modalities
        self.in_channels = in_channels
        self.patch_shape = tuple(patch_shape)
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_shape,
            stride=self.patch_shape,
        )
        self.modality_embedding = nn.Embedding(num_modalities, embed_dim)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def forward(self, patches: Tensor, modality_ids: Tensor) -> Tensor:
        """Tokenize patches shaped ``[B, N, (C), 16, 16, 1]``."""

        if modality_ids.ndim != 2:
            raise ValueError("modality_ids must have shape [batch, tokens]")
        batch, n_tokens = modality_ids.shape
        expected_implicit = (batch, n_tokens, *self.patch_shape)
        expected_explicit = (batch, n_tokens, self.in_channels, *self.patch_shape)
        if tuple(patches.shape) == expected_implicit:
            patches = patches.unsqueeze(2)
        elif tuple(patches.shape) != expected_explicit:
            raise ValueError(
                f"patches must have shape {expected_implicit} or {expected_explicit}, "
                f"got {tuple(patches.shape)}"
            )
        if modality_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("modality_ids must contain integer indices")
        if modality_ids.numel() and (
            int(modality_ids.min()) < 0 or int(modality_ids.max()) >= self.num_modalities
        ):
            raise ValueError(f"modality_ids must be in [0, {self.num_modalities})")

        content = self.projection(
            patches.reshape(batch * n_tokens, self.in_channels, *self.patch_shape)
        )
        content = content.flatten(1).reshape(batch, n_tokens, -1)
        return content + self.modality_embedding(modality_ids)


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float,
        min_wavelength_mm: float,
        max_wavelength_mm: float,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = embed_dim // num_heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.rope = MillimetreRoPE(
            head_dim,
            min_wavelength_mm=min_wavelength_mm,
            max_wavelength_mm=max_wavelength_mm,
        )

    def forward(
        self,
        tokens: Tensor,
        relative_coordinates_mm: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, n_tokens, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.rope(query, relative_coordinates_mm)
        key = self.rope(key, relative_coordinates_mm)

        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            if padding_mask.shape != (batch, n_tokens) or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean with shape [batch, tokens]")
            if padding_mask.all(dim=1).any():
                raise ValueError("each bag must contain at least one non-padding source token")
            logits = logits.masked_fill(
                padding_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )
        weights = self.attention_dropout(F.softmax(logits, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, n_tokens, self.embed_dim)
        return self.output_dropout(self.output(attended))


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embed_dim)
        self.attention = RotarySelfAttention(
            config.embed_dim,
            config.num_heads,
            dropout=config.dropout,
            min_wavelength_mm=config.min_wavelength_mm,
            max_wavelength_mm=config.max_wavelength_mm,
        )
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp_norm = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        tokens: Tensor,
        relative_coordinates_mm: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        tokens = tokens + self.attention(
            self.attention_norm(tokens), relative_coordinates_mm, padding_mask
        )
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask[..., None], 0)
        return tokens


class CrossModalEncoder(nn.Module):
    """Jointly contextualize separate modality-specific patch tokens.

    There are deliberately no CLS tokens, registers, absolute positions, or
    sequence-index embeddings.  The output has exactly one token for every
    source patch supplied by the caller, in the same order.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.patch_stem = ConvPatchStem(
            num_modalities=self.config.num_modalities,
            embed_dim=self.config.embed_dim,
            patch_shape=self.config.patch_shape,
            in_channels=self.config.in_channels,
        )
        self.blocks = nn.ModuleList(EncoderBlock(self.config) for _ in range(self.config.depth))
        self.output_norm = nn.LayerNorm(self.config.embed_dim)

    def forward(
        self,
        patches: Tensor,
        modality_ids: Tensor,
        coordinates_mm: Tensor,
        anchor_mm: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if coordinates_mm.shape != (*modality_ids.shape, 3):
            raise ValueError("coordinates_mm must have shape [batch, tokens, 3]")
        relative_coordinates_mm = anchor_relative_coordinates(coordinates_mm, anchor_mm)
        tokens = self.patch_stem(patches, modality_ids)
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask[..., None], 0)
        for block in self.blocks:
            tokens = block(tokens, relative_coordinates_mm, padding_mask)
        tokens = self.output_norm(tokens)
        if padding_mask is not None:
            tokens = tokens.masked_fill(padding_mask[..., None], 0)
        return tokens
