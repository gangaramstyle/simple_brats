"""Shallow target-modality-conditioned predictor for blind patch targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import MillimetreRoPE, anchor_relative_coordinates


class RotaryCrossAttention(nn.Module):
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.scale = self.head_dim**-0.5
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_value_projection = nn.Linear(embed_dim, 2 * embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.rope = MillimetreRoPE(
            self.head_dim,
            min_wavelength_mm=min_wavelength_mm,
            max_wavelength_mm=max_wavelength_mm,
        )

    def forward(
        self,
        queries: Tensor,
        source_tokens: Tensor,
        query_coordinates_mm: Tensor,
        source_coordinates_mm: Tensor,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        batch, n_queries, _ = queries.shape
        n_sources = source_tokens.shape[1]
        query = (
            self.query_projection(queries)
            .reshape(batch, n_queries, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_value = (
            self.key_value_projection(source_tokens)
            .reshape(batch, n_sources, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        key, value = key_value.unbind(0)
        query = self.rope(query, query_coordinates_mm)
        key = self.rope(key, source_coordinates_mm)

        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if source_padding_mask is not None:
            if (
                source_padding_mask.shape != (batch, n_sources)
                or source_padding_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "source_padding_mask must be boolean with shape [batch, source tokens]"
                )
            if source_padding_mask.all(dim=1).any():
                raise ValueError("each bag must contain at least one non-padding source token")
            logits = logits.masked_fill(
                source_padding_mask[:, None, None, :], torch.finfo(logits.dtype).min
            )
        weights = self.attention_dropout(F.softmax(logits, dim=-1))
        attended = (
            torch.matmul(weights, value).transpose(1, 2).reshape(batch, n_queries, self.embed_dim)
        )
        return self.output_dropout(self.output_projection(attended))


class PredictorBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        dropout: float,
        min_wavelength_mm: float,
        max_wavelength_mm: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.source_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = RotaryCrossAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            min_wavelength_mm=min_wavelength_mm,
            max_wavelength_mm=max_wavelength_mm,
        )
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        source_tokens: Tensor,
        query_coordinates_mm: Tensor,
        source_coordinates_mm: Tensor,
        source_padding_mask: Tensor | None,
    ) -> Tensor:
        queries = queries + self.cross_attention(
            self.query_norm(queries),
            self.source_norm(source_tokens),
            query_coordinates_mm,
            source_coordinates_mm,
            source_padding_mask,
        )
        return queries + self.mlp(self.mlp_norm(queries))


class TargetModalityPredictor(nn.Module):
    """Predict blind target embeddings from contextualized source tokens.

    Query content is only a learned *target modality* embedding.  Its location
    enters solely through cross-attention RoPE.  The default is intentionally
    one block so the encoder, rather than a deep decoder, must carry useful
    semantic information.
    """

    def __init__(
        self,
        *,
        num_modalities: int = 4,
        embed_dim: int = 192,
        output_dim: int | None = None,
        depth: int = 1,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        min_wavelength_mm: float = 2.0,
        max_wavelength_mm: float = 1024.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.num_modalities = num_modalities
        self.target_modality_embedding = nn.Embedding(num_modalities, embed_dim)
        nn.init.normal_(self.target_modality_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            PredictorBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                min_wavelength_mm=min_wavelength_mm,
                max_wavelength_mm=max_wavelength_mm,
            )
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(
        self,
        source_tokens: Tensor,
        source_coordinates_mm: Tensor,
        query_coordinates_mm: Tensor,
        target_modality_ids: Tensor,
        anchor_mm: Tensor,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape [batch, source tokens, embedding]")
        batch, n_sources = source_tokens.shape[:2]
        if source_coordinates_mm.shape != (batch, n_sources, 3):
            raise ValueError("source_coordinates_mm must have shape [batch, source tokens, 3]")
        if target_modality_ids.ndim != 2 or target_modality_ids.shape[0] != batch:
            raise ValueError("target_modality_ids must have shape [batch, queries]")
        n_queries = target_modality_ids.shape[1]
        if query_coordinates_mm.shape != (batch, n_queries, 3):
            raise ValueError("query_coordinates_mm must have shape [batch, queries, 3]")
        if target_modality_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("target_modality_ids must contain integer indices")
        # The owning matching system performs this bounds check before calling
        # the registered compiled predictor.  Avoid scalar GPU reads inside the
        # compiled graph while preserving standalone eager validation.
        if (
            not torch.compiler.is_compiling()
            and target_modality_ids.numel()
            and (
                int(target_modality_ids.min()) < 0
                or int(target_modality_ids.max()) >= self.num_modalities
            )
        ):
            raise ValueError(f"target_modality_ids must be in [0, {self.num_modalities})")

        source_relative = anchor_relative_coordinates(source_coordinates_mm, anchor_mm)
        query_relative = anchor_relative_coordinates(query_coordinates_mm, anchor_mm)
        queries = self.target_modality_embedding(target_modality_ids)
        for block in self.blocks:
            queries = block(
                queries,
                source_tokens,
                query_relative,
                source_relative,
                source_padding_mask,
            )
        return self.output_projection(self.output_norm(queries))
