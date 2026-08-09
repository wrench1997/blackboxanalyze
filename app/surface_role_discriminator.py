"""Generic response-shape features and a calibrated surface-role head.

This head is intentionally separate from the vulnerability-family decoder.  It
receives only bounded structural shape (content-type class, parser counts,
header count, status class, and length buckets); it never receives an oracle
field, family label, route token, marker value, or evaluator state.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn


SURFACE_ROLES = (
    "reflected_attribute",
    "reflected_text",
    "json_echo",
    "header_echo",
    "plain_control",
)
SURFACE_ROLE_FEATURE_DIM = 25


def _bucket(value: Any, limits: tuple[int, ...]) -> int:
    try:
        number = max(0, int(value))
    except (TypeError, ValueError):
        number = 0
    for index, limit in enumerate(limits):
        if number <= limit:
            return index
    return len(limits)


def surface_shape_feature_vector(record_or_shape: dict[str, Any]) -> list[float]:
    """Project a bounded generic shape; role/oracle fields are not read."""

    shape = record_or_shape.get("surface_shape") if isinstance(record_or_shape, dict) else None
    if not isinstance(shape, dict):
        shape = record_or_shape if isinstance(record_or_shape, dict) else {}
    values = [0.0] * SURFACE_ROLE_FEATURE_DIM
    content_type = str(shape.get("content_type_class", "other")).casefold()
    values[{"html": 0, "json": 1, "other": 2}.get(content_type, 2)] = 1.0
    status_class = str(shape.get("status_class", "other")).casefold()
    values[3 + {"2xx": 0, "3xx": 1, "4xx": 2, "5xx": 3}.get(status_class, 4)] = 1.0
    values[8] = min(float(max(0, int(shape.get("html_tag_count", 0) or 0))) / 32.0, 1.0)
    values[9] = min(float(max(0, int(shape.get("html_attribute_count", 0) or 0))) / 16.0, 1.0)
    values[10] = min(float(max(0, int(shape.get("script_count", 0) or 0))) / 16.0, 1.0)
    values[11] = min(float(max(0, int(shape.get("json_field_count", 0) or 0))) / 16.0, 1.0)
    values[12] = min(float(max(0, int(shape.get("response_header_count", 0) or 0))) / 16.0, 1.0)
    body_bucket = _bucket(shape.get("body_length", 0), (64, 128, 256, 512, 1024))
    delta_bucket = _bucket(shape.get("body_length_delta_abs", 0), (0, 16, 64, 128, 256))
    values[13 + body_bucket] = 1.0
    values[19 + delta_bucket] = 1.0
    return values


class SurfaceRoleDiscriminator(nn.Module):
    """Small role classifier with an explicit abstain threshold at inference."""

    def __init__(self, feature_dim: int = SURFACE_ROLE_FEATURE_DIM, hidden_dim: int = 64, dropout: float = 0.08):
        super().__init__()
        if feature_dim != SURFACE_ROLE_FEATURE_DIM:
            raise ValueError(f"SurfaceRoleDiscriminator expects feature_dim={SURFACE_ROLE_FEATURE_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(SURFACE_ROLES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor, *, abstain_threshold: float = 0.80) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for value, index in zip(values.cpu(), indices.cpu()):
            confidence = float(value)
            outputs.append({
                "role": SURFACE_ROLES[int(index)] if confidence >= float(abstain_threshold) else None,
                "candidate_role": SURFACE_ROLES[int(index)],
                "confidence": round(confidence, 6),
                "abstained": confidence < float(abstain_threshold),
            })
        return outputs


__all__ = [
    "SURFACE_ROLE_FEATURE_DIM",
    "SURFACE_ROLES",
    "SurfaceRoleDiscriminator",
    "surface_shape_feature_vector",
]
