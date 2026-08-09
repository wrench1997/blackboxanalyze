"""Feature-space novelty gate for the frozen Rule IR decoder."""

from __future__ import annotations

from typing import Any

import torch


OOD_GATE_SCHEMA = "sift-feature-ood-gate-v1"


def fit_ood_reference(reference_features: torch.Tensor, *, quantile: float = 0.95, slack: float = 1.25) -> dict[str, Any]:
    """Fit a leave-one-out nearest-neighbor threshold on training surfaces."""

    if reference_features.ndim != 2 or reference_features.shape[0] < 2:
        raise ValueError("OOD reference requires at least two 2D feature rows")
    quantile = float(quantile)
    slack = float(slack)
    if not 0.0 < quantile <= 1.0 or slack <= 0:
        raise ValueError("OOD quantile must be in (0,1] and slack must be positive")
    distances = torch.cdist(reference_features.float(), reference_features.float())
    distances.fill_diagonal_(float("inf"))
    leave_one_out = distances.min(dim=1).values
    base = float(torch.quantile(leave_one_out, quantile).item())
    threshold = max(1e-6, base * slack)
    return {
        "schema_version": OOD_GATE_SCHEMA,
        "reference_count": int(reference_features.shape[0]),
        "feature_dim": int(reference_features.shape[1]),
        "quantile": quantile,
        "slack": slack,
        "leave_one_out_max": float(leave_one_out.max().item()),
        "threshold": threshold,
    }


def nearest_reference_distances(features: torch.Tensor, reference_features: torch.Tensor) -> list[float]:
    if features.ndim != 2 or reference_features.ndim != 2 or features.shape[1] != reference_features.shape[1]:
        raise ValueError("OOD query/reference feature dimensions must match")
    return [float(value) for value in torch.cdist(features.float(), reference_features.float()).min(dim=1).values]


def ood_flags(distances: list[float], fit: dict[str, Any]) -> list[bool]:
    threshold = float(fit["threshold"])
    return [float(distance) > threshold for distance in distances]


__all__ = ["OOD_GATE_SCHEMA", "fit_ood_reference", "nearest_reference_distances", "ood_flags"]
