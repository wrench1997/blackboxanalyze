"""Small family-specific decoder for abstract SQL response channels."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


SQL_CHANNEL_CLASSES = ("control", "injection")
SQL_CHANNEL_FEATURE_DIM = 32


def sql_channel_feature_vector(record: dict[str, Any]) -> list[float]:
    """Build features only from the visible probe and bounded response shape."""

    values = [0.0] * SQL_CHANNEL_FEATURE_DIM
    payload = record.get("payload") or {}
    response = record.get("response_projection") or {}
    probe = str(payload.get("probe", "")).casefold()
    values[0] = int(str(payload.get("probe_kind", "")) == "sql_channel_class")
    values[1] = min(len(probe), 64) / 64.0
    values[2] = int("syntax" in probe)
    values[3] = int("blind" in probe)
    values[4] = int("row" in probe)
    values[5] = int("time" in probe)
    values[6] = int("local" in probe)
    values[7] = int("operator" in probe)
    values[8] = int("quoted" in probe or "plain" in probe)
    status = int(response.get("status_code", 0) or 0)
    values[9] = int(200 <= status < 300)
    values[10] = int(400 <= status < 500)
    values[11] = int(status >= 500)
    body_length = max(0, int(response.get("body_length", 0) or 0))
    values[12] = min(body_length, 512) / 512.0
    shape = response.get("json_shape") or {}
    values[13] = min(max(0, int(shape.get("key_count", 0) or 0)), 16) / 16.0
    values[14] = int(str(shape.get("type", "")) == "object")
    values[15] = int(bool(response.get("headers")))
    values[16] = int(bool(response.get("body_sha256")))
    # Keep a bounded structural checksum for unknown abstract probe names;
    # it is not derived from oracle fields or route labels.
    values[17] = min(sum(ord(char) for char in probe) % 97, 96) / 96.0 if probe else 0.0
    values[18] = int("%" in str(payload.get("probe", "")))
    values[19] = int("_" in probe)
    values[20] = int(len(probe.split("_")) >= 2)
    values[21] = int(response.get("status_code") == 422)
    values[22] = int(body_length <= 64)
    values[23] = int(body_length >= 96)
    values[24] = int("parameter" in probe)
    values[25] = int("error" in probe)
    values[26] = int("callback" in probe)
    values[27] = int("channel" in probe)
    values[28] = int("class" in probe)
    values[29] = int("url" in str((record.get("probe_artifact") or {}).get("encoding", "")))
    values[30] = int(response.get("status_code") != 200)
    values[31] = min(len(str(response.get("headers", {}))), 256) / 256.0
    return values


class SqlChannelDecoder(nn.Module):
    def __init__(self, feature_dim: int = SQL_CHANNEL_FEATURE_DIM, hidden_dim: int = 64, dropout: float = 0.08):
        super().__init__()
        if feature_dim != SQL_CHANNEL_FEATURE_DIM:
            raise ValueError(f"SqlChannelDecoder expects feature_dim={SQL_CHANNEL_FEATURE_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(SQL_CHANNEL_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor, *, abstain_threshold: float = 0.80) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index in zip(values.cpu(), indices.cpu()):
            value = float(confidence)
            candidate = SQL_CHANNEL_CLASSES[int(index)]
            outputs.append({
                "family": "injection" if candidate == "injection" and value >= float(abstain_threshold) else None,
                "candidate_class": candidate,
                "candidate_family": "injection" if candidate == "injection" else "control",
                "confidence": round(value, 6),
                "abstained": value < float(abstain_threshold),
            })
        return outputs


__all__ = ["SQL_CHANNEL_CLASSES", "SQL_CHANNEL_FEATURE_DIM", "SqlChannelDecoder", "sql_channel_feature_vector"]
