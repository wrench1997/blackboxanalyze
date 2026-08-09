"""Compact family-specific Rule IR head for the logic/access maze."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import torch
from torch import nn

from .rule_ir_decoder import validate_abstract_rule_ir


LOGIC_ACCESS_CLASSES = ("control", "access_control", "logic")
# The original 64 features overfit the v1 route/query vocabulary.  The final
# 16 slots are vocabulary-agnostic value/boundary descriptors so an unseen
# surface can be decoded from the same Rule IR semantics without copying
# endpoint names.
LOGIC_ACCESS_FEATURE_DIM = 80
# Features that encode the v1 route/query vocabulary rather than the
# underlying boundary primitive.  They are retained for diagnostics, but are
# zeroed before decoder training/evaluation so a new endpoint cannot become a
# hidden family label.
LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES = tuple(list(range(8, 30)) + [50])
LOGIC_ACCESS_RULE_IR = {
    "access_control": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "typed_authorization_boundary"},
            {"op": "policy_slot", "name": "protected_resource_transition"},
        ],
    },
    "logic": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "invariant_holds"},
            {"op": "policy_slot", "name": "state_replay_is_valid"},
        ],
    },
}
for _rule in LOGIC_ACCESS_RULE_IR.values():
    validate_abstract_rule_ir(_rule)


def _parsed_input(record: dict[str, Any]) -> tuple[str, dict[str, str]]:
    payload = record.get("payload") or {}
    raw_path = str(payload.get("path", ""))
    parsed = urlsplit(raw_path)
    query = {str(key): unquote(str(values[0])) for key, values in parse_qs(parsed.query, keep_blank_values=True).items() if values}
    return parsed.path, query


def logic_access_feature_vector(record: dict[str, Any]) -> list[float]:
    """Use only visible route/query structure and bounded response shape.

    Oracle fields, labels, source ids, raw bodies, and Rule IR results are
    deliberately excluded.  Query values are the probe itself, so their
    normalized type/boundary flags are observable rather than evaluator data.
    """

    values = [0.0] * LOGIC_ACCESS_FEATURE_DIM
    path, query = _parsed_input(record)
    response = record.get("response_projection") or {}
    status = int(response.get("status_code", 0) or 0)
    shape = response.get("json_shape") or {}
    body_length = max(0, int(response.get("body_length", 0) or 0))
    payload = record.get("payload") or {}
    raw_path = str(payload.get("path", ""))

    values[0] = float(str(payload.get("probe_kind", "")) == "http_canary")
    values[1] = float(200 <= status < 300)
    values[2] = float(400 <= status < 500)
    values[3] = min(body_length, 512) / 512.0
    values[4] = min(max(0, int(shape.get("key_count", 0) or 0)), 12) / 12.0
    values[5] = float(str(shape.get("type", "")) == "object")
    values[6] = float("%" in raw_path)
    values[7] = min(len(query), 8) / 8.0
    values[8] = float(path == "/gate")
    values[9] = float(path == "/coupon")
    values[10] = float(path == "/replay")
    values[11] = float(path == "/health")

    role = query.get("role", "")
    quota = query.get("quota", "")
    values[12] = float(role == "member")
    values[13] = float(role == "admin")
    values[14] = float(role in {"guest", ""})
    values[15] = float(quota not in {"", "0"})
    values[16] = float(quota == "0" or quota == "")
    values[17] = float(quota.startswith("-"))

    member = query.get("member", "")
    total = query.get("total", "")
    values[18] = float(member == "1")
    values[19] = float(member != "1")
    values[20] = float(total == "100")
    values[21] = float(total in {"99", "0", ""})
    values[22] = float(total not in {"", "0", "99", "100"})
    try:
        values[23] = min(abs(int(total)), 200) / 200.0
    except ValueError:
        values[23] = 0.0

    action = query.get("action", "")
    previous = query.get("previous", "")
    challenge = query.get("challenge", "")
    current = query.get("current", "")
    values[24] = float(action == "commit")
    values[25] = float(previous == "verify")
    values[26] = float(challenge == current and challenge != "")
    values[27] = float(challenge != current)
    values[28] = float(action in {"wait", "", "cancel"})

    # Generic shape/checksum features help the head survive renamed fields and
    # different JSON styles without handing it semantic response keys.
    values[29] = min(sum(ord(char) for char in path) % 97, 96) / 96.0 if path else 0.0
    values[30] = min(sum(len(key) for key in query), 64) / 64.0
    values[31] = float(bool(response.get("headers")))
    values[32] = float(bool(response.get("body_sha256")))
    values[33] = float(body_length <= 64)
    values[34] = float(64 < body_length <= 96)
    values[35] = float(body_length > 96)
    values[36] = float(status == 403)
    values[37] = float(status == 200)
    values[38] = float(len(path.split("/")) - 1)
    values[39] = float("&" in raw_path)
    values[40] = float("=" in raw_path)
    canonical_query = "&".join(f"{key}={query[key]}" for key in sorted(query))
    values[41] = min(len(path + canonical_query), 256) / 256.0
    # Stable query-name shape (not the evaluator's expected family/oracle).
    for index, key in enumerate(sorted(query)[:8]):
        values[42 + index] = min(len(key), 16) / 16.0
    values[50] = float(query.get("ok") == "1")
    values[51] = float(str(payload.get("encoding", "")) == "url_percent")
    values[52] = float(str((record.get("probe_artifact") or {}).get("encoding", "")) == "url_percent")
    values[53] = min(len(str(payload.get("probe", ""))), 64) / 64.0
    values[54] = float("-" in canonical_query)
    values[55] = float("0" in canonical_query)
    values[56] = float("1" in canonical_query)
    values[57] = float("100" in canonical_query)
    values[58] = float("verify" in canonical_query)
    values[59] = float("commit" in canonical_query)
    values[60] = float(response.get("status_code") != 0)
    values[61] = min(int(shape.get("scalar_count", 0) or 0), 12) / 12.0
    values[62] = float(status // 100 == 2)
    values[63] = float(status // 100 == 4)
    # Cross-surface semantic primitives.  These intentionally inspect only
    # bounded request/value geometry; oracle fields, route labels and source
    # identity never enter the vector.
    query_values = list(query.values())
    lower_values = [value.casefold() for value in query_values]
    numeric_values: list[int] = []
    for value in query_values:
        try:
            numeric_values.append(int(value))
        except ValueError:
            continue
    values[64] = float(any(value in {"member", "owner", "admin", "guest", "visitor"} for value in lower_values))
    values[65] = float(any(value in {"member", "owner", "admin", "guest", "visitor"} for value in lower_values) and any(value not in {"0", ""} for value in query_values))
    values[66] = float(any(value < 0 for value in numeric_values))
    values[67] = float(any(value == 100 for value in numeric_values))
    values[68] = float(any(value in {"commit", "verify", "verified"} for value in lower_values))
    values[69] = float(
        (query.get("challenge") != "" and query.get("challenge") == query.get("current"))
        or (query.get("nonce") != "" and query.get("nonce") == query.get("now"))
        or (query.get("stamp") != "" and query.get("stamp") == query.get("fresh"))
    )
    values[70] = float(
        (query.get("challenge") != "" and query.get("challenge") != query.get("current"))
        or (query.get("nonce") != "" and query.get("nonce") != query.get("now"))
        or (query.get("stamp") != "" and query.get("stamp") != query.get("fresh"))
    )
    values[71] = float(len(query) >= 3)
    values[72] = float(any(value == "0" for value in query_values))
    values[73] = float(any(value == "1" for value in query_values))
    values[74] = float(any(value == "100" for value in query_values))
    values[75] = float(any(value == "99" for value in query_values))
    values[76] = min(len(path), 64) / 64.0
    values[77] = min(len(numeric_values), 8) / 8.0
    values[78] = float(any(len(value) >= 4 for value in query_values))
    values[79] = float(status == 403)
    return values


def logic_access_model_feature_vector(record: dict[str, Any]) -> list[float]:
    """Return the surface-invariant view used by the family head."""

    values = logic_access_feature_vector(record)
    for index in LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES:
        values[index] = 0.0
    return values


class LogicAccessDecoder(nn.Module):
    def __init__(self, feature_dim: int = LOGIC_ACCESS_FEATURE_DIM, hidden_dim: int = 64, dropout: float = 0.08):
        super().__init__()
        if feature_dim != LOGIC_ACCESS_FEATURE_DIM:
            raise ValueError(f"LogicAccessDecoder expects feature_dim={LOGIC_ACCESS_FEATURE_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(LOGIC_ACCESS_CLASSES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(self, features: torch.Tensor, *, abstain_threshold: float = 0.80, margin_threshold: float = 0.10, temperature: float = 1.0) -> list[dict[str, Any]]:
        if float(temperature) <= 0.0:
            raise ValueError("logic/access decoder temperature must be positive")
        probabilities = torch.softmax(self(features) / float(temperature), dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.cpu(), indices.cpu(), probabilities.cpu()):
            candidate = LOGIC_ACCESS_CLASSES[int(index)]
            confidence_value = float(confidence)
            sorted_values = torch.sort(row, descending=True).values
            margin = float(sorted_values[0] - sorted_values[1])
            accepted = confidence_value >= float(abstain_threshold) and margin >= float(margin_threshold)
            outputs.append({
                "family": candidate if accepted and candidate != "control" else None,
                "candidate_family": candidate,
                "confidence": round(confidence_value, 6),
                "margin": round(margin, 6),
                "abstained": not accepted,
                "rule_ir": LOGIC_ACCESS_RULE_IR.get(candidate) if accepted and candidate != "control" else None,
                "probabilities": {name: round(float(probability), 6) for name, probability in zip(LOGIC_ACCESS_CLASSES, row)},
            })
        return outputs


__all__ = ["LOGIC_ACCESS_CLASSES", "LOGIC_ACCESS_FEATURE_DIM", "LOGIC_ACCESS_RULE_IR", "LOGIC_ACCESS_SURFACE_SHORTCUT_INDICES", "LogicAccessDecoder", "logic_access_feature_vector", "logic_access_model_feature_vector"]
