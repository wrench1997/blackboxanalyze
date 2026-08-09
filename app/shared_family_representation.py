"""Anonymous cross-family observation representation and routing head.

This module is intentionally narrower than the family-specific decoders.  It
removes oracle/semantic labels, raw response bytes, route tokens, and source
provenance before a shared encoder sees an observation.  The router may emit a
candidate family and a grammar-checked Rule IR template, but it never proves a
finding; family-specific typed oracles remain the only positive gate.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import torch
from torch import nn

from .rule_ir_decoder import validate_abstract_rule_ir


SHARED_FAMILY_CLASSES = ("control", "xss", "injection", "access_control", "logic")
SHARED_FEATURE_DIM = 128
SHARED_EMBEDDING_DIM = 96
# OOD for the shared *family route* should be invariant to the response
# surface itself.  Surface-specific geometry is handled by the typed sink
# discriminator; including it here would turn every legitimate content-type
# variant into a hard OOD abstain.
OOD_INVARIANT_FEATURE_INDICES = tuple(
    # Transport method/status and attested local-run metadata are shared
    # across content surfaces.  Path/query geometry, body shape, content
    # type, parser counts and probe syntax are intentionally absent: a new
    # endpoint or representation must be handled by the typed sink oracle,
    # not rejected as an OOD family route.
    [0, 7, 16, 17, 18, 19, 20, 21, 33, 34, 89, 90, 91, 92, 93, 94, 95, 105, 109, 112, 127]
)
SHARED_RULE_IR = {
    "xss": {"op": "not", "arg": {"op": "html_creates_nodes", "arg": {"op": "policy_slot", "name": "untrusted_text"}}},
    "injection": {"op": "policy_slot", "name": "untrusted_data_cannot_change_interpreter_structure"},
    "access_control": {"op": "and", "args": [{"op": "policy_slot", "name": "subject_authenticated"}, {"op": "policy_slot", "name": "subject_authorized_for_resource"}]},
    "logic": {"op": "and", "args": [{"op": "policy_slot", "name": "invariant_holds"}, {"op": "policy_slot", "name": "state_replay_is_valid"}]},
}
for _rule in SHARED_RULE_IR.values():
    validate_abstract_rule_ir(_rule)


def _bounded_hash(value: str, buckets: int = 16) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def _path_and_query(record: dict[str, Any]) -> tuple[str, dict[str, str], str]:
    payload = record.get("payload") or {}
    raw_path = str(payload.get("path", ""))
    parsed = urlsplit(raw_path)
    query = {
        str(key): unquote(str(values[0]))
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
        if values
    }
    return parsed.path, query, raw_path


def shared_anonymous_feature_vector(record: dict[str, Any]) -> list[float]:
    """Project one visible observation without family/oracle shortcuts."""

    values = [0.0] * SHARED_FEATURE_DIM
    payload = record.get("payload") or {}
    response = record.get("response_projection") or {}
    shape = response.get("json_shape") or {}
    surface_shape = record.get("surface_shape") or {}
    path, query, raw_path = _path_and_query(record)
    # Canonicalize one transport decode before measuring probe geometry.  The
    # representation must not learn that ``%6D`` is a new family or surface.
    raw_probe = str(payload.get("probe", ""))
    decoded_probe = unquote(raw_probe)
    status = int(response.get("status_code", 0) or 0)
    body_length = max(0, int(response.get("body_length", 0) or 0))
    headers = response.get("headers") or {}
    content_type = str(headers.get("content-type", "")).casefold()
    canonical_query = "&".join(f"{key}={query[key]}" for key in sorted(query))

    # Request geometry, not route tokens or probe words.
    values[0] = float(str(payload.get("method", "GET")).upper() == "GET")
    values[1] = min(len(path.split("/")) - 1, 8) / 8.0
    values[2] = float(bool(query))
    values[3] = min(len(query), 8) / 8.0
    values[4] = 0.0
    values[5] = min(len(canonical_query), 256) / 256.0
    values[6] = min(len(decoded_probe), 256) / 256.0
    values[7] = float(str(payload.get("probe_kind", "")) != "")
    values[8] = 0.0
    values[9] = float("&" in raw_path)
    values[10] = float("=" in raw_path)
    values[11] = float(path.endswith("/"))
    values[12] = min(sum(len(key) for key in query), 96) / 96.0
    values[13] = min(sum(len(value) for value in query.values()), 256) / 256.0
    values[14] = float(any(value.lstrip("+-").isdigit() for value in query.values()))
    values[15] = float(any(value in {"0", "1", "true", "false"} for value in query.values()))

    # Bounded response class/shape.  No body text, semantic JSON keys, or
    # oracle projections enter this vector.
    values[16] = float(status // 100 == 2)
    values[17] = float(status // 100 == 3)
    values[18] = float(status // 100 == 4)
    values[19] = float(status // 100 == 5)
    values[20] = float(status == 200)
    values[21] = float(status == 403)
    values[22] = float("json" in content_type)
    values[23] = float("html" in content_type)
    values[24] = float("text" in content_type)
    values[25] = min(body_length, 4096) / 4096.0
    values[26] = float(body_length <= 64)
    values[27] = float(64 < body_length <= 128)
    values[28] = float(128 < body_length <= 512)
    values[29] = float(body_length > 512)
    values[30] = min(max(0, int(shape.get("key_count", 0) or 0)), 32) / 32.0
    values[31] = float(str(shape.get("type", "")) == "object")
    values[32] = min(max(0, int(shape.get("scalar_count", 0) or 0)), 32) / 32.0
    values[33] = float(bool(response.get("body_sha256")))
    values[34] = float(bool(headers))
    values[35] = min(len(headers), 32) / 32.0

    # Surface geometry is generic parser output, not sink/oracle flags.
    values[36] = min(max(0, int(surface_shape.get("html_tag_count", 0) or 0)), 64) / 64.0
    values[37] = min(max(0, int(surface_shape.get("html_attribute_count", 0) or 0)), 64) / 64.0
    values[38] = min(max(0, int(surface_shape.get("script_count", 0) or 0)), 32) / 32.0
    values[39] = min(max(0, int(surface_shape.get("json_field_count", 0) or 0)), 32) / 32.0
    values[40] = min(max(0, int(surface_shape.get("response_header_count", 0) or 0)), 32) / 32.0
    values[41] = min(max(0, int(surface_shape.get("body_length", body_length) or 0)), 4096) / 4096.0
    values[42] = min(max(0, int(surface_shape.get("body_length_delta_abs", 0) or 0)), 4096) / 4096.0
    values[43] = float(str(surface_shape.get("content_type_class", "")) == "html")
    values[44] = float(str(surface_shape.get("content_type_class", "")) == "json")
    values[45] = float(str(surface_shape.get("content_type_class", "")) == "other")

    # Query value geometry is decoded once, so a transport encoding cannot
    # create a new family.  Names contribute only anonymous length/hash shape.
    values[46] = min(sum(len(value) for value in query.values()), 256) / 256.0
    values[47] = float(any("-" in value for value in query.values()))
    values[48] = 0.0
    values[49] = float(any(value.lstrip("+-").isdigit() for value in query.values()))
    values[50] = min(sum(ord(char) for char in canonical_query) % 97, 96) / 96.0 if canonical_query else 0.0
    for index, key in enumerate(sorted(query)[:8]):
        values[51 + index] = min(len(key), 24) / 24.0
        values[59 + index] = _bounded_hash(key, 32) / 31.0
    values[67] = float(len(query) == 1)
    values[68] = float(len(query) == 2)
    values[69] = float(len(query) >= 3)
    values[70] = float(any(len(value) <= 2 for value in query.values()))
    values[71] = float(any(len(value) >= 8 for value in query.values()))

    # Hash only broad structural descriptors; this is not a payload/body hash
    # and cannot reproduce a marker or source label.
    structure = f"{len(path.split('/'))}:{len(query)}:{status // 100}:{len(headers)}:{int('json' in content_type)}:{int('html' in content_type)}"
    digest = hashlib.sha256(structure.encode()).digest()
    for index in range(16):
        values[72 + index] = digest[index] / 255.0
    values[88] = min(len(str(record.get("schema_version", ""))), 64) / 64.0
    values[89] = float(bool(record.get("replay")))
    values[90] = float(bool(record.get("safety")))
    values[91] = float(bool(record.get("evidence")))
    values[92] = float(record.get("evaluator_state_visible") is False)
    values[93] = float(str((record.get("safety") or {}).get("local_only", "")) == "True")
    values[94] = float(str((record.get("safety") or {}).get("read_only", "")) == "True")
    values[95] = float("/" in path)
    values[96] = min(len(path), 128) / 128.0
    values[97] = float("." in path.rsplit("/", 1)[-1])
    values[98] = min(len(str(payload.get("marker", ""))), 64) / 64.0
    values[99] = float(bool(payload.get("headers")))
    values[100] = min(len(str(payload.get("headers", {}))), 256) / 256.0
    values[101] = float("accept" in str(payload.get("headers", {})).casefold())
    values[102] = float("application/json" in str(payload.get("headers", {})).casefold())
    values[103] = min(len(str(response.get("headers", {}))), 256) / 256.0
    values[104] = float("content-type" in str(response.get("headers", {})).casefold())
    values[105] = min(int(response.get("status_code", 0) or 0), 599) / 599.0
    values[106] = float(bool(record.get("pair")))
    values[107] = 0.0
    values[108] = 0.0
    values[109] = float(bool(record.get("surface_shape")))
    values[110] = float(bool(record.get("oracle_projection")))  # presence only; contents are not read
    values[111] = float(bool(record.get("semantic")))  # presence only; contents are not read
    values[112] = float("GET" == str(payload.get("method", "GET")).upper())
    values[113] = min(len(path + canonical_query), 256) / 256.0
    values[114] = float("?" in raw_path)
    values[115] = float("%" in decoded_probe)
    values[116] = float("&" in decoded_probe)
    values[117] = float("<" in decoded_probe)
    values[118] = float("=" in decoded_probe)
    values[119] = min(len(decoded_probe), 2048) / 2048.0
    values[120] = float(bool(record.get("candidate_status")))
    values[121] = float(bool(record.get("rule_ir_result")))  # overwritten to 0 by callers for model input
    values[122] = float(bool(record.get("oracle_revalidated")))  # presence only; overwritten by callers
    values[123] = float(bool(record.get("prediction")))
    values[124] = float(bool(record.get("safety", {}).get("external_network", False)))
    values[125] = float(bool(record.get("safety", {}).get("state_mutated", False)))
    values[126] = min(len(str(record.get("sample_id", ""))), 128) / 128.0
    values[127] = float(bool(record.get("payload")))
    # These last fields must never become label channels.  Any caller can
    # defensively zero them before training/evaluation.
    values[110:112] = [0.0, 0.0]
    values[120:127] = [0.0] * 7
    return values


class SharedFamilyRouter(nn.Module):
    """Shared encoder with a small family routing head."""

    def __init__(self, feature_dim: int = SHARED_FEATURE_DIM, embedding_dim: int = SHARED_EMBEDDING_DIM, hidden_dim: int = 128, dropout: float = 0.08):
        super().__init__()
        if feature_dim != SHARED_FEATURE_DIM:
            raise ValueError(f"SharedFamilyRouter expects feature_dim={SHARED_FEATURE_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, len(SHARED_FAMILY_CLASSES))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(features))

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        *,
        abstain_threshold: float = 0.75,
        margin_threshold: float = 0.10,
        temperature: float = 1.0,
    ) -> list[dict[str, Any]]:
        if float(temperature) <= 0.0:
            raise ValueError("shared-router temperature must be positive")
        # Apply scalar temperature to logits, not to the final top-1 score;
        # this preserves class ordering while making confidence thresholds
        # portable across fixtures.
        logits = self(features) / float(temperature)
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = probabilities.max(dim=-1)
        outputs: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.cpu(), indices.cpu(), probabilities.cpu()):
            candidate = SHARED_FAMILY_CLASSES[int(index)]
            sorted_values = torch.sort(row, descending=True).values
            margin = float(sorted_values[0] - sorted_values[1])
            confidence_value = float(confidence)
            accepted = confidence_value >= float(abstain_threshold) and margin >= float(margin_threshold)
            outputs.append({
                "family": candidate if accepted and candidate != "control" else None,
                "candidate_family": candidate,
                "confidence": round(confidence_value, 6),
                "margin": round(margin, 6),
                "abstained": not accepted,
                "rule_ir": SHARED_RULE_IR.get(candidate) if accepted and candidate != "control" else None,
                "probabilities": {name: round(float(probability), 6) for name, probability in zip(SHARED_FAMILY_CLASSES, row)},
            })
        return outputs


def shared_model_input(record: dict[str, Any]) -> list[float]:
    """Defensive wrapper that strips any post-collection evaluator additions."""

    sanitized = dict(record)
    for key in ("family", "semantic", "oracle_projection", "rule_ir_result", "oracle_revalidated", "prediction", "candidate_family", "candidate_status"):
        sanitized.pop(key, None)
    return shared_anonymous_feature_vector(sanitized)


__all__ = [
    "SHARED_EMBEDDING_DIM",
    "SHARED_FAMILY_CLASSES",
    "SHARED_FEATURE_DIM",
    "SHARED_RULE_IR",
    "OOD_INVARIANT_FEATURE_INDICES",
    "SharedFamilyRouter",
    "shared_anonymous_feature_vector",
    "shared_model_input",
]
