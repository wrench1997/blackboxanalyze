"""PG-23 multi-task Rule IR decoder.

This module is deliberately small and catalog-grounded.  It predicts a
family, a surface role, and whether the current visible trace is safe to
promote to an abstract Rule IR.  It never generates executable code or raw
attack strings; accepted predictions are mapped to the existing grammar-
checked family templates.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .catalog_rule_decoder import CATALOG_DECODER_FAMILIES, abstract_catalog_rule_ir
from .rule_ir_decoder import FEATURE_DIM, trace_feature_vector, validate_abstract_rule_ir


PG23_SURFACE_ROLES = (
    "xss_reflected_get",
    "xss_dom_value_source",
    "sqli_str",
    "sqli_search",
    "sqli_blind_boolean",
    "sqli_blind_time",
    "url_redirect_response",
    "unknown",
)
PG23_SURFACE_INDEX = {name: index for index, name in enumerate(PG23_SURFACE_ROLES)}
PG23_SCHEMA = "sift-pikachu-pg23-multitask-decoder-v2"
PG23_EVIDENCE_DIM = 32


def _coarse_encoding(value: Any) -> str:
    """Map provenance strings to a bounded, label-free encoding class."""

    text = str(value or "").casefold()
    if "double" in text and "html" in text:
        return "double_html_entity"
    if "html" in text or "entity" in text:
        return "html_entity"
    if "percent" in text or "url" in text:
        return "url_percent"
    return "plain"


def _coarse_probe_kind(value: Any) -> str:
    text = str(value or "").casefold()
    if "dom" in text and "markup" in text:
        return "inert_dom_markup"
    if "canary" in text:
        return "http_canary"
    if "sql" in text:
        return "sql_channel_class"
    return "generic_canary"


def _path_shape(path: Any) -> dict[str, Any]:
    value = str(path or "").split("?", 1)[0]
    parts = [part for part in value.split("/") if part]
    basename = parts[-1] if parts else ""
    extension = basename.rsplit(".", 1)[-1].casefold() if "." in basename else ""
    return {
        "segment_count": min(len(parts), 16),
        "has_extension": bool(extension),
        "extension": extension,
        "has_query": "?" in str(path or ""),
    }


def _content_type_class(value: Any) -> str:
    text = str(value or "").casefold()
    if "html" in text:
        return "html"
    if "json" in text:
        return "json"
    if "text" in text:
        return "text"
    return "other"


def pg23_visible_trace(row: dict[str, Any]) -> dict[str, Any]:
    """Project one catalog row without family, surface, oracle, or provenance.

    Counterfactual names and raw marker strings are intentionally collapsed.
    This prevents the negative-control construction from becoming a label
    side-channel while retaining observable modality and response shape.
    """

    payload = dict(row.get("payload") or {})
    response = dict(row.get("response_projection") or {})
    headers = dict(response.get("headers") or {})
    oracle = dict(row.get("oracle_projection") or {})
    probe_kind = _coarse_probe_kind(payload.get("probe_kind"))
    encoding = _coarse_encoding((row.get("probe_artifact") or {}).get("encoding"))
    # Keep only a bounded modality token.  In particular, do not pass the
    # marker text or an intervention name such as "counterfactual".
    probe_token = {
        "http_canary": "marker",
        "inert_dom_markup": "inert_markup",
        "sql_channel_class": "channel_class",
        "generic_canary": "marker",
    }.get(probe_kind, "marker")
    shape = {
        "content_type_class": _content_type_class(headers.get("content-type")),
        "status_class": (
            "success" if 200 <= int(response.get("status_code", 0)) < 300
            else "redirect" if 300 <= int(response.get("status_code", 0)) < 400
            else "client_error" if 400 <= int(response.get("status_code", 0)) < 500
            else "server_error" if int(response.get("status_code", 0)) >= 500
            else "transport_error"
        ),
        "body_bucket": min(max(int(response.get("body_length", 0)) // 256, 0), 32),
    }
    return {
        "input": {
            "action": {
                "method": str(payload.get("method", "GET")).upper(),
                "path": "relative_path",
                "path_shape": _path_shape(payload.get("path", "")),
            },
            "probe_kind": probe_kind,
            "probe": probe_token,
            "encoding": encoding,
        },
        "context": {
            "response": {
                "status_code": int(response.get("status_code", 0)),
                "content_type": _content_type_class(headers.get("content-type")),
                "body_shape": json.dumps(shape, sort_keys=True, separators=(",", ":")),
            },
            # Only a count is observable here.  Individual oracle keys and
            # their values stay in the post-decode evidence gate.
            "oracle_shape": {
                "field_count": len(oracle),
            },
        },
        "state": {
            "body_length": int(response.get("body_length", 0)),
        },
        "history": [],
        "output": False,
    }


def pg23_feature_vector(row: dict[str, Any]) -> list[float]:
    return trace_feature_vector([pg23_visible_trace(row)])


def pg23_evidence_vector(row: dict[str, Any]) -> list[float]:
    """Encode bounded post-replay oracle evidence, never the raw response."""

    response = dict(row.get("response_projection") or {})
    oracle = dict(row.get("oracle_projection") or {})
    status = int(response.get("status_code", 0))
    delta = abs(int(oracle.get("body_length_delta_abs", 0) or 0))
    marker_count = int(oracle.get("marker_count", 0) or 0)
    values = [0.0] * PG23_EVIDENCE_DIM
    values[0] = float(bool(oracle.get("marker_reflected")))
    values[1] = float(bool(oracle.get("marker_in_html_text")))
    values[2] = float(bool(oracle.get("marker_in_attribute")))
    values[3] = float(bool(oracle.get("marker_in_script_source")))
    values[4] = float(marker_count > 0)
    values[5] = float(bool(oracle.get("sql_error_shape")))
    values[6] = float(delta >= 32)
    values[7] = float(delta >= 128)
    values[8] = float(delta >= 256)
    values[9] = float(delta >= 1024)
    values[10] = float(bool(oracle.get("external_redirect")))
    values[11] = float(bool(oracle.get("redirect_present")))
    values[12] = float(bool(oracle.get("status_changed")))
    values[13] = float(status <= 0)
    values[14] = float(200 <= status < 300)
    values[15] = float(300 <= status < 400)
    values[16] = float(400 <= status < 500)
    values[17] = float(status >= 500)
    values[18] = min(delta / 4096.0, 1.0)
    values[19] = min(marker_count / 8.0, 1.0)
    values[20] = float(bool(oracle.get("title_present")))
    values[21] = float(bool(oracle.get("content_type_class") == "html"))
    values[22] = float(bool(oracle.get("form_count", 0)))
    values[23] = float(bool(oracle.get("input_count", 0)))
    values[24] = float(bool(oracle.get("script_count", 0)))
    values[25] = min(float(len(oracle)) / 32.0, 1.0)
    return values


def pg23_labels(row: dict[str, Any]) -> tuple[int, int, float]:
    """Return family index, surface index, and safe Rule IR emission label."""

    family = str((row.get("semantic") or {}).get("family", ""))
    surface = str((row.get("semantic") or {}).get("surface", "unknown"))
    if family not in CATALOG_DECODER_FAMILIES:
        raise ValueError(f"unsupported PG-23 family: {family}")
    surface_index = PG23_SURFACE_INDEX.get(surface, PG23_SURFACE_INDEX["unknown"])
    # A positive label means the catalog's bounded oracle accepted the
    # abstract Rule IR.  It is not a claim that an exploit executed.
    emit = float(bool(row.get("rule_ir_result")) and not bool(row.get("counterfactual")))
    return CATALOG_DECODER_FAMILIES.index(family), surface_index, emit


class PG23MultiTaskDecoder(nn.Module):
    """Compact shared encoder with family/surface/abstain heads."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        evidence_dim: int = PG23_EVIDENCE_DIM,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"PG-23 expects feature_dim={FEATURE_DIM}")
        if evidence_dim != PG23_EVIDENCE_DIM:
            raise ValueError(f"PG-23 expects evidence_dim={PG23_EVIDENCE_DIM}")
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evidence_tower = nn.Sequential(
            nn.Linear(evidence_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.family_head = nn.Linear(embedding_dim, len(CATALOG_DECODER_FAMILIES))
        self.surface_head = nn.Linear(embedding_dim, len(PG23_SURFACE_ROLES))
        self.emit_head = nn.Linear(embedding_dim + 32, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(features))

    def forward(
        self,
        features: torch.Tensor,
        evidence_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding = self.encode(features)
        if evidence_features is None:
            evidence_features = torch.zeros(
                (features.shape[0], PG23_EVIDENCE_DIM),
                dtype=features.dtype,
                device=features.device,
            )
        evidence_embedding = self.evidence_tower(evidence_features)
        return {
            "embedding": embedding,
            "evidence_embedding": evidence_embedding,
            "family_logits": self.family_head(embedding),
            "surface_logits": self.surface_head(embedding),
            "emit_logits": self.emit_head(torch.cat((embedding, evidence_embedding), dim=-1)).squeeze(-1),
        }

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        evidence_features: torch.Tensor | None = None,
        *,
        family_threshold: float = 0.50,
        emit_threshold: float = 0.50,
        margin_threshold: float = 0.05,
    ) -> list[dict[str, Any]]:
        outputs = self(features, evidence_features)
        family_probabilities = torch.softmax(outputs["family_logits"], dim=-1)
        surface_probabilities = torch.softmax(outputs["surface_logits"], dim=-1)
        emit_probabilities = torch.sigmoid(outputs["emit_logits"])
        family_values, family_indices = family_probabilities.max(dim=-1)
        surface_values, surface_indices = surface_probabilities.max(dim=-1)
        decoded: list[dict[str, Any]] = []
        for family_conf, family_index, surface_conf, surface_index, emit_conf, family_row in zip(
            family_values.cpu(),
            family_indices.cpu(),
            surface_values.cpu(),
            surface_indices.cpu(),
            emit_probabilities.cpu(),
            family_probabilities.cpu(),
        ):
            sorted_family = torch.sort(family_row, descending=True).values
            margin = float(sorted_family[0] - sorted_family[1])
            family_name = CATALOG_DECODER_FAMILIES[int(family_index)]
            surface_name = PG23_SURFACE_ROLES[int(surface_index)]
            accepted = (
                float(family_conf) >= float(family_threshold)
                and float(emit_conf) >= float(emit_threshold)
                and margin >= float(margin_threshold)
            )
            rule_ir = abstract_catalog_rule_ir(family_name) if accepted else None
            if rule_ir is not None:
                validate_abstract_rule_ir(rule_ir)
            decoded.append({
                "family": family_name if accepted else None,
                "candidate_family": family_name,
                "surface": surface_name,
                "family_confidence": round(float(family_conf), 6),
                "surface_confidence": round(float(surface_conf), 6),
                "emit_confidence": round(float(emit_conf), 6),
                "family_margin": round(margin, 6),
                "accepted": bool(accepted),
                "abstained": not bool(accepted),
                "rule_ir": copy.deepcopy(rule_ir),
            })
        return decoded


def pair_consistency_loss(
    embeddings: torch.Tensor,
    rows: Iterable[dict[str, Any]],
) -> torch.Tensor:
    """Pull only same-pair encoding views together."""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair = row.get("pair") or {}
        pair_id = str(pair.get("pair_id", ""))
        if pair_id:
            groups.setdefault(pair_id, []).append(index)
    normalized = F.normalize(embeddings, dim=-1)
    losses: list[torch.Tensor] = []
    for indices in groups.values():
        for left_index in range(len(indices)):
            for right_index in range(left_index + 1, len(indices)):
                left = indices[left_index]
                right = indices[right_index]
                losses.append(1.0 - (normalized[left] * normalized[right]).sum())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def assert_visible_trace_redacted(row: dict[str, Any]) -> None:
    """Fail fast if evaluator labels leak into the learner projection."""

    text = json.dumps(pg23_visible_trace(row), ensure_ascii=False).casefold()
    forbidden = ("counterfactual", "rule_ir_result", "surface_role", "oracle_projection")
    if any(token in text for token in forbidden):
        raise AssertionError("PG-23 visible trace contains an evaluator-side token")
    # Family names are not passed as raw route or probe tokens.
    if any(token in text for token in ("xss_reflected_get", "xss_dom_value_source", "sqli_blind")):
        raise AssertionError("PG-23 visible trace contains a family/surface shortcut")
