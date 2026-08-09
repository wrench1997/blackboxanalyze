"""Small Rule IR decoder trained from the PG-03 replay catalog."""

from __future__ import annotations

import copy
import json
from typing import Any

import torch
from torch import nn

from .rule_ir import canonical
from .rule_ir_decoder import FEATURE_DIM, abstract_rule_ir, trace_feature_vector, validate_abstract_rule_ir


CATALOG_DECODER_FAMILIES = (
    "access_control",
    "injection",
    "logic",
    "url_redirect",
    "xss",
)
CATALOG_DECODER_SCHEMA = "sift-catalog-rule-ir-decoder-v1"
_PROJECTION_LABEL_KEYS = {
    "family",
    "source_id",
    "semantic",
    "rule_ir",
    "evaluator",
    "intended_output",
    "is_counterexample",
}


def _path_shape(path: Any) -> dict[str, Any]:
    """Keep route structure while dropping language/family token names."""

    value = str(path).split("?", 1)[0]
    parts = [part for part in value.split("/") if part]
    basename = parts[-1] if parts else ""
    extension = basename.rsplit(".", 1)[-1].casefold() if "." in basename else ""
    return {
        "segment_count": min(len(parts), 16),
        "has_extension": bool(extension),
        "extension": extension,
        "has_query": "?" in str(path),
    }


def abstract_catalog_rule_ir(family: str) -> dict[str, Any]:
    if family == "logic":
        return {
            "op": "and",
            "args": [
                {"op": "policy_slot", "name": "invariant_holds"},
                {"op": "policy_slot", "name": "state_replay_is_valid"},
            ],
        }
    if family not in CATALOG_DECODER_FAMILIES:
        raise KeyError(f"unsupported catalog decoder family: {family}")
    return copy.deepcopy(abstract_rule_ir(family))


for _family in CATALOG_DECODER_FAMILIES:
    validate_abstract_rule_ir(abstract_catalog_rule_ir(_family))


def _shape_text(value: Any) -> str:
    """Keep only non-semantic response-shape information.

    The maze registry intentionally contains fields such as ``family`` and
    ``exit_oracle``.  Passing object key names through to the learner would
    turn the response-shape feature into an evaluator/label side channel.
    Replacing every ``keys`` list with its cardinality preserves useful
    structural information while making that leakage impossible.
    """

    def redact_shape(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                ("key_count" if key == "keys" else key):
                (len(value) if key == "keys" and isinstance(value, list) else redact_shape(value))
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [redact_shape(item) for item in node]
        return node

    return json.dumps(redact_shape(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)[:2048]


def catalog_visible_trace(record: dict[str, Any]) -> dict[str, Any]:
    """Project a catalog row without labels, provenance, oracle values or IR."""

    payload = record["payload"]
    response = record.get("response_projection") or {}
    headers = response.get("headers") or {}
    oracle_projection = record.get("oracle_projection") or {}
    probe_artifact = record.get("probe_artifact") or {}
    # Encoding is observable input structure, not a family/evaluator label.
    # Keeping the bounded encoding descriptor visible lets the pair-invariance
    # trainer learn that plain, percent, and entity forms are related views.
    encoding = str(probe_artifact.get("encoding", ""))[:80]
    return {
        "input": {
            "action": {
                "method": payload.get("method", "GET"),
                # Raw route tokens (for example ``xss`` or ``sqli``) are a
                # shortcut that does not transfer across languages.  Keep a
                # neutral path placeholder plus bounded structural shape.
                "path": "relative_path",
                "path_shape": _path_shape(payload.get("path", "")),
            },
            "probe_kind": payload.get("probe_kind", "http_canary"),
            "probe": payload.get("probe", ""),
            "encoding": encoding,
        },
        "context": {
            "response": {
                "status_code": int(response.get("status_code", 0)),
                "content_type": headers.get("content-type", ""),
                "body_shape": _shape_text(response.get("json_shape")),
            },
            # Only aggregate shape counts are visible.  Oracle key names such
            # as ``sql_error_shape`` would be a family shortcut; their values
            # remain in the evaluator layer.
            "oracle_shape": {
                "field_count": len([
                    key for key in oracle_projection.keys()
                    if str(key).casefold() not in _PROJECTION_LABEL_KEYS
                ]),
            },
        },
        "state": {
            "body_length": int(response.get("body_length", 0)),
        },
        "history": [],
        # Never pass the replay/evaluator result to the learner.  The decoder
        # must infer a candidate from the visible probe/response projection;
        # the oracle is used only after emission to score the experiment.
        "output": False,
    }


def catalog_feature_vector(record: dict[str, Any]) -> list[float]:
    return trace_feature_vector([catalog_visible_trace(record)])


class CatalogRuleIRDecoder(nn.Module):
    """Compact supervised family decoder with grammar-checked Rule IR output."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(CATALOG_DECODER_FAMILIES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(features))

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        *,
        abstain_threshold: float = 0.60,
        margin_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        logits = self(features)
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = probabilities.max(dim=-1)
        decoded: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.cpu(), indices.cpu(), probabilities.cpu()):
            candidate_family = CATALOG_DECODER_FAMILIES[int(index)]
            confidence_value = float(confidence)
            sorted_probabilities = torch.sort(row, descending=True).values
            margin = float(sorted_probabilities[0] - sorted_probabilities[1])
            accepted = confidence_value >= float(abstain_threshold) and margin >= float(margin_threshold)
            decoded.append({
                "family": candidate_family if accepted else None,
                "candidate_family": candidate_family,
                "confidence": round(confidence_value, 6),
                "margin": round(margin, 6),
                "abstained": not accepted,
                "rule_ir": abstract_catalog_rule_ir(candidate_family) if accepted else None,
                "probabilities": {
                    name: round(float(probability), 6)
                    for name, probability in zip(CATALOG_DECODER_FAMILIES, row)
                },
            })
        return decoded


def canonical_template(family: str) -> str:
    return canonical(abstract_catalog_rule_ir(family))


CATALOG_SURFACE_START = 145
CATALOG_SURFACE_END = 189
CATALOG_SURFACE_DIM = CATALOG_SURFACE_END - CATALOG_SURFACE_START
CATALOG_CONTEXT_DIM = FEATURE_DIM - CATALOG_SURFACE_DIM
CATALOG_DECODER_V2_SCHEMA = "sift-catalog-rule-ir-decoder-v2"


class CatalogRuleIRDecoderV2(nn.Module):
    """Two-view Rule IR decoder with a contrastive-friendly embedding.

    The v1 MLP treats the whole trace vector as one undifferentiated block.
    This version separates surface cues (encoding/sink/URL shape) from the
    remaining context, fuses them with a learned gate, and exposes the fused
    embedding for supervised contrastive training.  It still emits only the
    small grammar-checked family templates above; it is not a free-form code
    generator.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        branch_dim: int = 128,
        embedding_dim: int = 96,
        dropout: float = 0.08,
    ):
        super().__init__()
        if feature_dim != FEATURE_DIM:
            raise ValueError(f"CatalogRuleIRDecoderV2 expects feature_dim={FEATURE_DIM}")
        self.surface_tower = nn.Sequential(
            nn.Linear(CATALOG_SURFACE_DIM, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )
        self.context_tower = nn.Sequential(
            nn.Linear(CATALOG_CONTEXT_DIM, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2, branch_dim),
            nn.Sigmoid(),
        )
        self.projector = nn.Sequential(
            nn.Linear(branch_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, len(CATALOG_DECODER_FAMILIES))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        surface = features[:, CATALOG_SURFACE_START:CATALOG_SURFACE_END]
        context = torch.cat(
            (features[:, :CATALOG_SURFACE_START], features[:, CATALOG_SURFACE_END:]),
            dim=-1,
        )
        surface_hidden = self.surface_tower(surface)
        context_hidden = self.context_tower(context)
        joined = torch.cat((surface_hidden, context_hidden), dim=-1)
        gate = self.gate(joined)
        fused = torch.cat((surface_hidden * gate, context_hidden * (1.0 - gate)), dim=-1)
        return self.projector(fused)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(features))

    @torch.inference_mode()
    def decode(
        self,
        features: torch.Tensor,
        *,
        abstain_threshold: float = 0.25,
        margin_threshold: float = 0.08,
    ) -> list[dict[str, Any]]:
        probabilities = torch.softmax(self(features), dim=-1)
        values, indices = probabilities.max(dim=-1)
        decoded: list[dict[str, Any]] = []
        for confidence, index, row in zip(values.cpu(), indices.cpu(), probabilities.cpu()):
            candidate_family = CATALOG_DECODER_FAMILIES[int(index)]
            confidence_value = float(confidence)
            sorted_probabilities = torch.sort(row, descending=True).values
            margin = float(sorted_probabilities[0] - sorted_probabilities[1])
            accepted = confidence_value >= float(abstain_threshold) and margin >= float(margin_threshold)
            decoded.append({
                "family": candidate_family if accepted else None,
                "candidate_family": candidate_family,
                "confidence": round(confidence_value, 6),
                "margin": round(margin, 6),
                "abstained": not accepted,
                "rule_ir": abstract_catalog_rule_ir(candidate_family) if accepted else None,
                "probabilities": {
                    name: round(float(probability), 6)
                    for name, probability in zip(CATALOG_DECODER_FAMILIES, row)
                },
            })
        return decoded
