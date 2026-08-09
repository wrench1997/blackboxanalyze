"""PG-230 quality funnel for causal next-token process data.

Next-token supervision is useful as a representation objective, but loss alone
cannot tell a reusable trace from a memorized template.  This module assigns a
bounded trace to a retention lane and emits abstract event tokens without raw
payloads, response bodies, route identities or evaluator keys.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

import torch
from torch import nn


PG230_SCHEMA = "pg230-next-token-quality-funnel-v1"
LANES = ("gold", "hard_negative", "silver", "quarantine", "reject")
LANE_INDEX = {name: index for index, name in enumerate(LANES)}
REPAIR_ACTIONS = ("abstain", "recheck_oracle", "compare_reference", "inspect_binding", "retry_candidate", "inspect_environment", "gate_correction")
REPAIR_INDEX = {name: index for index, name in enumerate(REPAIR_ACTIONS)}
SPECIAL_TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[UNK]")
FORBIDDEN_KEYS = frozenset({"raw_payload", "payload", "raw_response", "response_body", "body_preview", "evaluator_key", "challenge_key", "route_identity"})
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.=:+/-]{1,64}$")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _flag(value: Any) -> str:
    return "1" if bool(value) else "0"


def _surface_class(row: Mapping[str, Any]) -> str:
    value = str(row.get("surface_role") or row.get("surface") or row.get("route_class") or "unknown").casefold()
    for needle, category in (
        ("metric", "observability"),
        ("observ", "observability"),
        ("swagger", "documentation"),
        ("api-doc", "documentation"),
        ("graphql", "documentation"),
        ("robot", "discovery"),
        ("sitemap", "discovery"),
        ("ftp", "information_exposure"),
        ("backup", "information_exposure"),
        ("xss", "dom_surface"),
        ("sqli", "sql_surface"),
        ("redirect", "redirect_surface"),
        ("search", "query_surface"),
        ("missing", "missing_control"),
    ):
        if needle in value:
            return category
    return "generic_surface"


def _failure_kind(row: Mapping[str, Any]) -> str:
    if row.get("model_self_error_detected") or row.get("model_self_error_kind"):
        return "model_self_error"
    # A deliberately matched counterfactual is a controlled no-effect probe,
    # not a reference disagreement.  Preserve that attribution so the repair
    # head learns the actual failure stage while the lane can still retain it
    # as a hard negative.
    if str(row.get("failure_signature", "")) == "counterfactual_candidate_no_effect":
        return "candidate_no_effect"
    if row.get("transport_error"):
        return "environment_failure"
    if row.get("binding_valid") is False:
        return "binding_failure"
    if row.get("candidate_reference_agreement") is False:
        return "reference_disagreement"
    if row.get("result_mismatch_observed"):
        return "result_mismatch"
    if row.get("oracle_available") is False:
        return "oracle_unavailable"
    if row.get("typed_effect_observed") or row.get("typed_effect_confirmed") or row.get("result_fixture_verified"):
        return "typed_effect"
    return "candidate_no_effect"


def _repair_action(row: Mapping[str, Any], lane: str) -> str:
    if row.get("model_gate_corrected_diagnosis") == "confirmed_local_effect":
        return "gate_correction"
    if row.get("next_step") in REPAIR_INDEX:
        return str(row["next_step"])
    if lane == "gold":
        return "abstain"
    if lane == "hard_negative":
        return "gate_correction" if _failure_kind(row) == "model_self_error" else "recheck_oracle"
    if lane == "silver":
        return "recheck_oracle"
    return "abstain"


def _bucket(value: Any) -> str:
    try:
        number = max(int(value or 0), 0)
    except (TypeError, ValueError):
        number = 0
    if number == 0:
        return "0"
    if number == 1:
        return "1"
    if number == 2:
        return "2"
    return "3+"


def quality_lane(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return a retention lane and explicit reasons; never infer quality from loss."""

    reasons: list[str] = []
    if FORBIDDEN_KEYS.intersection(row.keys()):
        return "reject", [f"forbidden_key:{key}" for key in sorted(FORBIDDEN_KEYS.intersection(row.keys()))]
    for key in ("source", "method", "seed", "evidence_hash"):
        if key not in row or row.get(key) in (None, ""):
            reasons.append(f"missing:{key}")
    if len(str(row.get("evidence_hash", ""))) != 64:
        reasons.append("invalid:evidence_hash")
    if not bool(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False):
        reasons.append("raw_retention_not_false")
    candidate_sent = bool(row.get("candidate_sent", row.get("ai_sent", True)))
    typed = bool(row.get("typed_effect_confirmed") or row.get("typed_effect_observed") or row.get("result_fixture_verified"))
    reference = bool(row.get("candidate_reference_agreement", False))
    negative = bool(row.get("negative_clean", False))
    self_error = bool(row.get("model_self_error_detected") or row.get("model_self_error_kind"))
    if reasons:
        return "quarantine", reasons
    if self_error:
        return "hard_negative", ["model_self_error_with_repair_target"]
    if (
        bool(row.get("abstention_required"))
        and not bool(row.get("oracle_available"))
        and not candidate_sent
        and bool(row.get("fresh_reset_ok", False))
        and bool(row.get("reset_completed", False))
        and negative
        and row.get("failure_signature")
    ):
        return "hard_negative", ["explicit_oracle_gap_abstain"]
    if (
        bool(row.get("negative_control_confirmed"))
        and candidate_sent
        and bool(row.get("oracle_available"))
        and negative
        and row.get("failure_signature")
    ):
        return "hard_negative", ["matched_negative_oracle_clean"]
    if typed and reference and negative and bool(row.get("fresh_reset_ok", True)) and bool(row.get("reset_completed", True)):
        return "gold", ["typed_oracle_reference_negative_complete"]
    if candidate_sent and not bool(row.get("oracle_available", False)):
        return "silver", ["bounded_projection_without_family_specific_oracle"]
    if not candidate_sent or row.get("reset_not_attempted"):
        return "quarantine", ["preflight_or_incomplete_transition"]
    return "quarantine", ["incomplete_typed_evidence"]


def event_tokens(row: Mapping[str, Any], lane: str | None = None) -> list[str]:
    """Encode an abstract bounded event; route/payload/raw values are excluded."""

    lane = lane or quality_lane(row)[0]
    method = str(row.get("method", "GET")).upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "HEAD", "OPTIONS"}:
        method = "OTHER"
    status = str(row.get("status_class", "unknown"))
    if status not in {"1xx", "2xx", "3xx", "4xx", "5xx", "unknown"}:
        status = "unknown"
    tokens = [
        "[BOS]",
        f"surface={_surface_class(row)}",
        f"method={method}",
        f"status={status}",
        f"candidate_sent={_flag(row.get('candidate_sent', row.get('ai_sent', True)))}",
        f"oracle_available={_flag(row.get('oracle_available'))}",
        f"typed_effect={_flag(row.get('typed_effect_confirmed') or row.get('typed_effect_observed'))}",
        f"result_verified={_flag(row.get('result_fixture_verified'))}",
        f"reference_agreement={_flag(row.get('candidate_reference_agreement'))}",
        f"negative_clean={_flag(row.get('negative_clean'))}",
        f"field_bucket={_bucket(row.get('field_count'))}",
        f"history_bucket={_bucket(row.get('history_len'))}",
        f"feedback={str(row.get('previous_feedback', 'none'))[:24]}",
        f"candidate_present={_flag(row.get('candidate_result_present'))}",
        f"model_claimed_positive={_flag(row.get('model_claimed_positive'))}",
        f"failure={_failure_kind(row)}",
        f"lane={lane}",
        f"repair={_repair_action(row, lane)}",
        f"self_error={_flag(row.get('model_self_error_detected') or row.get('model_self_error_kind'))}",
        "[EOS]",
    ]
    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        if not TOKEN_RE.fullmatch(token):
            raise ValueError(f"PG-230 generated token is outside the bounded vocabulary: {token!r}")
    return tokens


def prepare_record(row: Mapping[str, Any]) -> dict[str, Any]:
    lane, reasons = quality_lane(row)
    tokens = event_tokens(row, lane)
    return {
        "source": str(row.get("source", "unknown")),
        "seed": int(row.get("seed", 0) or 0),
        "surface_class": _surface_class(row),
        "method": str(row.get("method", "GET")).upper(),
        "lane": lane,
        "lane_index": LANE_INDEX[lane],
        "repair_action": _repair_action(row, lane),
        "repair_index": REPAIR_INDEX[_repair_action(row, lane)],
        "failure_kind": _failure_kind(row),
        "failure_signature": str(row.get("failure_signature", "")),
        "abstention_required": bool(row.get("abstention_required", False)),
        "tokens": tokens,
        "token_hash": digest(tokens),
        "quality_reasons": reasons,
        "source_evidence_hash": str(row.get("evidence_hash", "")),
        "model_self_error_detected": bool(row.get("model_self_error_detected") or row.get("model_self_error_kind")),
        "payload_grounded_eligible": bool(lane == "gold" and row.get("payload_grounded_eligible", False)),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


class FrozenXXLNextTokenAdapter(nn.Module):
    """Trainable next-token/lane/repair heads over a frozen XXL context."""

    def __init__(self, *, d_model: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.context_projection = nn.Sequential(nn.LayerNorm(int(d_model)), nn.Linear(int(d_model), int(hidden_dim)), nn.GELU())
        self.token_head = nn.Linear(int(hidden_dim), int(vocab_size))
        self.lane_head = nn.Linear(int(hidden_dim), len(LANES))
        self.repair_head = nn.Linear(int(hidden_dim), len(REPAIR_ACTIONS))

    def forward(self, context: torch.Tensor, *, pooled: torch.Tensor | None = None, classification_positions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hidden = self.context_projection(context)
        token_logits = self.token_head(hidden)
        if classification_positions is not None:
            if hidden.ndim != 3:
                raise ValueError("PG-230 classification positions require a sequence context")
            positions = classification_positions.to(device=hidden.device, dtype=torch.long).clamp(min=0, max=hidden.shape[1] - 1)
            pooled_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        else:
            pooled_hidden = hidden[:, -1] if pooled is None else hidden if hidden.ndim == 2 else hidden[:, -1]
        return {"token": token_logits, "lane": self.lane_head(pooled_hidden), "repair": self.repair_head(pooled_hidden)}


def split_quality_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by source/seed/surface class and keep quarantine out of training."""

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for record in records:
        if record["lane"] in {"quarantine", "reject"}:
            quarantine.append(dict(record))
            continue
        key = (str(record.get("source", "")), int(record.get("seed", 0) or 0), str(record.get("surface_class", "")))
        # Even seeds and one held-out surface class form a true OOD partition.
        is_holdout = key[1] % 2 == 0 or key[2] in {"observability", "dom_surface", "sql_surface"}
        (holdout if is_holdout else train).append(dict(record))
    if not train or not holdout:
        raise ValueError("PG-230 requires non-empty train and holdout quality records")
    return train, holdout, quarantine


def build_vocabulary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tokens = sorted({str(token) for record in records for token in record["tokens"] if str(token) not in SPECIAL_TOKENS})
    itos = list(SPECIAL_TOKENS) + tokens
    return {token: index for index, token in enumerate(itos)}


def encode_sequences(records: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise ValueError("PG-230 cannot encode an empty record set")
    encoded = [[int(vocabulary.get(token, vocabulary["[UNK]"])) for token in record["tokens"]] for record in records]
    width = max(len(seq) for seq in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, seq in enumerate(encoded):
        ids[index, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return ids[:, :-1], ids[:, 1:], torch.tensor([LANE_INDEX[str(row["lane"])] for row in records], dtype=torch.long, device=device), torch.tensor([REPAIR_INDEX[str(row["repair_action"])] for row in records], dtype=torch.long, device=device)


def quality_metrics(outputs: Mapping[str, torch.Tensor], targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor], *, pad_index: int = 0) -> dict[str, Any]:
    token_logits, lane_logits, repair_logits = outputs["token"], outputs["lane"], outputs["repair"]
    token_targets, lane_targets, repair_targets = targets
    token_loss = nn.functional.cross_entropy(token_logits.reshape(-1, token_logits.shape[-1]), token_targets.reshape(-1), ignore_index=pad_index)
    valid = token_targets.ne(pad_index)
    token_pred = token_logits.argmax(-1)
    token_count = int(valid.sum().item())
    return {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(token_count, 1)), 8),
        "token_count": token_count,
        "lane_accuracy": round(float((lane_logits.argmax(-1) == lane_targets).float().mean().item()), 8),
        "repair_accuracy": round(float((repair_logits.argmax(-1) == repair_targets).float().mean().item()), 8),
        "self_error_recall": None,
    }


__all__ = [
    "FrozenXXLNextTokenAdapter",
    "LANES",
    "LANE_INDEX",
    "PG230_SCHEMA",
    "REPAIR_ACTIONS",
    "REPAIR_INDEX",
    "build_vocabulary",
    "digest",
    "encode_sequences",
    "event_tokens",
    "prepare_record",
    "quality_lane",
    "quality_metrics",
    "split_quality_records",
]
