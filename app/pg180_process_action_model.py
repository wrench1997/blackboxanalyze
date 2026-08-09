"""PG-180 process-token action model utilities.

The model sees only bounded, abstract process tokens from a PG-179B trace.  It
does not see route paths, family labels, raw request values, response bodies,
oracle authority, or exploit syntax.  Its output vocabulary is restricted to
safe abstract next actions; a later replay controller still validates the
action manifest before any local request is sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .causal_trace_transformer import CausalTraceTransformer
from .moe_trace_transformer import MoETraceTransformer


SCHEMA_VERSION = "sift-pg180-process-action-model-v1"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
ACTION_TOKENS = (
    "action::repeat_matched_negative_pair",
    "action::abstain_unknown_oracle",
    "action::probe_candidate_other_method",
)
ALLOWED_ACTIONS = tuple(token.split("::", 1)[1] for token in ACTION_TOKENS)
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "path",
        "route",
        "family",
        "body",
        "raw",
        "payload",
        "oracle_projection",
        "target_instance_id",
        "marker",
        "probe_value",
    }
)


MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "small": {"kind": "causal", "d_model": 128, "nhead": 4, "layers": 2, "max_len": 128},
    "medium": {"kind": "causal", "d_model": 256, "nhead": 8, "layers": 3, "max_len": 128},
    "moe_large": {"kind": "moe", "d_model": 512, "nhead": 8, "layers": 3, "n_experts": 4, "expert_ff": 1024, "max_len": 128},
}


def _bucket(value: int, cuts: tuple[int, ...]) -> str:
    for cut in cuts:
        if value <= cut:
            return str(cut)
    return "plus"


def _safe_token(value: Any) -> str:
    token = str(value)
    if any(key in token.casefold() for key in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "onerror")):
        raise ValueError("unsafe token reached PG-180 model input")
    return token


def abstract_step_tokens(step: Mapping[str, Any], history: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    """Project a trace step into family-free process tokens.

    Only shape, method, failure, belief, and bounded budget information is
    retained.  Parameter *names* are intentionally reduced to a field count;
    the crawler remains the authority for names at replay time.
    """

    action = dict(step.get("action_manifest") or {})
    response = dict(step.get("response_projection") or {})
    failure = dict(step.get("failure_signature") or {})
    belief = dict(step.get("belief_after") or {})
    method = str(action.get("method", "GET")).upper()
    placement = str(action.get("placement", "none"))
    status = str(response.get("status_class", "other"))
    shape = dict(response.get("shape") or {})
    shape_kind = str(shape.get("kind", "unknown"))
    methods_seen = {str(item).upper() for item in failure.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}}
    top_belief = "none"
    if belief:
        top_belief = max(sorted(belief), key=lambda key: float(belief.get(key, 0.0)))
    field_count = len(action.get("form_field_names", [])) if method == "POST" else 0
    tokens = [
        f"method::{_safe_token(method)}",
        f"placement::{_safe_token(placement)}",
        f"encoding::{_safe_token((action.get('encoding_chain') or ['identity'])[0])}",
        f"failure::{_safe_token(failure.get('kind', 'unknown'))}",
        f"gate::{_safe_token(failure.get('failed_gate', 'unknown'))}",
        f"candidate::{int(bool(failure.get('candidate_signal')))}",
        f"typed_available::{int(bool(failure.get('typed_available')))}",
        f"authority::{int(bool(failure.get('positive_authority')))}",
        f"status::{_safe_token(status)}",
        f"shape::{_safe_token(shape_kind)}",
        f"status_chain_len::{_bucket(len(response.get('status_chain', [])), (0, 1, 2, 4))}",
        f"round::{_bucket(int(failure.get('probe_round', 1) or 1), (1, 2, 3, 4))}",
        f"budget::{_bucket(int(failure.get('remaining_probe_budget', 0) or 0), (0, 1, 2, 4))}",
        f"methods_seen::{_bucket(len(methods_seen), (0, 1, 2))}",
        f"field_count::{_bucket(field_count, (0, 1, 2, 4, 8))}",
        f"belief_top::{_safe_token(top_belief)}",
        f"history_len::{_bucket(len(history), (0, 1, 2, 4))}",
    ]
    if any(any(key in token.casefold() for key in FORBIDDEN_INPUT_KEYS) for token in tokens):
        raise AssertionError("PG-180 projection accidentally retained a forbidden input key")
    return tokens


def example_tokens(step: Mapping[str, Any], history: Sequence[Mapping[str, Any]] = ()) -> tuple[list[str], str]:
    """Build context tokens and the allow-listed next-action target."""

    target = str(step.get("next_action", ""))
    if target not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported PG-180 action target: {target}")
    context: list[str] = [BOS_TOKEN]
    for previous in history:
        context.extend(f"history::{token}" for token in abstract_step_tokens(previous, history=()))
    context.extend(abstract_step_tokens(step, history=history))
    return context, f"action::{target}"


def build_vocabulary(examples: Sequence[tuple[list[str], str]]) -> dict[str, int]:
    tokens = {PAD_TOKEN, BOS_TOKEN, *ACTION_TOKENS}
    for context, target in examples:
        tokens.update(context)
        tokens.add(target)
    ordered = [PAD_TOKEN, BOS_TOKEN] + sorted(tokens - {PAD_TOKEN, BOS_TOKEN})
    return {token: index for index, token in enumerate(ordered)}


def encode_examples(examples: Sequence[tuple[list[str], str]], vocabulary: Mapping[str, int]) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for context, target in examples:
        if any(token not in vocabulary for token in context) or target not in vocabulary:
            raise ValueError("PG-180 vocabulary is missing an abstract token")
        encoded.append({"ids": [int(vocabulary[token]) for token in context], "target": int(vocabulary[target]), "target_action": target.split("::", 1)[1]})
    return encoded


def build_model(vocabulary_size: int, variant: str) -> nn.Module:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown PG-180 model variant: {variant}")
    config = MODEL_VARIANTS[variant]
    if config["kind"] == "moe":
        return MoETraceTransformer(vocabulary_size, d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], n_experts=config["n_experts"], expert_ff=config["expert_ff"], max_len=config["max_len"])
    return CausalTraceTransformer(vocabulary_size, d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=config["max_len"])


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def collate(encoded: Sequence[Mapping[str, Any]], *, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not encoded:
        raise ValueError("cannot collate an empty PG-180 batch")
    max_len = max(len(row["ids"]) for row in encoded)
    ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
    targets: list[int] = []
    for index, row in enumerate(encoded):
        values = torch.tensor(list(row["ids"]), dtype=torch.long)
        ids[index, : len(values)] = values
        mask[index, : len(values)] = True
        targets.append(int(row["target"]))
    return ids, mask, torch.tensor(targets, dtype=torch.long)


def last_logits(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids, mask)
    lengths = mask.long().sum(dim=1).clamp_min(1) - 1
    return logits[torch.arange(ids.shape[0], device=ids.device), lengths]


def restrict_action(logits: torch.Tensor, vocabulary: Mapping[str, int], *, context: Mapping[str, Any] | None = None) -> tuple[str, float]:
    action_ids = [int(vocabulary[token]) for token in ACTION_TOKENS if token in vocabulary]
    values = logits[action_ids]
    probabilities = torch.softmax(values, dim=0)
    confidence, index = torch.max(probabilities, dim=0)
    action = ACTION_TOKENS[int(index)].split("::", 1)[1]
    # If the crawler did not observe an alternate parameterized channel, the
    # controller must not let the model invent one.
    if context and bool(context.get("single_channel")) and action == "probe_candidate_other_method":
        action = "abstain_unknown_oracle"
    return action, float(confidence.detach().cpu())


__all__ = [
    "ACTION_TOKENS",
    "ALLOWED_ACTIONS",
    "MODEL_VARIANTS",
    "SCHEMA_VERSION",
    "abstract_step_tokens",
    "build_model",
    "build_vocabulary",
    "collate",
    "encode_examples",
    "example_tokens",
    "last_logits",
    "parameter_count",
    "restrict_action",
]
