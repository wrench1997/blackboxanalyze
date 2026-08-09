"""PG-288 structural verifier for abstract Rule-IR wire plans.

PG-285 deliberately reports action and safety metrics, but those two heads can
look perfect while a decoder emits an incomplete or internally inconsistent
sequence.  This module adds a separate, deterministic verifier for the
*abstract* plan.  It never accepts a literal payload and it never authorizes a
request by itself; a typed evaluator and fresh replay remain external gates.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pg288-rule-ir-verifier-v1"
TARGET_BOS = "[TARGET_BOS]"
TARGET_EOS = "[TARGET_EOS]"

REQUIRED_FIELDS = frozenset(
    {
        "plan",
        "method",
        "probe_class",
        "channel",
        "encoding",
        "wire",
        "field_slot",
        "repair_delta",
        "family_agnostic",
        "final_action",
        "safe_to_send",
    }
)
ALLOWED_ACTIONS = frozenset(
    {
        "abstain",
        "negative_control",
        "reference_probe",
        "candidate_probe",
        "repair_alternate",
        "replay_confirmed",
    }
)
ALLOWED_METHODS = frozenset({"GET", "POST"})
ALLOWED_CHANNELS = frozenset({"query", "form", "path", "header", "unknown"})
ALLOWED_ENCODINGS = frozenset({"plain", "url_percent", "base64", "json", "unknown"})
WIRE_FOR_CHANNEL = {
    "query": "query_param",
    "form": "form_field",
    "path": "path_segment",
    "header": "header_value",
    "unknown": "none",
}
ALLOWED_WIRES = frozenset(WIRE_FOR_CHANNEL.values())
ALLOWED_FIELD_SLOTS = frozenset({"observed_or_runtime_canary", "none"})

# These are only a last-resort guard against accidentally putting a literal
# probe in a structured target.  Abstract values such as ``probe_class=sql``
# and ``encoding=url_percent`` are intentionally not matched.
_LITERAL_PROBE_RE = re.compile(
    r"(?:<\s*script\b|javascript\s*:|onerror\s*=|union\s+select|sleep\s*\(|benchmark\s*\(|\b(?:or|and)\s+\d+\s*=\s*\d+|file\s*:\s*/)",
    re.IGNORECASE,
)
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/<>-]{1,96}$")


def _result(
    *,
    valid_structure: bool,
    safe_consistent: bool,
    renderable: bool,
    eligible_for_send: bool,
    errors: Sequence[str],
    fields: Mapping[str, str],
    literal_probe: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "valid_structure": bool(valid_structure),
        "safe_consistent": bool(safe_consistent),
        "renderable": bool(renderable),
        # This is intentionally false unless the caller supplies an external
        # typed evaluator verdict.  A plan verifier cannot grant permission.
        "eligible_for_send": bool(eligible_for_send),
        "errors": list(dict.fromkeys(str(item) for item in errors)),
        "fields": dict(fields),
        "literal_probe_token": bool(literal_probe),
    }


def parse_plan_tokens(tokens: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Parse a bounded target sequence without interpreting raw request data."""

    values = [str(token) for token in tokens]
    errors: list[str] = []
    if len(values) < 3 or values[0] != TARGET_BOS or values[-1] != TARGET_EOS:
        errors.append("missing_target_boundaries")
        core = values[1:-1] if values else []
    else:
        core = values[1:-1]
    if len(values) > 64:
        errors.append("sequence_too_long")
    fields: dict[str, str] = {}
    for token in core:
        if "=" not in token:
            errors.append("non_assignment_token")
            continue
        key, value = token.split("=", 1)
        if not _KEY_RE.fullmatch(key) or not _VALUE_RE.fullmatch(value):
            errors.append("malformed_assignment")
            continue
        if key in fields:
            errors.append(f"duplicate_field:{key}")
            continue
        fields[key] = value
        if _LITERAL_PROBE_RE.search(token):
            errors.append("literal_probe_token")
    return fields, errors


def verify_plan_tokens(
    tokens: Sequence[str],
    *,
    typed_oracle_confirmed: bool = False,
) -> dict[str, Any]:
    """Verify an abstract plan; do not authorize network activity."""

    fields, errors = parse_plan_tokens(tokens)
    missing = sorted(REQUIRED_FIELDS - set(fields))
    errors.extend(f"missing_field:{key}" for key in missing)

    plan = fields.get("plan", "")
    action = fields.get("final_action", "")
    method = fields.get("method", "")
    channel = fields.get("channel", "")
    encoding = fields.get("encoding", "")
    wire = fields.get("wire", "")
    field_slot = fields.get("field_slot", "")
    repair_delta = fields.get("repair_delta", "")
    family_agnostic = fields.get("family_agnostic", "")
    safe_value = fields.get("safe_to_send", "")
    literal_probe = "literal_probe_token" in errors

    if method and method not in ALLOWED_METHODS:
        errors.append("unsupported_method")
    if channel and channel not in ALLOWED_CHANNELS:
        errors.append("unsupported_channel")
    if encoding and encoding not in ALLOWED_ENCODINGS:
        errors.append("unsupported_encoding")
    if wire and wire not in ALLOWED_WIRES:
        errors.append("unsupported_wire")
    if field_slot and field_slot not in ALLOWED_FIELD_SLOTS:
        errors.append("unsupported_field_slot")
    if action and action not in ALLOWED_ACTIONS:
        errors.append("unsupported_action")
    if plan and action and plan != action:
        errors.append("plan_action_mismatch")
    if family_agnostic and family_agnostic != "1":
        errors.append("family_context_not_agnostic")
    if safe_value not in {"0", "1"}:
        errors.append("invalid_safe_bit")

    safe = safe_value == "1"
    expected_wire = WIRE_FOR_CHANNEL.get(channel)
    # An abstain plan intentionally carries no wire even if the observed
    # surface was a query/form surface; it is a decision not to send.
    if expected_wire and wire and wire != expected_wire and action != "abstain":
        errors.append("channel_wire_mismatch")
    if channel == "unknown" and encoding != "unknown":
        errors.append("unknown_channel_requires_unknown_encoding")
    if action == "abstain":
        if safe:
            errors.append("abstain_must_not_be_safe")
        if wire != "none" or field_slot != "none":
            errors.append("abstain_must_have_no_wire")
    elif action:
        if wire == "none" or field_slot != "observed_or_runtime_canary":
            errors.append("action_requires_runtime_slot")
        if encoding == "unknown":
            errors.append("action_requires_known_encoding")
    if action in {"negative_control", "reference_probe", "repair_alternate"} and safe:
        errors.append("non_candidate_action_must_not_be_safe")
    if action in {"candidate_probe", "replay_confirmed"} and not safe:
        errors.append("candidate_action_requires_safe_bit")
    if action == "repair_alternate" and repair_delta == "none":
        errors.append("repair_action_requires_delta")
    if action != "repair_alternate" and repair_delta not in {"", "none"}:
        errors.append("unexpected_repair_delta")
    if action == "replay_confirmed" and not typed_oracle_confirmed:
        errors.append("replay_confirmation_requires_typed_oracle")

    structural_errors = [error for error in errors if error != "replay_confirmation_requires_typed_oracle"]
    valid_structure = not structural_errors
    safe_consistent = valid_structure and not any(
        error
        in {
            "abstain_must_not_be_safe",
            "non_candidate_action_must_not_be_safe",
            "candidate_action_requires_safe_bit",
            "invalid_safe_bit",
        }
        for error in errors
    )
    renderable = valid_structure and action != "abstain" and wire != "none" and not literal_probe
    eligible_for_send = bool(renderable and safe and typed_oracle_confirmed and not errors)
    return _result(
        valid_structure=valid_structure,
        safe_consistent=safe_consistent,
        renderable=renderable,
        eligible_for_send=eligible_for_send,
        errors=errors,
        fields=fields,
        literal_probe=literal_probe,
    )


def evaluate_decoded_plans(
    rows: Sequence[Mapping[str, Any]],
    predicted_tokens: Sequence[Sequence[str]],
    *,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Score structural quality separately from action/safety labels."""

    if len(rows) != len(predicted_tokens):
        raise ValueError("rows and predicted_tokens must have equal length")
    structural = 0
    safe_consistent = 0
    renderable = 0
    action_correct = 0
    exact = 0
    false_allow = 0
    confirmation_claims = 0
    for row, predicted in zip(rows, predicted_tokens):
        verification = verify_plan_tokens(predicted)
        structural += int(verification["valid_structure"])
        safe_consistent += int(verification["safe_consistent"])
        renderable += int(verification["renderable"])
        fields = verification["fields"]
        expected = row.get("target") or {}
        expected_action = str(expected.get("next_action", ""))
        expected_safe = bool(expected.get("safe_to_send", False))
        predicted_action = str(fields.get("final_action", ""))
        predicted_safe = fields.get("safe_to_send") == "1"
        action_correct += int(predicted_action == expected_action)
        exact += int([str(item) for item in predicted] == [str(item) for item in row.get("target_tokens", [])])
        false_allow += int(predicted_safe and (hard_negative or not expected_safe))
        confirmation_claims += int(predicted_action == "replay_confirmed")
    count = len(rows)
    return {
        "count": count,
        "structural_valid_rate": round(structural / max(count, 1), 6),
        "safe_consistent_rate": round(safe_consistent / max(count, 1), 6),
        "renderable_rate": round(renderable / max(count, 1), 6),
        "action_accuracy": round(action_correct / max(count, 1), 6),
        "sequence_exact_accuracy": round(exact / max(count, 1), 6),
        "false_allow_count": int(false_allow),
        "confirmation_claim_count": int(confirmation_claims),
        "hard_negative_false_allow": int(false_allow) if hard_negative else None,
    }


def constrained_greedy_decode(
    model: Any,
    context_values: Any,
    context_lengths: Any,
    target_vocab: Mapping[str, int],
    *,
    max_tokens: int = 24,
) -> list[list[str]]:
    """Decode a valid abstract slot skeleton without seeing target labels.

    The decoder still chooses plan/action values from its logits.  The mask
    only prevents impossible syntax (for example ``wire=none`` for a
    candidate) and literal/free-form values; evaluator confirmation remains
    outside this function.
    """

    reverse = {int(index): str(token) for token, index in target_vocab.items()}
    device = context_values.device
    state = model.encode_context(context_values, context_lengths)
    bos_id = int(target_vocab.get("[TARGET_BOS]"))
    eos_id = int(target_vocab.get("[TARGET_EOS]"))
    current = __import__("torch").full((context_values.shape[0], 1), bos_id, dtype=__import__("torch").long, device=device)
    outputs: list[list[str]] = [[TARGET_BOS] for _ in range(context_values.shape[0])]
    actions = ["abstain"] * context_values.shape[0]
    finished = [False] * context_values.shape[0]

    def ids(prefix: str, *, allow_unknown: bool = True, none_only: bool = False) -> list[int]:
        result: list[int] = []
        for token, index in target_vocab.items():
            if not str(token).startswith(prefix):
                continue
            value = str(token).split("=", 1)[1] if "=" in str(token) else ""
            if none_only and value != "none":
                continue
            if not allow_unknown and value == "unknown":
                continue
            result.append(int(index))
        return result

    torch = __import__("torch")
    for position in range(1, int(max_tokens)):
        decoded, state = model.decoder(model.target_embedding(current[:, -1:]), state)
        logits = model.output(model.norm(decoded[:, -1]))
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for row_index, output in enumerate(outputs):
            action = actions[row_index]
            if position == 1:
                allowed = ids("plan=")
            elif position == 2:
                allowed = ids("method=", allow_unknown=False)
            elif position == 3:
                allowed = ids("probe_class=")
            elif position == 4:
                allowed = ids("channel=")
            elif position == 5:
                allowed = ids("encoding=", allow_unknown=action == "abstain")
            elif position == 6:
                allowed = ids("wire=", none_only=action == "abstain")
            elif position == 7:
                allowed = ids("field_slot=", none_only=action == "abstain")
            elif position == 8:
                allowed = [int(index) for token, index in target_vocab.items() if str(token).startswith("repair_delta=") and (action == "repair_alternate" or str(token) == "repair_delta=none")]
            elif position == 9:
                allowed = ids("family_agnostic=")
                allowed = [index for index in allowed if reverse.get(index) == "family_agnostic=1"] or allowed
            elif position == 10:
                allowed = [int(target_vocab[token]) for token in (f"final_action={action}",) if token in target_vocab]
            elif position == 11:
                safe = "1" if action in {"candidate_probe", "replay_confirmed"} else "0"
                allowed = [int(target_vocab[token]) for token in (f"safe_to_send={safe}",) if token in target_vocab]
            else:
                allowed = [eos_id]
            if not allowed:
                allowed = list(range(logits.shape[-1]))
            mask[row_index, allowed] = True
        next_ids = logits.masked_fill(~mask, float("-inf")).argmax(-1)
        current = torch.cat([current, next_ids.unsqueeze(1)], dim=1)
        for row_index, token_id in enumerate(next_ids.detach().cpu().tolist()):
            token = reverse.get(int(token_id), "[UNK]")
            if finished[row_index]:
                continue
            outputs[row_index].append(token)
            if position == 1 and token.startswith("plan="):
                candidate = token.split("=", 1)[1]
                actions[row_index] = candidate if candidate in ALLOWED_ACTIONS else "abstain"
            if token_id == eos_id:
                finished[row_index] = True
        if all(finished):
            break
    return outputs


def apply_context_safety_gate(
    rows: Sequence[Mapping[str, Any]],
    predicted_tokens: Sequence[Sequence[str]],
) -> tuple[list[list[str]], int]:
    """Force an abstract abstain when observed context is evaluator-unknown.

    This is a runtime safety layer, not a model score and not a target-label
    lookup.  It uses only context tokens that an observer could have seen:
    unavailable typed evidence, unresolved feedback, and no consistent replay.
    The returned plan is still abstract and cannot authorize a send.
    """

    if len(rows) != len(predicted_tokens):
        raise ValueError("rows and predicted_tokens must have equal length")
    guarded: list[list[str]] = []
    changed = 0
    for row, predicted in zip(rows, predicted_tokens):
        context = {str(token).split("=", 1)[0]: str(token).split("=", 1)[1] for token in list(row.get("context_tokens") or []) if "=" in str(token)}
        requires_abstain = (
            context.get("typed_available") == "0"
            and context.get("feedback") == "unresolved"
            and context.get("candidate_sent") == "0"
            and context.get("replay_consistent") == "0"
        )
        if not requires_abstain:
            guarded.append([str(token) for token in predicted])
            continue
        fields, _ = parse_plan_tokens(predicted)
        method = fields.get("method", context.get("method", "GET"))
        if method not in ALLOWED_METHODS:
            method = "GET"
        guarded.append(
            [
                TARGET_BOS,
                "plan=abstain",
                f"method={method}",
                "probe_class=other",
                "channel=unknown",
                "encoding=unknown",
                "wire=none",
                "field_slot=none",
                "repair_delta=none",
                "family_agnostic=1",
                "final_action=abstain",
                "safe_to_send=0",
                TARGET_EOS,
            ]
        )
        changed += 1
    return guarded, changed


__all__ = ["SCHEMA_VERSION", "apply_context_safety_gate", "constrained_greedy_decode", "evaluate_decoded_plans", "parse_plan_tokens", "verify_plan_tokens"]
