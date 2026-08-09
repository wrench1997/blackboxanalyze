"""Fail-closed guard for bounded Rule-IR action contracts.

The guard is an engineering safety layer, not a replacement for the model:
it records which slot=value pairs were present in the training split and
enforces typed-positive/candidate progress contracts.  Unseen current pairs
cannot cause an unsafe repeat/stop; an allow-listed alternate-method probe is
preserved when it is safe, otherwise the guard abstains.  Unknown typed
availability always maps to ``abstain_unknown_oracle``.  No raw source,
response, target name, or evaluator action is inspected.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


SAFE_ABSTAIN_ACTIONS = frozenset({"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pair(slot: Any, value: Any) -> str:
    return f"{str(slot)[:64]}={str(value)[:64]}"


def iter_rule_ir_pairs(row: Mapping[str, Any]) -> Iterable[str]:
    for step in list(row.get("layered_steps") or []):
        for token in list((step.get("ir_layer") or {}).get("tokens") or []):
            if isinstance(token, Mapping):
                yield _pair(token.get("slot_id", "unknown"), token.get("value", "unknown"))


def iter_current_rule_ir_pairs(row: Mapping[str, Any]) -> Iterable[str]:
    """Yield only the current observation; history remains available to GRU."""

    steps = list(row.get("layered_steps") or [])
    if not steps:
        return
    current = steps[-1]
    for token in list((current.get("ir_layer") or {}).get("tokens") or []):
        if isinstance(token, Mapping):
            yield _pair(token.get("slot_id", "unknown"), token.get("value", "unknown"))


def known_rule_ir_pairs(rows: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(pair for row in rows for pair in iter_rule_ir_pairs(row))


def unseen_rule_ir_pairs(row: Mapping[str, Any], known_pairs: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted({pair for pair in iter_current_rule_ir_pairs(row) if pair not in known_pairs}))


def known_pairs_sha256(known_pairs: frozenset[str]) -> str:
    return hashlib.sha256(_canonical(sorted(known_pairs)).encode("utf-8")).hexdigest()


def guard_action(action: str, row: Mapping[str, Any], known_pairs: frozenset[str]) -> tuple[str, dict[str, Any]]:
    """Return a fail-closed action and bounded reason metadata."""

    signature = dict(row.get("failure_signature") or {})
    if not bool(signature.get("typed_available", True)):
        return "abstain_unknown_oracle", {"guarded": action != "abstain_unknown_oracle", "reason": "typed_oracle_unavailable", "unseen_pairs": []}
    unseen = unseen_rule_ir_pairs(row, known_pairs)
    kind = str(signature.get("kind", ""))
    gate = str(signature.get("failed_gate", ""))
    methods = {str(item).upper() for item in signature.get("methods_seen", [])}
    # Enforce the typed-positive safety contract even when the Rule-IR pair
    # is known.  A model may abstain or repeat on a familiar token, but a
    # typed positive is only safe to probe until both methods are observed;
    # after that the only terminal action is the typed-positive stop.
    if kind == "typed_positive":
        if len(methods) < 2:
            if action == "probe_candidate_other_method":
                return str(action), {"guarded": False, "reason": "unseen_typed_positive_progress" if unseen else "typed_positive_progress", "unseen_pairs": list(unseen[:8])}
            return "probe_candidate_other_method", {"guarded": True, "reason": "unseen_typed_positive_progress" if unseen else "typed_positive_progress", "unseen_pairs": list(unseen[:8])}
        if action == "stop_confirmed_positive":
            return str(action), {"guarded": False, "reason": "unseen_typed_positive_confirmed" if unseen else "typed_positive_confirmed", "unseen_pairs": list(unseen[:8])}
        return "stop_confirmed_positive", {"guarded": True, "reason": "unseen_typed_positive_confirmed" if unseen else "typed_positive_confirmed", "unseen_pairs": list(unseen[:8])}
    if kind == "no_surface_delta" and gate == "matched_negative_control":
        if action == "repeat_matched_negative_pair":
            return str(action), {"guarded": False, "reason": "matched_negative_control", "unseen_pairs": list(unseen[:8])}
        return "repeat_matched_negative_pair", {"guarded": True, "reason": "matched_negative_control", "unseen_pairs": list(unseen[:8])}
    if kind == "candidate_without_typed_effect":
        remaining = int(signature.get("remaining_probe_budget", 0) or 0)
        expected = "probe_candidate_other_method" if len(methods) < 2 or remaining > 0 else "abstain_candidate_only"
        if action == expected:
            return str(action), {"guarded": False, "reason": "candidate_progress_contract", "unseen_pairs": list(unseen[:8])}
        return expected, {"guarded": True, "reason": "candidate_progress_contract", "unseen_pairs": list(unseen[:8])}
    if unseen and action not in SAFE_ABSTAIN_ACTIONS:
        # A typed candidate/positive with an allow-listed alternate-method
        # probe can still make safe progress. The guard blocks only actions
        # that would stop or repeat on an unseen abstract surface.
        if action == "probe_candidate_other_method" and kind in {"candidate_without_typed_effect"}:
            return str(action), {"guarded": False, "reason": "unseen_rule_ir_pair_probe_allowed", "unseen_pairs": list(unseen[:8])}
        return "abstain_candidate_only", {"guarded": True, "reason": "unseen_rule_ir_pair", "unseen_pairs": list(unseen[:8])}
    return str(action), {"guarded": False, "reason": "none", "unseen_pairs": list(unseen[:8])}


__all__ = ["SAFE_ABSTAIN_ACTIONS", "guard_action", "iter_current_rule_ir_pairs", "iter_rule_ir_pairs", "known_pairs_sha256", "known_rule_ir_pairs", "unseen_rule_ir_pairs"]
