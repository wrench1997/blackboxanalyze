from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.rule_ir_ood_guard import guard_action, known_rule_ir_pairs


def _row(kind: str, action: str, methods: list[str], value: str = "authorization") -> dict[str, object]:
    return {
        "failure_signature": {"kind": kind, "typed_available": True, "methods_seen": methods},
        "layered_steps": [{"ir_layer": {"tokens": [{"slot_id": "response.transition_delta", "value": value}]}}],
        "label": action,
    }


def test_ood_guard_is_fail_closed_but_preserves_safe_progress() -> None:
    known = known_rule_ir_pairs([_row("no_surface_delta", "repeat_matched_negative_pair", ["GET"], value="none")])
    guarded, reason = guard_action("repeat_matched_negative_pair", _row("candidate_without_typed_effect", "probe_candidate_other_method", ["GET"], value="authorization"), known)
    assert guarded == "probe_candidate_other_method"
    assert reason["reason"] == "candidate_progress_contract"
    guarded, reason = guard_action("repeat_matched_negative_pair", _row("typed_positive", "probe_candidate_other_method", ["GET"], value="authorization"), known)
    assert guarded == "probe_candidate_other_method"
    assert reason["reason"] == "unseen_typed_positive_progress"
    guarded, reason = guard_action("stop_confirmed_positive", _row("typed_positive", "stop_confirmed_positive", ["GET", "POST"], value="authorization"), known)
    assert guarded == "stop_confirmed_positive"
    assert reason["reason"] == "unseen_typed_positive_confirmed"


def test_pg135_report_is_exactly_balanced_and_guarded() -> None:
    report = json.loads(Path("research/pg135_balanced_policy_report_v1.json").read_text(encoding="utf-8"))
    assert report["hard_gates_passed"] is True
    assert report["training_eligible"] is False
    assert report["transport_balance"] == {"get_count": 132, "post_count": 132, "exact": True}
    raw = report["holdout"]["pg122_family_ood"]
    guarded = report["holdout"]["pg122_family_ood_guarded"]
    assert raw["safety_compliance_rate"] == 0.9375
    assert guarded["safety_compliance_rate"] == 1.0
    assert guarded["metrics"]["accuracy"] == 1.0
    assert guarded["guard_override_count"] == 3
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg135_report_hash_is_recomputable() -> None:
    report = json.loads(Path("research/pg135_balanced_policy_report_v1.json").read_text(encoding="utf-8"))
    declared = report.pop("report_sha256")
    actual = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
