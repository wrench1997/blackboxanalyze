from __future__ import annotations

import json
from pathlib import Path


def test_stateful_disposable_exception_requires_full_role_lifecycle_and_keeps_promotion_closed() -> None:
    rules = json.loads((Path("research") / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    policy = rules["research_execution_policy_v1"]["stateful_persistent_evaluator_exception_v1"]
    lifecycle = set(policy["required_lifecycle"])
    assert {
        "fresh container per seed/route/role", "state_reset_before_role",
        "database_clean_attestation", "candidate/reference/negative/replay typed evaluator",
        "role_bound_evidence_sha256", "state_reset_after_role", "teardown or reset",
    } <= lifecycle
    assert {"external_network", "bind_mount_or_volume", "raw_payload_or_response_in_model_context"} <= set(policy["forbidden"])
    assert "ASK/safe_to_send=false" in policy["training_and_memory"]


def test_framework_fast_lane_defers_only_diagnostic_details_and_never_bypasses_hard_gates() -> None:
    rules = json.loads((Path("research") / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    lane = rules["research_execution_policy_v1"]["framework_first_fast_lane_v1"]
    assert "diagnostic/unknown/incomplete" in lane["fast_result_semantics"]
    assert {"authorization", "fresh_reset_for_live_state", "context_firewall", "split_isolation", "capacity_no_truncation"} <= set(lane["never_bypass"])


def test_pg339_points_to_latest_and_requires_new_hash_lock_for_future_run() -> None:
    rules = json.loads((Path("research") / "improvement_rules.json").read_text(encoding="utf-8-sig"))
    pg339 = rules["research_execution_policy_v1"]["pg339_multi_shape_information_preserving_v1"]
    assert pg339["a800"]["latest_report"] == "research/pg339_a800_multi_shape_representation_e6_v1.json"
    assert "SHA-256" in pg339["rules_hash_lock_note"]
    assert all(value is False for value in pg339["promotion"].values())
