from __future__ import annotations

import json
from pathlib import Path

from scripts.plan_pg374_model_selected_replay import (
    build_pg374_plan,
    select_staged_candidate,
    validate_pg374_plan,
)


def _full_rule_tokens(*, method: str = "GET", safe: str = "1") -> list[str]:
    return [
        "[TARGET_BOS]",
        "question=none",
        "ask_reason=none",
        "next_action=select_probe_variant",
        "repair_action=none",
        f"transport_ref={'get_query' if method == 'GET' else 'post_form'}",
        f"field_role_ref={'query_term' if method == 'GET' else 'form_field'}",
        f"encoding_ref={'identity' if method == 'GET' else 'form_urlencoded'}",
        "syntax_category_ref=marker",
        "probe_variant_ref=source_attested_candidate",
        f"safe_to_send={safe}",
        f"payload_shape_ref={'query_marker' if method == 'GET' else 'html_form_marker'}",
        "oracle_ref=response_shape",
        "negative_control_presence_ref=matched_triplet",
        "[TARGET_EOS]",
    ]


def test_staged_model_selection_is_not_typed_effect_or_wire():
    result = select_staged_candidate(_full_rule_tokens(), expected_method="GET", role="candidate")
    assert result["model_selected"] is True
    assert result["status"] == "blocked_missing_typed_evidence"
    assert result["typed_effect_confirmed"] is False
    assert result["wire_created"] is False
    assert result["target_contacted"] is False


def test_staged_model_abstention_is_explicit_and_safe():
    result = select_staged_candidate(_full_rule_tokens(method="POST", safe="0"), expected_method="POST", role="negative")
    assert result["model_selected"] is True
    assert result["status"] == "abstain_safe_to_send_false"
    assert result["safe_to_send"] is False
    assert result["wire_created"] is False


def test_incomplete_staged_output_becomes_ask_without_model_selection():
    result = select_staged_candidate(["[TARGET_BOS]", "question=ask_typed", "[TARGET_EOS]"], expected_method="GET", role="reference")
    assert result["status"] == "candidate_decode_incomplete"
    assert result["model_selected"] is False
    assert result["typed_effect_confirmed"] is False
    assert result["wire_created"] is False


def test_pg374_plan_preserves_get_post_roles_and_fresh_typed_contract():
    report = build_pg374_plan()
    assert validate_pg374_plan(report)["status"] == "passed"
    assert report["status"] == "planning_only_blocked"
    assert report["counts"] == {
        "seeds": 3,
        "routes": 2,
        "episodes": 6,
        "roles": 24,
        "get_rows": 12,
        "post_rows": 12,
        "candidate_rows": 6,
        "reference_rows": 6,
        "negative_rows": 6,
        "replay_rows": 6,
        "model_selected": 0,
        "typed_effect_confirmed": 0,
        "wire_created": 0,
        "target_contacted": 0,
    }
    assert report["staged_candidate"]["output_materialized"] is False
    assert report["staged_candidate"]["full_13_slot_output_materialized"] is False
    assert report["fresh_typed_replay_contract"]["observed_in_this_plan"] is False


def test_pg374_plan_has_no_raw_wire_or_url_content_and_promotion_is_closed():
    report = build_pg374_plan()
    text = str(report).casefold()
    assert "http://" not in text
    assert "https://" not in text
    assert "response_body" not in text
    assert "/webgoat" not in text
    assert all(value is False for key, value in report["promotion"].items() if key.endswith("_allowed"))


def test_pg373_report_is_the_locked_staged_candidate_source():
    data = json.loads((Path("research") / "pg373_staged_pretrain_candidate_v1.json").read_text(encoding="utf-8-sig"))
    report = build_pg374_plan()
    assert report["staged_candidate"]["schema_version"] == data["schema_version"]
    assert report["staged_candidate"]["candidate_seed_count"] == len(data["candidates"])
    assert report["staged_candidate"]["promotion_closed"] is True

