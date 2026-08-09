import json
from pathlib import Path

import pytest

from app.pg195_request_surface_adapter import build_surface_action_manifest, build_surface_values


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg195_get_post_layout_sql_holdout_report_v1.json"
TRACE = ROOT / "research" / "pg195_get_post_layout_sql_holdout_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg195_get_post_layout_sql_holdout_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg195_adapter_binds_both_methods_and_rejects_credential_fields() -> None:
    get_manifest = build_surface_action_manifest(
        path="/vul/xss/xss_01.php",
        method="GET",
        surface="xss01",
        field_names=["message", "submit"],
        probe_role="candidate",
        marker="pg195-test-marker",
    )
    post_manifest = build_surface_action_manifest(
        path="/vul/xss/xsspost/post_login.php",
        method="POST",
        surface="postlogin",
        field_names=["username", "submit"],
        probe_role="candidate",
        marker="pg195-test-marker",
    )
    assert get_manifest["method"] == "GET"
    assert get_manifest["placement"] == "query"
    assert post_manifest["method"] == "POST"
    assert post_manifest["placement"] == "form"
    assert post_manifest["form_field_names"] == ["submit", "username"]
    assert post_manifest["safety"]["no_database_write"] is True
    values = build_surface_values(field_names=["username", "submit"], probe_role="candidate", marker="pg195-test-marker")
    assert values["submit"] == "submit"
    assert "data-sift-marker" in values["username"]
    with pytest.raises(ValueError):
        build_surface_action_manifest(
            path="/vul/xss/xsspost/post_login.php",
            method="POST",
            surface="postlogin",
            field_names=["password", "submit"],
            probe_role="candidate",
            marker="pg195-test-marker",
        )


def test_pg195_xxl_gate_and_get_post_matrix_are_real_replays() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_xxl_get_post_matrix_and_independent_sql_holdout"
    assert report["model"]["variant"] == "xxl"
    assert report["model"]["parameter_count"] > 100_000_000
    assert report["model"]["online_weight_update"] is False
    assert report["gate_training"]["holdout"]["accuracy"] == 1.0
    assert report["gate_training"]["holdout"]["allow_candidate_recall"] == 1.0
    assert report["gate_training"]["holdout"]["unsafe_allow_count"] == 0
    assert len(report["route_runs"]) == 18
    assert report["counts"]["fresh_container_count"] == 3
    assert report["counts"]["get_send_count"] == 36
    assert report["counts"]["post_send_count"] == 18
    assert report["counts"]["send_count"] == 54
    assert protocol["methods"] == ["GET", "POST"]
    assert protocol["selected_surface_count"] == 6
    assert protocol["fresh_container_per_seed"] is True


def test_pg195_unknown_pikachu_sql_abstains_and_dom_effects_are_not_xss() -> None:
    report = _load(REPORT)
    runs = report["route_runs"]
    assert {row["method"] for row in runs} == {"GET", "POST"}
    assert len({row["seed"] for row in runs}) == 3
    assert all(row["fresh_container"] for row in runs)
    assert all(row["confirmed_positive"] is False for row in runs)
    assert all(row["vulnerability_claim_allowed"] is False for row in runs)
    xss = [row for row in runs if row["family"] == "xss"]
    sql_unknown = [row for row in runs if row["family"] == "sql_unknown"]
    assert len(xss) == 12
    assert all(row["typed_oracle_available"] for row in xss)
    assert all(any(step["controller_decision"].startswith("send_evaluator_aware_candidate_") for step in row["steps"]) for row in xss)
    assert len(sql_unknown) == 6
    assert all(row["typed_oracle_available"] is False for row in sql_unknown)
    assert all(any(step.get("abstain_reason") == "pikachu_surface_oracle_unknown" for step in row["steps"]) for row in sql_unknown)
    assert report["counts"]["dom_typed_surface_effect_count"] == 3
    assert report["counts"]["pikachu_confirmed_positive_count"] == 0
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span data-sift-marker" not in serialized
    assert "response_body" not in serialized


def test_pg195_independent_sql_v4_has_get_post_typed_holdout_only() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    sql_runs = report["sql_runs"]
    assert len(sql_runs) == 3
    assert {row["variant"] for row in sql_runs} == {"delta", "epsilon", "zeta"}
    assert all(row["fresh_target"] for row in sql_runs)
    assert all(row["implementation"] == "independent_shape_only_v4" for row in sql_runs)
    assert all(len(row["runs"]) == 2 for row in sql_runs)
    assert all({method_run["method"] for method_run in row["runs"]} == {"GET", "POST"} for row in sql_runs)
    assert all(row["typed_positive_count"] == 2 for row in sql_runs)
    assert report["counts"]["sql_get_post_typed_positive_count"] == 6
    assert all(row["vulnerability_claim_allowed"] is False for row in sql_runs)
    assert protocol["independent_sql_oracle"] == "synthetic_sql_shape_differential_v4"
    assert protocol["fresh_sql_target_per_variant"] is True
    assert protocol["typed_oracle_required_before_positive"] is True


def test_pg195_keeps_training_and_raw_material_quarantined_and_rule_matches() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    rules = _load(ROOT / "research" / "improvement_rules.json")
    for section in (report["promotion"], report["safety"]):
        assert section["raw_payload_strings_stored"] is False
        assert section["raw_response_bodies_stored"] is False
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    rule = rules["pg195_get_post_layout_sql_holdout"]
    assert rule["selected_surface_count"] == 6
    assert rule["route_replay_count"] == 18
    assert rule["get_send_count"] == 36
    assert rule["post_send_count"] == 18
    assert rule["pikachu_confirmed_positive_count"] == 0
    assert rule["false_positive_count"] == 0
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
    assert rule["vulnerability_claim_allowed"] is False
