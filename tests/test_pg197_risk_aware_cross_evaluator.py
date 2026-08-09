import json
from pathlib import Path

from app.pg197_alt_dom_oracle import run_alt_dom_oracle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg197_risk_aware_cross_evaluator_report_v1.json"
TRACE = ROOT / "research" / "pg197_risk_aware_cross_evaluator_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg197_risk_aware_cross_evaluator_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg197_static_dom_evaluator_is_independent_and_non_executing() -> None:
    positive = run_alt_dom_oracle('<main><span data-sift-marker="pg197-test">pg197-test</span></main>', marker="pg197-test")
    script = run_alt_dom_oracle('<script>pg197-test</script>', marker="pg197-test")
    assert positive["oracle_id"] == "pg197-static-dom-parser-v1"
    assert positive["dom_change"] is True
    assert positive["script_execution"] is False
    assert positive["raw_markup_stored"] is False
    assert script["dom_change"] is False
    assert script["script_execution"] is False


def test_pg197_xxl_risk_decoder_and_learned_gate_are_scored_separately() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_risk_aware_decoder_and_dual_evaluator_replay"
    assert report["model"]["variant"] == "xxl"
    assert report["model"]["base_parameter_count"] > 100_000_000
    raw = report["decoder_training"]["holdout"]
    assert raw["raw_unsafe_allow_count"] == 8
    assert raw["raw_safe_candidate_recall"] == 0.59259259
    assert raw["gate_accuracy"] == 1.0
    assert raw["gate_allow_recall"] == 1.0
    assert raw["gate_unsafe_allow_count"] == 0
    assert raw["gated_unsafe_allow_count"] == 0
    assert protocol["action_decoder"] == "risk_aware_action_plus_candidate_gate"
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg197_pikachu_dual_dom_and_unknown_oracle_routes() -> None:
    report = _load(REPORT)
    runs = report["route_runs"]
    assert len(runs) == 9
    assert len({row["seed"] for row in runs}) == 3
    dom = [row for row in runs if row["family"] == "xss"]
    unknown = [row for row in runs if row["family"] != "xss"]
    assert len(dom) == 3
    assert all(row["fresh_container"] for row in dom)
    assert all(row["typed_oracle_available"] for row in dom)
    assert all(row["dual_dom_effect_agreement"] for row in dom)
    assert all(row["typed_surface_effect"] for row in dom)
    assert all(row["confirmed_positive"] is False for row in dom)
    assert all(any(step["controller_decision"].startswith("send_dual_dom_evaluator_candidate_") for step in row["steps"]) for row in dom)
    assert len(unknown) == 6
    assert all(row["typed_oracle_available"] is False for row in unknown)
    assert all(any(step.get("abstain_reason") == "pikachu_surface_oracle_unknown" for step in row["steps"]) for row in unknown)
    assert report["counts"]["dom_dual_agreement_count"] == 3
    assert report["counts"]["unknown_oracle_abstain_count"] == 6
    assert report["counts"]["false_positive_count"] == 0


def test_pg197_sql_v4_v5_get_post_source_agreement() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    pairs = report["sql_pairs"]
    assert len(pairs) == 3
    assert {row["v4_variant"] for row in pairs} == {"delta", "epsilon", "zeta"}
    assert {row["v5_variant"] for row in pairs} == {"indigo", "jade", "krypton"}
    assert all(row["agreement_count"] == 2 for row in pairs)
    assert all(len(row["runs"]) == 2 for row in pairs)
    assert all({run["method"] for run in row["runs"]} == {"GET", "POST"} for row in pairs)
    assert report["counts"]["sql_get_post_agreement_count"] == 6
    assert report["counts"]["sql_typed_positive_count"] == 6
    assert protocol["sql_evaluators"] == ["synthetic_sql_shape_differential_v4", "synthetic_sql_shape_differential_v5"]


def test_pg197_promotion_and_raw_material_are_quarantined() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    rules = _load(ROOT / "research" / "improvement_rules.json")
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span data-sift-marker" not in serialized
    assert "response_body" not in serialized
    rule = rules["pg197_risk_aware_cross_evaluator"]
    assert rule["raw_action_non_authoritative"] is True
    assert rule["learned_gate_holdout"]["unsafe_allow_count"] == 0
    assert rule["dom_dual_evaluator_agreement_count"] == 3
    assert rule["sql_v4_v5_get_post_agreement_count"] == 6
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
    assert rule["vulnerability_claim_allowed"] is False
