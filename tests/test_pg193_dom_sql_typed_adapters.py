import json
from pathlib import Path

from app.pg193_browser_dom_oracle import run_browser_dom_oracle


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg193_dom_sql_typed_adapters_report_v1.json"
TRACE = ROOT / "research" / "pg193_dom_sql_typed_adapters_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg193_dom_sql_typed_adapters_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_browser_dom_oracle_disables_js_and_reports_only_bounded_effect() -> None:
    result = run_browser_dom_oracle('<html><body><span data-sift-marker="pg193-test">x</span></body></html>', marker="pg193-test")
    assert result["oracle_id"] == "pg193-browser-dom-nojs-v1"
    assert result["browser_dom_observed"] is True
    assert result["dom_change"] is True
    assert result["marker_hits"] == 1
    assert result["script_execution"] is False
    assert result["network_access"] is False
    assert result["raw_markup_stored"] is False
    assert result["evidence_hash"]


def test_pg193_pikachu_dom_effect_is_not_xss_positive_and_sql_fixture_is_separate() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    dom = next(row for row in report["runs"] if row["surface"] == "pg193_xss_reflected_get")
    sql = next(row for row in report["runs"] if row["surface"] == "pg193_sql_v3_fixture")
    assert dom["typed_oracle_available"] is True
    assert dom["typed_surface_effect"] is True
    assert dom["confirmed_positive"] is False
    assert all(step["vulnerability_claim_allowed"] is False for step in dom["steps"])
    assert sql["target"] == "http://127.0.0.1:8810"
    assert sql["typed_positive_count"] == 1
    assert sql["vulnerability_claim_allowed"] is False
    assert protocol["sql_adapter_target"].startswith("independent loopback")
    assert protocol["dom_script_execution_forbidden"] is True
    assert protocol["sql_database_execution_forbidden"] is True


def test_pg193_keeps_raw_material_and_promotion_quarantined() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    rules = _load(ROOT / "research" / "improvement_rules.json")
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False
    assert trace["training_eligible"] is False
    assert trace["raw_payload_strings_stored"] is False
    rule = rules["pg193_dom_sql_typed_adapters"]
    assert rule["observed_result"]["vulnerability_claim_allowed"] == 0
    assert rule["dom_oracle"]["effect_is_not_xss_positive"] is True
    assert rule["sql_oracle"]["fixture_is_not_pikachu_backend"] is True
    assert rule["training_promotion_allowed"] is False
