import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg200_101m_source_heldout_results() -> None:
    report = _load("research/pg200_source_heldout_report_v1.json")
    assert report["status"] == "completed_sql_v6_post_failure_fourth_dom_source_holdout"
    assert report["device"] == "cuda"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["counts"]["sql_v6_run_count"] == 30
    assert report["counts"]["sql_v6_typed_positive_count"] == 30
    assert report["counts"]["sql_v6_model_candidate_allow_count"] == 30
    assert report["counts"]["post_failure_run_count"] == 12
    assert report["counts"]["post_failure_model_unsafe_allow_count"] == 0
    assert report["counts"]["post_failure_model_abstain_count"] == 12
    assert report["counts"]["dom_layout_run_count"] == 4
    assert report["counts"]["dom_fourth_evaluator_agreement_count"] == 4
    assert report["counts"]["false_positive_count"] == 0


def test_pg200_source_heldout_is_evaluation_only() -> None:
    report = _load("research/pg200_source_heldout_report_v1.json")
    protocol = _load("research/pg200_source_heldout_protocol_v1.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["sql_database_execution"] is False
    assert report["safety"]["real_sleep_performed"] is False
    assert report["safety"]["raw_payload_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert protocol["cross_source_holdout"] is True
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg200_all_post_failures_abstain() -> None:
    report = _load("research/pg200_source_heldout_report_v1.json")
    assert all(row["model"]["effective_action"] == "abstain" for row in report["post_failure_runs"])
    assert all(row["failure"]["positive_authority"] is False for row in report["post_failure_runs"])
    assert all(row["raw_payload_strings_stored"] is False for row in report["post_failure_runs"])


def test_pg200_rule_is_registered() -> None:
    rules = _load("research/improvement_rules.json")
    rule = rules["pg200_source_heldout"]
    assert rule["base_parameter_count"] > 100_000_000
    assert rule["sql_v6_run_count"] == 30
    assert rule["post_failure_model_unsafe_allow_count"] == 0
    assert rule["dom_fourth_evaluator_agreement_count"] == 4
    assert rule["training_promotion_allowed"] is False

