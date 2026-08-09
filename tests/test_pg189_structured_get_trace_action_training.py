import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg189_structured_get_trace_action_training_report_v1.json"
TRACE = ROOT / "research" / "pg189_structured_get_trace_action_training_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg189_structured_get_trace_action_training_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg189_has_nonempty_route_disjoint_holdout_and_forgetting_gate() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_structured_real_get_trace_action_training"
    assert report["training_rows"] == 225
    assert report["holdout_rows"] == 120
    assert report["row_stats"]["holdout_surfaces"] == ["xss_01_unseen_get", "xss_04_unseen_get"]
    assert protocol["route_holdout"] == ["xss_01_unseen_get", "xss_04_unseen_get"]
    assert protocol["gates"]["catastrophic_forgetting_blocks_selection"] is True
    assert all(row["holdout"]["count"] == 120 for row in report["results"])


def test_pg189_action_result_is_not_promoted_as_payload_or_vulnerability_capability() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["selection"]["selected_variant"] == "large"
    assert report["selection"]["training_artifact_promotion_allowed"] is False
    assert report["selection"]["memory_promotion_allowed"] is False
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert report["safety"]["raw_payloads_in_model"] is False
    assert report["safety"]["raw_responses_in_model"] is False
    assert report["safety"]["vulnerability_label_in_input"] is False
    assert trace["training_artifact_promotion_allowed"] is False
    assert trace["memory_promotion_allowed"] is False
    assert protocol["raw_payload_and_response_excluded"] is True


def test_pg189_reports_abstain_recall_as_true_positive_recall() -> None:
    report = _load(REPORT)
    for result in report["results"]:
        holdout = result["holdout"]
        assert holdout["expected_abstain_count"] > 0
        assert holdout["abstain_true_positive_count"] <= holdout["expected_abstain_count"]
        assert 0.0 <= holdout["abstain_recall"] <= 1.0
        assert holdout["false_candidate_on_abstain_count"] == 0


def test_pg189_rule_keeps_structured_get_diagnostic_separate_from_payload_claim() -> None:
    rules = _load(ROOT / "research" / "improvement_rules.json")
    rule = rules["pg189_structured_get_trace_action_training"]
    assert rule["target_route_excluded_from_training"] is True
    assert rule["structured_policy_labels_only"] is True
    assert rule["raw_exploit_string_generation"] is False
    assert rule["typed_effect_is_not_vulnerability_positive"] is True
    assert rule["selection_gate"]["vulnerability_claim_allowed"] is False
    assert rule["unknown_oracle_action"] == "abstain"
