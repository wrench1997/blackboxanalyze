import json
from pathlib import Path

from app.pg100_semantic_sink_oracle import (
    evaluate_browser_pair,
    evaluate_redirect_pair,
    evaluate_sql_pair,
    model_visible_has_evaluator_label,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg100_fresh_local_replay_revalidates_all_typed_modalities_without_promotion():
    report = _load("pg100_semantic_sink_report_v1.json")
    protocol = _load("pg100_semantic_sink_protocol_v1.json")
    dataset = _load("pg100_semantic_sink_visible_dataset_v1.json")
    trace = _load("pg100_semantic_sink_trace_v1.json")
    assert report["status"] == "blocked"
    assert report["metrics"]["case_count"] == 7
    assert report["metrics"]["confirmed_positive_count"] == 7
    assert report["metrics"]["abstain_count"] == 0
    assert report["metrics"]["get_post_covered"] == {"GET": 6, "POST": 1}
    assert report["target"]["fresh_target_rounds"] == 7
    assert report["target"]["container_instance_count"] == 7
    assert len(set(report["target"]["container_ids"].values())) == 7
    assert report["metrics"]["positive_by_modality"] == {
        "browser_dom_execution": 3,
        "redirect_destination_controlled": 1,
        "sql_ast_differential": 3,
    }
    assert report["oracle_contract"]["old_pg52_labels_discarded_before_revalidation"] is True
    assert report["oracle_contract"]["oracle_is_evaluator_only"] is True
    assert report["capability_gate"]["checks"]["model_input_excludes_evaluator_labels"] is True
    assert report["capability_gate"]["checks"]["negative_control_matched"] is True
    assert report["capability_gate"]["checks"]["fresh_reset_per_case"] is True
    assert report["pg99_unidentifiability_control"]["impossibility_witness"] is True
    assert report["pg99_unidentifiability_control"]["unknown_family_strict_abstain"] is False
    assert report["training_boundary"]["training_eligible"] is False
    assert report["training_boundary"]["catalog_generated"] is False
    assert report["training_boundary"]["long_term_memory_write"] is False
    assert protocol["evaluation_boundary"]["training_allowed"] is False
    assert protocol["evaluation_boundary"]["memory_promotion_allowed"] is False
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert all(row["oracle_label_is_not_model_input"] for row in report["detection_results"])
    assert all(row["negative_control"]["matched"] for row in report["detection_results"])
    assert all(row["raw_payload_stored"] is False for row in report["detection_results"])
    assert all(row["raw_response_body_stored"] is False for row in report["detection_results"])
    assert all(len(row["evidence_sha256"]) == 64 for row in report["detection_results"])
    text = json.dumps(report, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onerror", "union select", "password=", "authorization: bearer"):
        assert forbidden not in text


def test_pg100_browser_oracle_abstains_when_negative_control_or_safety_fails():
    kwargs = dict(
        control_executed=True,
        candidate_executed=True,
        control_execution_path="automatic",
        candidate_execution_path="automatic",
        same_origin=True,
        external_request_count=0,
        navigation_count=0,
        mode="reflected_get",
        fresh_reset=True,
        evidence_sha256="a" * 64,
    )
    assert evaluate_browser_pair(**kwargs)["status"] == "abstain"
    kwargs["external_request_count"] = 1
    kwargs["control_executed"] = False
    assert evaluate_browser_pair(**kwargs)["status"] == "abstain"


def test_pg100_sql_and_redirect_oracles_are_strict_and_typed():
    base_ast = {"ast_sha256": "a", "statement_class": "select", "unsafe_operator_set": []}
    candidate_ast = {"ast_sha256": "b", "statement_class": "select", "unsafe_operator_set": []}
    control_response = {"result_row_count": 0, "semantic_body_sha256": "same"}
    negative_response = {"result_row_count": 0, "semantic_body_sha256": "same"}
    candidate_response = {"result_row_count": 1, "semantic_body_sha256": "different"}
    sql = evaluate_sql_pair(
        control_ast=base_ast,
        negative_ast={**base_ast, "ast_sha256": "c"},
        candidate_ast=candidate_ast,
        control_response=control_response,
        negative_response=negative_response,
        candidate_response=candidate_response,
        fresh_reset=True,
        evidence_sha256="b" * 64,
    )
    assert sql["status"] == "confirmed_positive"
    redirect = evaluate_redirect_pair(
        control_location="",
        candidate_location="http://127.0.0.1:8768/callback",
        control_status=200,
        candidate_status=302,
        expected_destination="http://127.0.0.1:8768/callback",
        fresh_reset=True,
        evidence_sha256="c" * 64,
    )
    assert redirect["status"] == "confirmed_positive"
    unsafe = evaluate_redirect_pair(
        control_location="",
        candidate_location="http://example.invalid/callback",
        control_status=200,
        candidate_status=302,
        expected_destination="http://example.invalid/callback",
        fresh_reset=True,
        evidence_sha256="d" * 64,
    )
    assert unsafe["status"] == "abstain"


def test_pg100_model_visible_contract_rejects_evaluator_fields():
    assert model_visible_has_evaluator_label({"status_class": "2xx", "shape": {"count": 3}}) is False
    assert model_visible_has_evaluator_label({"shape": {"oracle": "secret"}}) is True
    assert model_visible_has_evaluator_label({"candidate_family": "xss"}) is True
