import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg52_real_local_oracles_confirm_effects_without_raw_payloads():
    report = _load("pg52_authoritative_local_oracle_report_v1.json")
    assert report["status"] == "real_local_oracle_completed"
    assert report["metrics"]["case_count"] == 7
    assert report["metrics"]["confirmed_positive_count"] == 7
    assert report["metrics"]["confirmed_positive_by_family"] == {"injection": 3, "url_redirect": 1, "xss": 3}
    assert report["metrics"]["get_post_covered"] == {"GET": 6, "POST": 1}
    assert report["oracle_contract"]["browser_execution"] is True
    assert report["oracle_contract"]["browser_offline_response_renderer"] is True
    assert report["oracle_contract"]["static_resource_tags_stripped"] is True
    assert report["oracle_contract"]["controlled_event_dispatch_is_explicit"] is True
    assert report["oracle_contract"]["sql_ast_differential"] is True
    assert report["oracle_contract"]["controlled_redirect"] is True
    assert report["oracle_contract"]["positive_requires_negative_control"] is True
    assert report["oracle_contract"]["positive_requires_fresh_reset"] is True
    assert report["oracle_contract"]["positive_requires_evidence_hash"] is True
    assert all(row["decision"] == "confirmed_positive" for row in report["detection_results"])
    assert all(row["oracle"]["positive_authority"] for row in report["detection_results"])
    assert all(row["fresh_reset"]["fresh_target"] and row["fresh_reset"]["completed"] for row in report["detection_results"])
    assert all(row["negative_control"]["matched"] for row in report["detection_results"])
    assert all(row["rule_ir_binding"]["executable"] is False for row in report["detection_results"])
    assert report["metrics"]["browser_execution_paths"] == {"automatic": 2, "controlled_event_dispatch": 1}
    assert all(len(row["evidence_sha256"]) == 64 for row in report["detection_results"])


def test_pg52_oracles_are_typed_and_model_error_is_visible():
    report = _load("pg52_authoritative_local_oracle_report_v1.json")
    modalities = {row["oracle"]["modality"] for row in report["detection_results"]}
    assert modalities == {"browser_dom_execution", "sql_ast_differential", "redirect_destination_controlled"}
    assert report["metrics"]["oracle_family_binding_match_count"] == 7
    assert report["metrics"]["model_family_match_count"] == 3
    assert report["metrics"]["model_family_misclassification_count"] == 4
    assert all(row["confirmed_family"] == row["family"] for row in report["detection_results"])
    serialized = json.dumps(report, ensure_ascii=False).casefold()
    for forbidden in ("<svg", "onload", "union select", "pg52missing", "123456", "password"):
        assert forbidden not in serialized


def test_pg52_training_and_memory_are_still_quarantined():
    report = _load("pg52_authoritative_local_oracle_report_v1.json")
    protocol = _load("pg52_authoritative_local_oracle_protocol_v1.json")
    assert report["training_boundary"]["training_eligible"] is False
    assert report["training_boundary"]["catalog_generated"] is False
    assert report["training_boundary"]["long_term_memory_write"] is False
    assert report["formal_claim"]["allowed"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["run_result"]["memory_promotion_allowed"] is False
