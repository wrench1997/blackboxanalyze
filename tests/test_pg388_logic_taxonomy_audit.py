from __future__ import annotations

from scripts.audit_pg388_logic_taxonomy import build_audit


def test_pg388_taxonomy_has_all_primary_category_anchors() -> None:
    report = build_audit()
    assert report["status"] == "passed_candidate_coverage_all_anchors"
    assert report["case_count"] == 66
    assert report["core_case_count"] == 56
    assert report["supplemental_case_count"] == 10
    assert report["missing_anchor_count"] == 0
    assert report["diagnostic_gap_count"] == 0
    assert report["candidate_only_count"] == 10
    assert report["training_eligible"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_pg388_taxonomy_keeps_supplemental_contracts_candidate_only() -> None:
    report = build_audit()
    candidate = {(item["category"], item["case_ref"]) for item in report["candidate_only_contracts"]}
    assert ("two_factor", "oauth_second_factor") in candidate
    assert ("captcha", "captcha_client_validation") in candidate
    assert ("session_randomness_other", "session_forgery") in candidate
    assert report["diagnostic_gaps"] == []
