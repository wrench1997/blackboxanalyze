import json
from pathlib import Path

from app.cross_lab_safe_catalog import validate_sample
from app.pg25d_acceptance_gate import evaluate_catalog
from app.pg25d_clickjacking_oracle import build_clickjacking_oracle, classify_frame_policy


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "3f13c399cebea2db7d40163529617814f82c44b480e791a0953dcbe4484b69f6"


def test_clickjacking_policy_is_bounded_and_typed():
    assert classify_frame_policy({"x-frame-options": "ALLOWALL"}) == "allowall"
    assert classify_frame_policy({"x-frame-options": "SAMEORIGIN"}) == "sameorigin"
    assert classify_frame_policy({"content-security-policy": "default-src 'self'; frame-ancestors 'none'"}) == "ancestors_none"
    oracle = build_clickjacking_oracle(
        oracle_contract_sha256=CONTRACT,
        frame_policy="allowall",
        expected_vulnerable=True,
    )
    assert oracle["positive"] is True
    assert oracle["positive_authority"] is True
    assert oracle["confirmed_effect"] == "frame_protection"


def test_clickjacking_catalog_has_typed_positive_but_no_training_promotion():
    catalog = json.loads(
        (ROOT / "research" / "pg_pk_25d_clickjacking_catalog_v1.json").read_text(encoding="utf-8")
    )
    candidate = next(row for row in catalog["samples"] if row["sample_role"] == "candidate")
    control = next(row for row in catalog["samples"] if row["sample_role"] == "negative_control")
    validate_sample(candidate, catalog["source"])
    validate_sample(control, catalog["source"])
    assert candidate["decision"]["evidence_status"] == "confirmed_positive"
    assert candidate["decision"]["training_action"] == "abstain"
    assert candidate["oracle_projection"]["confirmed_effect"] == "frame_protection"
    assert candidate["oracle_projection"]["signals"]["regex_evidence"]["pattern_id"] == "header_xfo_allowall"
    assert candidate["oracle_projection"]["signals"]["regex_evidence"]["matched"] is True
    assert candidate["negative_control"]["control_sample_id"] == control["sample_id"]
    report = evaluate_catalog(catalog)
    assert report["confirmed_positive_count"] == 1
    assert report["accepted_positive_count"] == 0
    assert report["training_eligible"] is False
