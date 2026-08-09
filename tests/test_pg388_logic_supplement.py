from __future__ import annotations

import json
from pathlib import Path

from app.pg388_logic_invariant_projection import SUPPLEMENTAL_LOGIC_CASES, project_logic_case
from scripts.audit_pg388_logic_supplement_dataset import audit_dataset
from scripts.build_pg388_logic_supplement_dataset import build_dataset
from scripts.run_pg388_logic_supplement_replay import build_report


def test_supplement_projection_covers_taxonomy_gaps_and_stays_abstract() -> None:
    assert len(SUPPLEMENTAL_LOGIC_CASES) == 10
    assert {item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES} >= {
        "oauth_second_factor",
        "captcha_predictability",
        "session_forgery",
        "session_leakage",
    }
    for item in SUPPLEMENTAL_LOGIC_CASES:
        projection = project_logic_case(item["case_ref"], role="candidate", feedback_state="invariant_mismatch")
        assert projection["target_projection"]["safe_to_send"] is False
        text = json.dumps({"context": projection["context_tokens"], "target": projection["target_tokens"], "logic": projection["logic_context"]}, ensure_ascii=False).casefold()
        for marker in ("http://", "https://", "payload=", "wire:", "response_body:", "<script"):
            assert marker not in text


def test_supplement_dataset_is_balanced_and_candidate_only() -> None:
    artifact = build_dataset()
    assert artifact["counts"] == {
        "records": 1600,
        "train": 800,
        "implementation_holdout": 800,
        "cases": 10,
        "implementations": 2,
        "seeds": 4,
        "feedback_states": 5,
        "roles": 4,
    }
    assert audit_dataset(artifact)["status"] == "passed_candidate_audit"
    assert artifact["training_eligible"] == 0
    assert artifact["promotion"]["training_allowed"] is False


def test_supplement_replay_has_negative_and_fresh_gates() -> None:
    report = build_report()
    assert report["counts"] == {
        "cases": 10,
        "seeds": 3,
        "roles": 4,
        "episodes": 120,
        "typed_effect": 90,
        "negative_episodes": 30,
        "negative_violation": 0,
        "fresh_reset": 120,
    }
    assert report["safety"]["external_network"] is False
    assert report["safety"]["state_mutated"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False


def test_supplement_cpu_candidate_is_wiring_only() -> None:
    report = json.loads(Path("research/pg388_logic_supplement_token_cpu_smoke_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "cpu_smoke_candidate_only"
    assert report["execution"]["optimizer_started"] is True
    assert report["execution"]["device"] == "cpu"
    assert report["execution"]["gpu_touched"] is False
    assert report["execution"]["docker_started"] is False
    assert report["execution"]["network_contacted"] is False
    assert report["training_eligible"] == 0
    assert report["promotion"]["payload_catalog_promotion_allowed"] is False
