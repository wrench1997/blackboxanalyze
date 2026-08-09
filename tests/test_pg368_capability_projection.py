from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.pg368_capability_projection import SCHEMA_VERSION, load_pg368_capability, project_pg368_capability


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "research" / "pg367_model_binder_replay_report_v1.json"
WAF = ROOT / "research" / "pg367_waf_staircase_replay_report_v1.json"
PROCESS = ROOT / "research" / "pg367_a800_process_candidate_v2.json"
SFT = ROOT / "research" / "pg367_a800_rule_ir_sft_candidate_v1.json"


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg368_projection_has_model_metrics_slots_and_hard_false_promotion() -> None:
    result = load_pg368_capability(MODEL, WAF, process_report_path=PROCESS, sft_report_path=SFT)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "completed_demo_evidence"
    assert result["model_replay"]["counts"] == {
        "holdout_rows": 77,
        "decoded_exact": 51,
        "bindable": 35,
        "confirmed_positive": 35,
        "abstain": 42,
        "safe_abstain": 42,
        "unsafe_allow": 0,
        "evidence": 35,
        "fresh_replay": 35,
        "negative_clean": 35,
    }
    assert result["model_replay"]["rule_ir"]["unique_rule_ir_count"] == 25
    assert {item["transport_ref"] for item in result["model_replay"]["rule_ir"]["examples"]} == {"get_query", "post_form"}
    assert result["waf_replay"]["counts"]["candidate_typed"] == 12
    assert result["waf_replay"]["counts"]["negative_violation"] == 0
    assert result["scope"]["independent_second_implementation"] is False
    assert result["scope"]["general_vulnerability_claim"] is False
    assert all(value is False for value in result["promotion"].values())
    assert len(result["projection_sha256"]) == 64


def test_pg368_projection_does_not_expose_raw_wire_payload_or_urls() -> None:
    result = load_pg368_capability(MODEL, WAF, process_report_path=PROCESS, sft_report_path=SFT)
    text = json.dumps(result, ensure_ascii=False, sort_keys=True).casefold()
    for forbidden in (
        "pg367-runtime-canary",
        "http://127.0.0.1:",
        "response_body",
        "raw_payload",
        "oracle_answer",
        "evaluator_answer",
    ):
        assert forbidden not in text
    assert result["model_replay"]["scope"]["raw_wire_exposed"] is False
    assert result["waf_replay"]["scope"]["raw_wire_exposed"] is False


def test_pg368_projection_is_pure_and_does_not_mutate_reports() -> None:
    model = _load("pg367_model_binder_replay_report_v1.json")
    waf = _load("pg367_waf_staircase_replay_report_v1.json")
    model_before = copy.deepcopy(model)
    waf_before = copy.deepcopy(waf)
    result = project_pg368_capability(model, waf)
    assert model == model_before
    assert waf == waf_before
    assert result["training_candidates"] == []


def test_pg368_rejects_raw_canary_in_input_projection() -> None:
    model = {"rows": [{"rule_ir": {"transport_ref": "get_query", "field_role_ref": "query_term", "encoding_ref": "identity", "probe_variant_ref": "source_attested_candidate", "payload_shape_ref": "query_marker", "oracle_ref": "typed_effect", "syntax_category_ref": "marker", "safe_to_send": True}, "evidence_sha256": "a" * 64}], "counts": {}}
    waf = {"counts": {}}
    model["rows"][0]["debug"] = "pg367-runtime-canary"
    with pytest.raises(ValueError, match="raw wire/canary"):
        project_pg368_capability(model, waf)


def test_pg368_loader_is_bounded_and_requires_existing_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_pg368_capability(missing, WAF)
