import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg63_frozen_model_transfers_after_canonical_surface_mapping():
    report = _read("pg63_independent_target_zone_report_v1.json")
    canonical = report["canonicalized_holdout"]
    raw = report["raw_shift_fail_closed"]
    assert report["status"] == "diagnostic_only"
    assert report["source"]["pg61_model_retrained"] is False
    assert report["source"]["pg61_task_generator_imported"] is False
    assert canonical["task_count"] == 192
    assert canonical["positive_action_accuracy"] == 1.0
    assert canonical["target_success_rate"] == 1.0
    assert canonical["negative_false_accept_count"] == 0
    assert canonical["unknown_strict_abstain"] is True
    assert canonical["selected_action_entropy"] >= 0.5
    assert raw["target_success_rate"] == 0.0
    assert raw["negative_false_accept_count"] == 0
    assert raw["raw_probe_stored_count"] == 0
    assert raw["raw_response_stored_count"] == 0
    assert report["hard_gate"]["status"] == "passed"
    assert report["hard_gate"]["claim_allowed"] is False


def test_pg63_independent_trace_keeps_oracle_after_action_and_hashes():
    trace = _read("pg63_independent_target_zone_trace_v1.json")
    assert trace["model_retrained_on_pg63"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    for group in (trace["canonicalized"], trace["raw_shift_fail_closed"]):
        assert group["step_count"] == 192
        for step in group["steps"]:
            assert step["reset"]["fresh_target"] is True
            assert step["reset"]["completed"] is True
            assert step["raw_probe_stored"] is False
            assert step["raw_response_stored"] is False
            assert step["online_weight_update"] is False
            assert step["long_term_memory_write"] is False
            assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_hash"])
            assert step["oracle_after_action"]["evaluator_state_hidden"] is True
    text = json.dumps(trace, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text


def test_pg63_protocol_makes_raw_shift_fail_closed_and_forbids_retraining():
    protocol = _read("pg63_independent_target_zone_protocol_v1.json")
    contract = protocol["independence_contract"]
    comparison = protocol["comparison"]
    assert protocol["authorized_scope"]["target_host"] == "127.0.0.1"
    assert protocol["authorized_scope"]["external_network"] is False
    assert contract["new_implementation"] is True
    assert contract["pg61_model_retraining_forbidden"] is True
    assert contract["pg61_task_generator_import_forbidden"] is True
    assert comparison["canonicalized_projection_required"] is True
    assert comparison["raw_shift_must_fail_closed"] is True
    assert comparison["typed_oracle_after_action_only"] is True
    assert protocol["run_result"]["status"] == "passed"
