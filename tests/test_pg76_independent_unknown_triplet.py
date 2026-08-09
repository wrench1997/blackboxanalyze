import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg76_independent_unknown_triplet_gate_is_passed_but_evaluation_only():
    report = _read("pg76_independent_unknown_triplet_report_v1.json")
    assert report["hard_gate"]["status"] == "passed"
    assert report["metrics"]["triplet_case_count"] == 12
    assert report["metrics"]["typed_positive_count"] == 12
    assert report["metrics"]["typed_negative_oracle_count"] == 24
    assert report["metrics"]["model_unknown_misname_count"] == 0
    assert report["metrics"]["model_unknown_strict_abstain"] is True
    assert report["source"]["family_outside_training_registry"] is True
    assert report["hard_gate"]["claim_allowed"] is False


def test_pg76_trace_is_raw_free_and_not_training_eligible():
    trace = _read("pg76_independent_unknown_triplet_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert len(trace["steps"]) == 12
