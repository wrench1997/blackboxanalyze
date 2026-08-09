import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg78_multisource_holdout_preserves_contract_and_recall_failure():
    report = _read("pg78_multisource_triplet_holdout_report_v1.json")
    assert report["dataset"]["case_count"] == 270
    assert report["dataset"]["fresh_reset_count"] == 648
    assert report["dataset"]["source_count"] == 5
    assert report["dataset"]["implementation_count"] == 5
    assert report["dataset"]["family_count"] == 9
    assert report["dataset"]["method_counts"] == {"GET": 135, "POST": 135}
    assert report["dataset"]["screen_missing_count"] == 162
    assert report["metrics"]["confirm_recall"] == 0.0
    assert report["metrics"]["false_accept_count"] == 0
    assert report["capability_gate"]["checks"]["known_recall_min"] is False
    assert report["capability_gate"]["checks"]["screen_contract_complete"] is False
    assert report["capability_gate"]["status"] == "blocked"
    assert report["promotion"]["training_allowed"] is False


def test_pg78_trace_has_no_raw_persistence_or_training_updates():
    trace = _read("pg78_multisource_triplet_holdout_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert len(trace["rows"]) == 270
    assert all(row["raw_probe_stored"] is False and row["raw_response_stored"] is False for row in trace["rows"])
