import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg70_keeps_unknown_family_out_of_training_and_reports_holdout_failure():
    report = _read("pg70_trace_abstention_head_report_v1.json")
    assert report["status"] == "candidate_training_completed"
    assert report["source"]["model_retrained_on_unknown_family"] is False
    assert report["source"]["family_in_features"] is False
    assert report["source"]["oracle_in_features"] is False
    assert report["dataset"]["known_train_example_count"] == 4
    assert report["dataset"]["known_dev_holdout_example_count"] == 4
    assert report["dataset"]["unknown_family_holdout_count"] == 8
    assert report["metrics"]["unknown_family_holdout"]["misname_count"] == 0
    assert report["metrics"]["unknown_family_holdout"]["strict_abstain"] is True
    assert report["capability_gate"]["status"] == "blocked"
    assert report["capability_gate"]["checks"]["dev_confirm_recall"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert (ROOT / report["training"]["checkpoint"]).exists()


def test_pg70_training_trace_contains_only_projection_and_hash_metadata():
    trace = _read("pg70_trace_abstention_head_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_unknown_family"] is False
    assert trace["family_in_features"] is False
    assert trace["oracle_in_features"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    for forbidden in ("workflow_invariant", "xss", "injection", "url_redirect", "<script", "union select", "onload"):
        assert forbidden not in serialized
