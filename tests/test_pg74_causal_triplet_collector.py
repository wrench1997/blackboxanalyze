import json
from pathlib import Path

from app.payload_catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg74_triplet_collection_passes_without_training_promotion():
    report = _read("pg74_causal_triplet_collector_report_v1.json")
    assert report["status"] == "completed_evaluation"
    assert report["source"]["seeds_requested"] == [74101, 74102, 74103]
    assert report["metrics"]["triplet_case_count"] == 21
    assert report["metrics"]["neutral_projection_count"] == 21
    assert report["metrics"]["negative_probe_projection_count"] == 21
    assert report["metrics"]["positive_probe_projection_count"] == 21
    assert report["metrics"]["typed_positive_count"] == 21
    assert report["metrics"]["typed_negative_oracle_count"] == 42
    assert report["metrics"]["unique_target_instance_count"] == 21
    assert report["hard_gate"]["status"] == "passed"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg74_trace_persists_triplet_projections_without_raw_values():
    trace = _read("pg74_causal_triplet_collector_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["accepted_episode_count"] == 3
    assert trace["validation_failures"] == []
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert all(all(key in step for key in ("neutral_projection", "negative_probe_projection", "response_projection", "neutral_oracle_projection", "negative_oracle_projection", "oracle_projection")) for step in trace["steps"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onload", "onerror", "union select", "password", "123456"):
        assert forbidden not in serialized
    catalog = load_catalog(ROOT / "research/pg74_causal_triplet_collector_catalog_v1.json")
    assert sum(len(source["samples"]) for source in catalog["sources"]) == 21
    assert catalog["safety"]["real_exploit_strings"] is False
