import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg79_fresh_triplet_collection_passes_without_training_promotion():
    report = _read("pg79_fresh_unified_triplet_collector_report_v1.json")
    assert report["hard_gate"]["status"] == "passed"
    assert report["metrics"]["triplet_case_count"] == 270
    assert report["metrics"]["typed_positive_count"] == 240
    assert report["metrics"]["typed_negative_oracle_count"] == 540
    assert report["metrics"]["negative_probe_positive_count"] == 0
    assert report["metrics"]["unique_target_instance_count"] == 270
    assert report["metrics"]["get_post_counts"] == {"GET": 135, "POST": 135}
    assert report["metrics"]["source_count"] == 5
    assert report["metrics"]["family_count"] == 9
    assert report["metrics"]["trace_accepted_episode_count"] == 3
    assert report["finalization"]["network_replay"] is False
    assert report["finalization"]["oracle_values_changed"] is False
    assert report["promotion"]["training_allowed"] is False


def test_pg79_trace_and_catalog_are_raw_free():
    trace = _read("pg79_fresh_unified_triplet_collector_trace_v1.json")
    catalog = _read("pg79_fresh_unified_triplet_collector_catalog_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    assert len(trace["steps"]) == 270
    assert len(catalog["sources"][0]["samples"]) == 270
