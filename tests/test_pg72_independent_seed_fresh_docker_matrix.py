import json
import re
from pathlib import Path

from app.payload_catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg72_collection_gate_passes_but_frozen_head_capability_gate_blocks():
    report = _read("pg72_independent_seed_fresh_docker_matrix_report_v1.json")
    assert report["status"] == "completed_evaluation"
    assert report["source"]["seeds_complete"] == [72101, 72102, 72103]
    assert report["metrics"]["typed_positive_count"] == 21
    assert report["metrics"]["negative_control_pass_count"] == 21
    assert report["metrics"]["evidence_hash_valid_count"] == 21
    assert report["metrics"]["unique_candidate_target_instance_count"] == 21
    assert report["metrics"]["fresh_reset_per_action"] is True
    assert report["metrics"]["trace_accepted_episode_count"] == 3
    assert report["metrics"]["frozen_head"]["confirm_recall"] == 0.0
    assert report["metrics"]["frozen_head"]["false_accept_count"] == 0
    assert report["hard_gate"]["checks"]["frozen_known_confirm_recall"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg72_catalog_and_trace_are_evaluation_only_and_raw_free():
    catalog = load_catalog(ROOT / "research/pg72_independent_seed_fresh_docker_matrix_catalog_v1.json")
    assert catalog["catalog_id"] == "pg72-independent-seed-fresh-docker-evaluation-only"
    assert sum(len(source["samples"]) for source in catalog["sources"]) == 21
    assert catalog["safety"]["real_exploit_strings"] is False
    trace = _read("pg72_independent_seed_fresh_docker_matrix_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["accepted_episode_count"] == 3
    assert trace["validation_failures"] == []
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    serialized = json.dumps({"catalog": catalog, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onload", "onerror", "union select", "password", "123456"):
        assert forbidden not in serialized
    for step in trace["steps"]:
        assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_sha256"])
