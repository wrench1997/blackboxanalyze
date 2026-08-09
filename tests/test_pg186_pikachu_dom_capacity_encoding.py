import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg186_pikachu_dom_capacity_encoding_report_v1.json"
TRACE = ROOT / "research" / "pg186_pikachu_dom_capacity_encoding_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg186_pikachu_dom_capacity_encoding_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg186_has_capacity_seed_and_encoding_matrix() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_frozen_capacity_seed_encoding_replay"
    assert len(report["model_summaries"]) == 6
    assert report["counts"]["episode_count"] == 36
    assert report["counts"]["sent_count"] == 180
    assert report["counts"]["candidate_sent_count"] == 72
    assert report["counts"]["typed_surface_effect_count"] == 12
    assert report["counts"]["typed_positive_count"] == 0
    assert all(item["parameter_count"] > 0 for item in report["model_summaries"])
    assert {item["model"] for item in report["model_summaries"]} == {
        "small_seed18101", "small_seed18102", "medium_seed18101", "medium_seed18102", "moe_large_seed18101", "moe_large_seed18102"
    }
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert protocol["fresh_restart_per_episode"] is True
    assert protocol["typed_dom_effect_not_vulnerability"] is True


def test_pg186_artifacts_are_raw_free_and_fail_closed() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT, TRACE, PROTOCOL))
    for forbidden in ("pg186-cand", "pg186-ctrl", "<span", "data-sift-marker", "<script", "javascript:"):
        assert forbidden not in serialized.casefold()
    assert report["selection"]["training_eligible"] is False
    assert report["selection"]["memory_promotion_allowed"] is False
    assert report["selection"]["vulnerability_claim_allowed"] is False
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert protocol["gates"]["training_allowed"] is False
    assert protocol["gates"]["memory_promotion_allowed"] is False
