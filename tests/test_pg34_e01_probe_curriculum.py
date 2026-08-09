import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg34_e01_active_stop_reduces_queries_without_false_accepts():
    report = _load("pg34_e01_probe_curriculum_report_v1.json")
    fixed = report["fixed_policy"]
    active = report["active_stop_policy"]
    assert report["controller"]["oracle_visible_before_probe"] is False
    assert report["controller"]["typed_oracle_used_after_probe_for_stop_only"] is True
    assert fixed["episode_count"] == 21
    assert active["episode_count"] == 21
    assert active["typed_recall"] == fixed["typed_recall"] == 1.0
    assert active["false_positive_count"] == fixed["false_positive_count"] == 0
    assert active["median_queries"] < fixed["median_queries"]
    assert active["mean_queries"] < fixed["mean_queries"]
    assert report["promotion"]["status"] == "diagnostic_only"
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg34_e01_manifest_keeps_oracle_and_raw_data_out_of_visible_input():
    manifest = _load("pg34_e01_probe_curriculum_manifest_v1.json")
    assert manifest["safety"]["new_network_requests"] is False
    assert manifest["safety"]["raw_probe_strings_stored"] is False
    assert manifest["safety"]["raw_response_bodies_stored"] is False
    assert manifest["safety"]["training_candidate"] is False
    assert "typed_oracle_projection" in manifest["hidden_until_after_probe"]
    assert "family" in manifest["hidden_until_after_probe"]
