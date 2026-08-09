import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg36_active_belief_reduces_queries_without_false_accepts():
    report = _load("pg36_active_belief_diagnostic_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["controller"]["typed_oracle_used_after_probe_for_stop_only"] is True
    assert report["controller"]["positive_authority"] is False
    assert report["fixed_policy"]["typed_recall"] == 1.0
    assert report["active_policy"]["typed_recall"] == 1.0
    assert report["active_policy"]["false_positive_rate"] == 0.0
    assert report["active_policy"]["mean_queries"] < report["fixed_policy"]["mean_queries"]
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg36_active_belief_report_does_not_retain_raw_probe_or_response_text():
    report_text = (ROOT / "research" / "pg36_active_belief_diagnostic_v1.json").read_text(encoding="utf-8").casefold()
    assert "<script" not in report_text
    assert "onerror" not in report_text
    assert "union select" not in report_text
