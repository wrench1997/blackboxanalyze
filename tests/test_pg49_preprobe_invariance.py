import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg49_preprobe_selection_is_invariant_to_response_masking():
    report = _load("pg49_preprobe_invariance_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["holdout_implementation"] == "frost"
    assert report["pair_count"] == 384
    assert report["episode_count"] == 48
    assert report["selection_order_match_rate"] == 1.0
    assert report["selection_changed_count"] == 0
    assert report["max_score_delta"] == 0.0
    assert report["response_projection_consumed_by_selection"] is False
    assert report["safe_gate"]["status"] == "passed"
    assert report["safe_gate"]["claim_allowed"] is True
    assert report["safe_gate"]["training_allowed"] is False
    assert report["safe_gate"]["memory_promotion_allowed"] is False


def test_pg49_protocol_keeps_invariance_separate_from_capability_promotion():
    protocol = _load("pg49_preprobe_invariance_protocol_v1.json")
    assert protocol["acceptance_gate"]["selection_order_match_required"] is True
    assert protocol["acceptance_gate"]["zero_selection_changes_required"] is True
    assert protocol["run_result"]["selection_order_match_rate"] == 1.0
    assert protocol["run_result"]["selection_changed_count"] == 0
    assert protocol["run_result"]["max_score_delta"] == 0.0
    assert protocol["run_result"]["response_projection_consumed_by_selection"] is False
    assert protocol["run_result"]["training_allowed"] is False
    assert protocol["run_result"]["memory_promotion_allowed"] is False
    assert protocol["status"] == "run_completed_no_response_dependency"
    serialized = json.dumps(protocol, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "union select" not in serialized
