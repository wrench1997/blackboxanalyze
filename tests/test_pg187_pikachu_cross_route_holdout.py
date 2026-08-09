import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg187_pikachu_cross_route_holdout_report_v1.json"
TRACE = ROOT / "research" / "pg187_pikachu_cross_route_holdout_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg187_pikachu_cross_route_holdout_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg187_unseen_route_double_holdout_is_complete_and_fail_closed() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_unseen_route_double_holdout"
    assert report["source"]["routes_unseen_in_pg185"] == ["xss_01_unseen_get", "xss_04_unseen_get"]
    assert report["counts"]["episode_count"] == 24
    assert report["counts"]["sent_count"] == 120
    assert report["counts"]["candidate_sent_count"] == 48
    assert report["counts"]["typed_surface_effect_count"] == 12
    assert report["counts"]["typed_positive_count"] == 0
    assert report["holdout"]["route_holdout"] is True
    assert report["holdout"]["encoding_holdout"] is True
    assert report["holdout"]["model_input_route_present"] is False
    assert report["holdout"]["model_input_family_present"] is False
    assert report["holdout"]["false_vulnerability_positive_count"] == 0
    assert trace["evaluation_only"] is True
    assert protocol["fresh_restart_per_episode"] is True
    assert protocol["model_input_excludes_route"] is True
    assert protocol["model_input_excludes_family"] is True


def test_pg187_does_not_persist_runtime_values() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT, TRACE, PROTOCOL))
    for forbidden in ("pg186-cand", "pg186-ctrl", "<span", "data-sift-marker", "<script", "javascript:"):
        assert forbidden not in serialized.casefold()
    assert _load(REPORT)["selection"]["vulnerability_claim_allowed"] is False
    assert _load(REPORT)["selection"]["training_eligible"] is False
    assert _load(TRACE)["raw_probe_strings_stored"] is False
    assert _load(TRACE)["raw_response_bodies_stored"] is False
