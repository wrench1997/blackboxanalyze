import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg185_pikachu_dom_replay_report_v1.json"
TRACE = ROOT / "research" / "pg185_pikachu_dom_replay_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg185_pikachu_dom_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg185_model_guided_read_only_replay_has_typed_dom_surface_projection() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_model_guided_read_only_dom_surface_replay"
    assert report["source"]["fresh_container"] is True
    assert report["source"]["observed_route_count"] == 2
    assert report["counts"]["sent_count"] == 10
    assert report["counts"]["candidate_sent_count"] == 4
    assert report["counts"]["typed_surface_effect_count"] == 2
    assert report["counts"]["typed_positive_count"] == 0
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["training_eligible"] is False
    assert trace["evaluation_only"] is True
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert protocol["manifest_validator_before_send"] is True
    assert protocol["probe_contract"]["script_execution"] is False
    assert protocol["oracle_contract"]["typed_dom_effect_not_vulnerability"] is True


def test_pg185_artifacts_do_not_persist_runtime_dom_marker_or_markup() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT, TRACE, PROTOCOL))
    for forbidden in ("pg185-canary", "pg185-control", "<span", "data-sift-marker", "<script", "javascript:"):
        assert forbidden not in serialized.casefold()
    for step in [step for run in _load(REPORT)["runs"] for step in run["steps"] if "action_manifest" in step]:
        assert "payload_sha256" in step["action_manifest"]
        assert "manifest_sha256" in step["action_manifest"]
        assert "marker" not in step["action_manifest"]
        assert step["oracle_projection"]["safety"]["script_execution"] is False
