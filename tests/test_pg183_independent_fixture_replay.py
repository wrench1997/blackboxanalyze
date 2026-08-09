import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg183_independent_fixture_replay_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg183_independent_fixture_replay_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg183_independent_fixture_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg183_frozen_manifest_decoder_replays_independent_surfaces() -> None:
    report = _load(REPORT_PATH)
    assert report["status"] == "completed_independent_implementation_evaluation"
    assert report["target"]["independent_from_pikachu"] is True
    assert report["target"]["loopback_only"] is True
    assert report["target"]["fresh_server_per_surface"] is True
    assert report["model"]["online_weight_update"] is False
    assert report["model"]["memory_promotion_allowed"] is False
    assert len(report["runs"]) == 3
    assert sum(item["sent_count"] for item in report["runs"]) == 15
    assert sum(item["candidate_sent_count"] for item in report["runs"]) == 6
    assert all(item["controller_abstain_count"] == 0 for item in report["runs"])
    assert all(item["typed_positive_count"] == 0 for item in report["runs"])
    assert report["oracle"]["positive_count"] == 0
    assert report["oracle"]["vulnerability_claim_allowed"] is False
    assert report["retention"]["old_checkpoint_hash_unchanged"] is True


def test_pg183_independent_artifacts_are_raw_free() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_PATH, TRACE_PATH, PROTOCOL_PATH))
    lowered = serialized.casefold()
    for forbidden in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "onerror", "pg183-canary-a1", "pg183-control-a1"):
        assert forbidden not in lowered
    report = _load(REPORT_PATH)
    assert report["safety"]["raw_probe_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert report["safety"]["script_execution"] is False
    assert report["safety"]["database_write"] is False
    assert report["safety"]["credential_access"] is False


def test_pg183_protocol_freezes_weights_and_requires_abstain() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["frozen_checkpoint_required"] is True
    assert protocol["independent_source_required"] is True
    assert protocol["fresh_server_per_surface"] is True
    assert protocol["parameter_authority"] == "independent fixture observed message field"
    assert protocol["gates"]["typed_positive_required_for_vulnerability_label"] is True
    assert protocol["gates"]["unknown_oracle_action"] == "abstain"
    assert protocol["gates"]["weight_update_during_evaluation"] is False
    assert protocol["gates"]["memory_promotion_during_evaluation"] is False
