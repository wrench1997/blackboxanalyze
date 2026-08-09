import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg181_manifest_decoder_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg181_model_manifest_actually_drives_local_safe_replay() -> None:
    report = _load(REPORT_PATH)
    replay = report["replay"]
    assert report["status"] == "completed_safe_model_guided_local_replay"
    assert report["source"]["image_digest"].startswith("sha256:")
    assert report["training"]["run_count"] == 6
    assert replay["target_route"] == "/vul/urlredirect/urlredirect.php"
    assert replay["parameter_name"] == "url"
    assert replay["variant"] == "moe_large"
    assert replay["step_count"] == 5
    assert replay["sent_count"] == 5
    assert replay["baseline_sent_count"] == 1
    assert replay["control_sent_count"] == 2
    assert replay["candidate_sent_count"] == 2
    assert replay["controller_abstain_count"] == 0
    assert replay["typed_positive_count"] == 0
    assert replay["vulnerability_claim_allowed"] is False
    assert all(step["controller_decision"] in {"send_safe_baseline", "send_safe_canary"} for step in replay["steps"])
    assert all(step["model_action"] in {"baseline", "matched_control", "safe_candidate", "abstain"} for step in replay["steps"])
    assert all(step["decision"] == "abstain" for step in replay["steps"] if "decision" in step)


def test_pg181_replay_never_persists_raw_values_or_external_effects() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_PATH, TRACE_PATH, PROTOCOL_PATH))
    lowered = serialized.casefold()
    for forbidden in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "onerror", "pg181-canary-a1", "pg181-control-a1"):
        assert forbidden not in lowered
    report = _load(REPORT_PATH)
    for artifact in (report, _load(TRACE_PATH)):
        encoded = json.dumps(artifact, ensure_ascii=False)
        assert "raw_probe_strings_stored\": true" not in encoded
        assert "raw_response_bodies_stored\": true" not in encoded
    assert report["safety"]["loopback_only"] is True
    assert report["safety"]["external_network"] is False
    assert report["safety"]["script_execution"] is False
    assert report["safety"]["database_write"] is False
    assert report["safety"]["credential_access"] is False
    assert report["safety"]["memory_promotion_allowed"] is False


def test_pg181_protocol_requires_manifest_validation_before_send() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["replay"]["fresh_container_per_replay"] is True
    assert protocol["replay"]["only_observed_parameter"] == "url"
    assert protocol["gates"]["manifest_validator_before_send"] is True
    assert protocol["gates"]["unobserved_field_forbidden"] is True
    assert protocol["gates"]["safe_canary_only"] is True
    assert protocol["gates"]["typed_positive_required_for_vulnerability_label"] is True
    assert protocol["gates"]["training_promotion_allowed"] is False
    assert protocol["gates"]["memory_promotion_allowed"] is False
