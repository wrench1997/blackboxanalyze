import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg182_juice_shop_manifest_replay_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg182_pikachu_model_replays_on_juice_shop_with_real_q_binding() -> None:
    report = _load(REPORT_PATH)
    replay = report["replay"]
    assert report["status"] == "completed_cross_app_evaluation_only"
    assert report["target"]["application"] == "Juice Shop"
    assert report["target"]["loopback"] is True
    assert report["target"]["external_network"] is False
    assert report["parameter_grounding"]["observed_parameter"] == "q"
    assert report["parameter_grounding"]["unobserved_method_forbidden"] is True
    assert replay["target_route"] == "/rest/products/search"
    assert replay["parameter_name"] == "q"
    assert replay["sent_count"] == 5
    assert replay["candidate_sent_count"] == 2
    assert replay["controller_abstain_count"] == 0
    assert replay["typed_positive_count"] == 0
    assert replay["vulnerability_claim_allowed"] is False
    assert all(step["model_action"] in {"baseline", "matched_control", "safe_candidate", "abstain"} for step in replay["steps"])
    assert all(step["controller_decision"] in {"send_safe_baseline", "send_safe_canary"} for step in replay["steps"])
    assert all(step["action_manifest"]["field_names"] in ([], ["q"]) for step in replay["steps"])


def test_pg182_cross_app_artifacts_are_raw_free_and_fail_closed() -> None:
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in (REPORT_PATH, TRACE_PATH, PROTOCOL_PATH))
    lowered = serialized.casefold()
    for forbidden in ("<script", "union select", "sleep(", "benchmark(", "javascript:", "onerror", "pg182-canary-a1", "pg182-control-a1"):
        assert forbidden not in lowered
    report = _load(REPORT_PATH)
    assert report["oracle"]["typed_execution_available"] is False
    assert report["oracle"]["family_specific_oracle_available"] is False
    assert report["oracle"]["positive_count"] == 0
    assert report["safety"]["raw_probe_strings_stored"] is False
    assert report["safety"]["raw_response_bodies_stored"] is False
    assert report["safety"]["memory_promotion_allowed"] is False


def test_pg182_protocol_requires_target_specific_oracle_before_positive() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["fresh_container_required"] is True
    assert protocol["parameter_authority"] == "allow-listed q from Juice Shop shadow collector"
    assert protocol["gates"]["manifest_validator_before_send"] is True
    assert protocol["gates"]["unobserved_parameter_forbidden"] is True
    assert protocol["gates"]["family_oracle_required_for_positive"] is True
    assert protocol["gates"]["unknown_oracle_action"] == "abstain"
    assert protocol["gates"]["training_promotion_allowed"] is False
    assert protocol["gates"]["memory_promotion_allowed"] is False
