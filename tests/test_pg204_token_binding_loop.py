import json
from pathlib import Path

from app.pg198_payload_grounding import generate_grounded_candidates
from app.pg204_token_binding_controller import build_runtime_token_packet, validate_runtime_token_packet


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def test_pg204_ai_participates_in_bounded_get_post_loop() -> None:
    report = _load("research/pg204_token_binding_loop_report_v1.json")
    assert report["status"] == "completed_token_aware_real_get_post_loop"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["counts"]["fresh_container_count"] == 2
    assert report["counts"]["route_replay_count"] == 6
    assert report["counts"]["valid_token_binding_count"] == 6
    assert report["counts"]["candidate_send_count"] == 4
    assert report["counts"]["encoded_variant_send_count"] == 2
    assert report["counts"]["post_unknown_abstain_count"] == 2
    assert report["counts"]["false_positive_count"] == 0
    assert all(row["model_decision"]["encoding_binding_match"] for row in report["route_runs"])
    assert all(row["model_decision"]["failure_binding_match"] for row in report["route_runs"])


def test_pg204_token_faults_are_stopped_before_network() -> None:
    report = _load("research/pg204_token_binding_loop_report_v1.json")
    assert report["counts"]["token_fault_count"] == 24
    assert report["counts"]["network_allowed_on_fault_count"] == 0
    assert {row["case"] for row in report["fault_runs"]} == {
        "missing_encoding_token",
        "binding_hash_mismatch",
        "token_features_mismatch",
        "method_path_binding_mismatch",
    }
    assert all(not row["network_allowed"] for row in report["fault_runs"])


def test_pg204_packet_binds_candidate_and_route_and_rejects_mutation() -> None:
    route = {"surface": "unit", "path": "/vul/xss/xss_01.php", "method": "GET", "fields": ["message", "submit"]}
    candidate = next(
        row
        for row in generate_grounded_candidates(
            family="xss",
            target="http://127.0.0.1:3112",
            path=route["path"],
            method=route["method"],
            fields=route["fields"],
            marker="pg204-unit-token",
        )
        if row["payload"]["probe_kind"] == "inert_dom_markup"
    )
    projection = {"status_class": "2xx"}
    packet = build_runtime_token_packet(candidate, route=route, failure_projection=projection, typed_available=True)
    assert validate_runtime_token_packet(packet, candidate=candidate, route=route, failure_projection=projection)["valid"] is True

    wrong = dict(packet)
    wrong["token_features"] = list(packet["token_features"])
    wrong["token_features"][0] = 1.0 - wrong["token_features"][0]
    checked = validate_runtime_token_packet(wrong, candidate=candidate, route=route, failure_projection=projection)
    assert checked == {"valid": False, "reason": "token_features_mismatch", "network_allowed": False}


def test_pg204_persists_only_hashes_and_is_quarantined() -> None:
    report = _load("research/pg204_token_binding_loop_report_v1.json")
    protocol = _load("research/pg204_token_binding_loop_protocol_v1.json")
    rules = _load("research/improvement_rules.json")
    serialized = json.dumps(report, ensure_ascii=False)
    assert "<span" not in serialized
    assert "response_body" not in serialized
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["safety"]["loopback_only"] is True
    assert protocol["raw_payload_and_response_excluded"] is True
    assert rules["pg204_token_binding_loop"]["network_allowed_on_fault_count"] == 0
    assert rules["pg204_token_binding_loop"]["training_promotion_allowed"] is False
