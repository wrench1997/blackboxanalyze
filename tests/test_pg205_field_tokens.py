import json
from pathlib import Path

from app.pg198_payload_grounding import generate_grounded_candidates
from app.pg205_field_token_controller import build_field_token_packet, validate_field_token_packet
from app.pg205_request_response_tokens import FIELD_TOKEN_DIM, field_tokens_for_runtime


ROOT = Path(__file__).resolve().parents[1]


def test_pg205_field_token_dimension_and_method_placement_are_explicit() -> None:
    projection = {
        "status_class": "3xx",
        "content_type_class": "html",
        "body_length_bucket": "256-1023",
        "marker": {"reflected": True},
        "redirect_chain": ["same_origin"],
        "location_origin_changed": False,
    }
    get_tokens = field_tokens_for_runtime(method="GET", field_names=["name", "submit"], projection=projection, typed_available=True, redirect_hops=1)
    post_tokens = field_tokens_for_runtime(method="POST", field_names=["content", "name", "submit"], projection=projection, typed_available=False, redirect_hops=1)
    assert len(get_tokens) == FIELD_TOKEN_DIM == 31
    assert len(post_tokens) == FIELD_TOKEN_DIM
    assert get_tokens != post_tokens


def test_pg205_binding_rejects_missing_and_mutated_request_response_tokens() -> None:
    route = {"surface": "unit", "path": "/vul/xss/xss_03.php", "method": "GET", "fields": ["message", "submit"]}
    candidate = next(
        row
        for row in generate_grounded_candidates(
            family="xss",
            target="http://127.0.0.1:3113",
            path=route["path"],
            method=route["method"],
            fields=route["fields"],
            marker="pg205-unit-token",
        )
        if row["payload"]["probe_kind"] == "inert_dom_markup"
    )
    projection = {"status_class": "2xx", "content_type_class": "html", "body_length_bucket": "short", "marker": {"reflected": False}, "redirect_chain": [], "projection_sha256": "a" * 64}
    packet = build_field_token_packet(candidate, route=route, response_projection=projection, typed_available=True, redirect_hops=0)
    assert validate_field_token_packet(packet, candidate=candidate, route=route, response_projection=projection, typed_available=True, redirect_hops=0)["valid"] is True
    assert validate_field_token_packet(None, candidate=candidate, route=route, response_projection=projection, typed_available=True, redirect_hops=0)["network_allowed"] is False
    mutated = dict(packet)
    mutated["field_tokens"] = list(packet["field_tokens"])
    mutated["field_tokens"][0] = 1.0 - mutated["field_tokens"][0]
    assert validate_field_token_packet(mutated, candidate=candidate, route=route, response_projection=projection, typed_available=True, redirect_hops=0)["network_allowed"] is False


def test_pg205_protocol_does_not_persist_raw_input() -> None:
    protocol = json.loads((ROOT / "research/pg204_token_binding_loop_protocol_v1.json").read_text(encoding="utf-8-sig"))
    assert protocol["raw_payload_and_response_excluded"] is True
