from __future__ import annotations

import json

import pytest

from scripts.plan_pg368_second_implementation import ROUTES, build_pg368_second_implementation_plan, route_ref_sha256
from scripts.run_pg368_webgoat_binder_replay import bind_rule_ir, replay


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_pg368_dry_run_is_all_ask_and_never_contacts_target():
    report = replay(live=False)
    assert report["status"] == "blocked_model_ask"
    assert report["counts"] == {
        "episodes": 6,
        "roles": 24,
        "ask_rows": 24,
        "target_contacted": 0,
        "typed_method_shape_confirmed": 0,
        "confirmed_positive": 0,
        "unsafe_allow": 0,
    }
    assert all(row["abstain"] is True for row in report["rows"])
    assert all(row["model_projection"]["safe_to_send"] is False for row in report["rows"])
    assert all(row["binding"]["status"] == "ASK" for row in report["rows"])
    assert all(row["confirmed_positive"] is False for row in report["rows"])


def test_pg368_route_hash_is_role_bound_without_path_leak():
    plan = build_pg368_second_implementation_plan()
    report = replay(live=False)
    expected = {str(route["method"]): route_ref_sha256(route) for route in ROUTES}
    for row in report["rows"]:
        assert row["route_ref_sha256"] == expected[row["method"]]
        assert row["binding"]["route_ref_sha256"] == row["route_ref_sha256"]
    serialized = json.dumps(report, ensure_ascii=False).casefold()
    assert "/webgoat" not in serialized
    assert "http://" not in serialized and "https://" not in serialized
    assert plan["plan_sha256"] == report["plan_sha256"]


def test_pg368_binder_rejects_sendable_rule_ir_and_requires_slots():
    route = ROUTES[0]
    base = {
        "question": "ask_typed",
        "next_action": "ask_typed",
        "transport_ref": "request_method",
        "encoding_ref": "none",
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": False,
    }
    binding = bind_rule_ir(base, route=route, role="candidate")
    assert binding["status"] == "ASK"
    assert binding["safe_to_send"] is False
    with pytest.raises(ValueError):
        bind_rule_ir({**base, "safe_to_send": True}, route=route, role="candidate")
    with pytest.raises(ValueError):
        bind_rule_ir({key: value for key, value in base.items() if key != "encoding_ref"}, route=route, role="candidate")


def test_pg368_live_requires_explicit_local_gate(monkeypatch):
    monkeypatch.delenv("PG368_LOCAL_DOCKER_EVAL", raising=False)
    with pytest.raises(RuntimeError, match="PG368_LOCAL_DOCKER_EVAL"):
        replay(live=True, seeds=(36801,))


def test_pg368_report_has_no_raw_wire_keys():
    report = replay(live=False)
    forbidden = {"url", "uri", "body", "response_body", "payload", "raw_payload", "raw_value", "wire"}
    assert not any(key.casefold() in forbidden for key, _ in _walk(report))
