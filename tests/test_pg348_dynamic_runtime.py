from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry, start_server


ROOT = Path(__file__).resolve().parents[1]


def _app() -> tuple[DynamicFixtureApplication, str]:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    app = DynamicFixtureApplication(registry)
    return app, str(registry["records"][0]["challenge_id"])


def test_dynamic_runtime_get_post_and_fresh_reset_are_ephemeral() -> None:
    app, challenge_id = _app()
    first = app.reset(challenge_id)
    get_result = app.handle("GET", challenge_id, {"term": ["opaque"]})
    post_result = app.handle("POST", challenge_id, {}, {"term": ["opaque"]})
    assert get_result["input_presence"] == "present"
    assert post_result["state_delta"] == "event_count_changed"
    assert "opaque" not in post_result["body"]
    second = app.reset(challenge_id)
    assert first["reset_id"] != second["reset_id"]
    assert app.handle("GET", challenge_id)["state_event_count"] == 0
    assert post_result["persistent_storage"] is False


def test_dynamic_runtime_http_is_loopback_only_and_supports_get_post() -> None:
    app, challenge_id = _app()
    app.reset(challenge_id)
    server, thread = start_server(app, port=0)
    try:
        origin = f"http://127.0.0.1:{server.server_port}/pg348/dynamic/{challenge_id}"
        get_body = urlopen(origin + "?term=opaque", timeout=2).read().decode("utf-8")
        request = Request(origin, data=b"term=opaque", method="POST", headers={"Content-Type": "application/x-www-form-urlencoded"})
        post_body = urlopen(request, timeout=2).read().decode("utf-8")
        assert 'data-runtime="dynamic"' in get_body
        assert 'data-runtime="dynamic"' in post_body
        assert "opaque" not in get_body and "opaque" not in post_body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dynamic_runtime_rejects_non_loopback_bind() -> None:
    app, _ = _app()
    with pytest.raises(ValueError, match="loopback-only"):
        start_server(app, host="0.0.0.0")


def test_dynamic_runtime_covers_all_520_registry_records() -> None:
    registry = load_registry(ROOT / "fixtures" / "pg348" / "registry_v1.json")
    app = DynamicFixtureApplication(registry)
    for row in registry["records"]:
        challenge_id = str(row["challenge_id"])
        attestation = app.reset(challenge_id)
        get_result = app.handle("GET", challenge_id, {"abstract": ["opaque"]})
        post_result = app.handle("POST", challenge_id, {}, {"abstract": ["opaque"]})
        assert attestation["fresh_reset"] is True
        assert get_result["external_network"] is False
        assert post_result["persistent_storage"] is False
        assert "opaque" not in get_result["body"] and "opaque" not in post_result["body"]


def test_dynamic_runtime_typed_variant_and_negative_are_distinguishable() -> None:
    app, challenge_id = _app()
    app.reset(challenge_id)
    candidate = app.handle("GET", challenge_id, {"probe": ["opaque"]}, probe_variant="candidate_surface")
    assert candidate["typed_effect_confirmed"] is True
    assert candidate["effect_class"] == "logic_transition"
    assert candidate["state_delta"] == "disposable_evaluator_state"

    app.reset(challenge_id)
    reference = app.handle("GET", challenge_id, {"probe": ["opaque"]}, probe_variant="reference_surface")
    assert reference["typed_effect_confirmed"] is True
    assert reference["effect_class"] == candidate["effect_class"]

    app.reset(challenge_id)
    negative = app.handle("GET", challenge_id, {"probe": ["opaque"]}, probe_variant="negative_control")
    assert negative["typed_effect_confirmed"] is False
    assert negative["state_delta"] == "none"
    assert "opaque" not in negative["body"]


def test_dynamic_runtime_failure_then_repair_changes_abstract_action() -> None:
    app, challenge_id = _app()
    app.reset(challenge_id)
    failed = app.handle("POST", challenge_id, {"probe": ["opaque"]}, probe_variant="unsupported_variant")
    assert failed["status"] == 400
    assert failed["failure_class"] == "blocked_variant"
    assert failed["repair_delta_axis"] == "probe_variant"

    app.reset(challenge_id)
    repaired = app.handle("POST", challenge_id, {"probe": ["opaque"]}, probe_variant="candidate_surface")
    assert repaired["status"] < 400
    assert repaired["typed_effect_confirmed"] is True
    assert repaired["failure_class"] == "none"
