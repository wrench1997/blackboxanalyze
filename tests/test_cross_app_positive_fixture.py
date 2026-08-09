import asyncio
import threading
import time

import httpx
import pytest

from app.cross_app_positive_fixture import (
    FIXTURE_BASE_URL,
    PositiveFixtureCollector,
    default_fixture_specs,
    fixture_source_sha256,
    make_server,
    validate_fixture_spec,
)


def _start_server():
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            if httpx.get(f"{FIXTURE_BASE_URL}/plain", timeout=0.2).status_code == 200:
                return server, thread
        except Exception:
            time.sleep(0.01)
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)
    raise RuntimeError("fixture did not start")


def test_fixture_specs_keep_plain_and_percent_transports_distinct():
    specs = default_fixture_specs()
    assert specs[0]["params"]["message"] != specs[1]["params"]["message"]
    assert specs[0]["probe"] != specs[1]["probe"]
    assert validate_fixture_spec(specs[1])["encoding"] == "url_percent"


def test_fixture_collector_emits_bounded_positive_and_negative_oracles():
    server, thread = _start_server()
    try:
        rows = asyncio.run(
            PositiveFixtureCollector(target_instance_id="fixture-test", source_hash=fixture_source_sha256()).collect_many(
                default_fixture_specs("fx-test")
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert len(rows) == 3
    assert rows[0]["rule_ir_result"] is True
    assert rows[1]["rule_ir_result"] is True
    assert rows[2]["rule_ir_result"] is False
    assert rows[0]["evidence"]["reset"]["fresh_target"] is True
    assert rows[0]["evidence"]["script_execution"] is False
    assert "raw_body" not in rows[0]["evidence"]
    assert rows[0]["evidence"]["evidence_hash"] != rows[1]["evidence"]["evidence_hash"]


def test_fixture_rejects_non_loopback_target_and_unsafe_path():
    with pytest.raises(ValueError):
        validate_fixture_spec({"target": "https://example.com", "path": "/plain", "marker": "fx-test"})
    with pytest.raises(ValueError):
        validate_fixture_spec({"path": "/api/challenges", "marker": "fx-test"})
