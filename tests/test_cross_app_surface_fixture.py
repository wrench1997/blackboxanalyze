import asyncio
import threading
import time

import httpx
import pytest

from app.cross_app_surface_fixture import (
    SURFACE_FIXTURE_BASE_URL,
    SurfaceFixtureCollector,
    default_surface_fixture_specs,
    make_surface_fixture_server,
    surface_fixture_source_sha256,
    validate_surface_fixture_spec,
)


def _start_server():
    server = make_surface_fixture_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for _ in range(50):
        try:
            if httpx.get(f"{SURFACE_FIXTURE_BASE_URL}/plain", timeout=0.2).status_code == 200:
                return server, thread
        except Exception:
            time.sleep(0.01)
    server.shutdown()
    server.server_close()
    thread.join(timeout=1)
    raise RuntimeError("surface fixture did not start")


def test_surface_specs_cover_distinct_transports_and_pairs():
    specs = default_surface_fixture_specs()
    assert len(specs) == 9
    assert specs[0]["params"]["message"] != specs[1]["params"]["message"]
    assert specs[0]["surface_role"] == "reflected_attribute"
    assert specs[2]["surface_role"] == "reflected_text"
    assert specs[-1]["surface_role"] == "plain_control"


def test_surface_collector_keeps_only_attribute_as_positive_oracle():
    server, thread = _start_server()
    try:
        rows = asyncio.run(
            SurfaceFixtureCollector(target_instance_id="surface-test", source_hash=surface_fixture_source_sha256()).collect_many(
                default_surface_fixture_specs("fx-test")
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
    assert len(rows) == 9
    positives = [row for row in rows if row["rule_ir_result"]]
    assert len(positives) == 2
    assert {row["semantic"]["surface_role"] for row in positives} == {"reflected_attribute"}
    text = next(row for row in rows if row["semantic"]["surface_role"] == "reflected_text")
    json_row = next(row for row in rows if row["semantic"]["surface_role"] == "json_echo")
    header = next(row for row in rows if row["semantic"]["surface_role"] == "header_echo")
    assert text["oracle_projection"]["marker_in_html_text"] is True
    assert json_row["oracle_projection"]["marker_in_json_value"] is True
    assert header["oracle_projection"]["marker_in_header"] is True
    assert text["surface_shape"]["content_type_class"] == "html"
    assert json_row["surface_shape"]["content_type_class"] == "json"
    assert header["surface_shape"]["response_header_count"] > text["surface_shape"]["response_header_count"]
    assert all("raw_body" not in row["evidence"] for row in rows)


def test_surface_fixture_rejects_unsafe_paths():
    with pytest.raises(ValueError):
        validate_surface_fixture_spec({"path": "/api/challenges", "marker": "fx-test", "source_id": "s", "lab_id": "l", "surface_role": "x"})
