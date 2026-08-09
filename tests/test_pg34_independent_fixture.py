import json
import threading
from pathlib import Path

import httpx

from app.pg34_independent_fixture import make_independent_fixture_server


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_independent_fixture_exposes_safe_get_post_typed_surfaces():
    server = make_independent_fixture_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(base_url=base, timeout=2.0) as client:
            xss_negative = client.get("/pg34/surface/surface-01", params={"probe_class": "normal"})
            xss_positive = client.get("/pg34/surface/surface-01", params={"probe_class": "markup_candidate"})
            sql_positive = client.post("/pg34/surface/surface-02", json={"fragment_class": "operator_like"})
        assert xss_negative.status_code == xss_positive.status_code == sql_positive.status_code == 200
        assert xss_negative.json()["dom_change"] is False
        assert xss_positive.json()["dom_change"] is True
        assert sql_positive.json()["controlled_differential"] is True
        assert sql_positive.json()["interpreter_boundary"] is True
        assert sql_positive.json()["database_touched"] is False
        assert sql_positive.json()["external_network"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_pg34_independent_catalog_has_eight_families_and_fresh_http_replay():
    catalog = _load("pg34_independent_fixture_catalog_v1.json")
    rows = catalog["samples"]
    assert catalog["independent_target_implementation"] is True
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["typed_positive_count"] == 48
    assert catalog["negative_control_count"] == 60
    assert catalog["target_instance_count"] == 108
    assert len(rows) == 108
    assert {row["family"] for row in rows} >= {
        "xss", "injection", "authentication", "access_control", "logic",
        "url_redirect", "input_validation", "command_injection", "ordinary_response",
    }
    assert {source["loopback_scope"]["port"] for source in catalog["sources"]} == {31934}
    assert all(row["reset"]["kind"] == "fresh_independent_http_server" for row in rows)
    assert all(row["reset"]["transport"] == "httpx_loopback" for row in rows)
    assert all(row["evidence"]["safety"]["raw_body_stored"] is False for row in rows)
    assert all(row["evidence"]["safety"]["attack_string_stored"] is False for row in rows)
    assert all(row["evidence"]["evidence_hash"] for row in rows)


def test_pg34_independent_trace_has_positive_and_negative_episodes():
    trace = _load("pg34_independent_fixture_trace_v1.json")
    assert trace["independent_target_implementation"] is True
    assert trace["methods"] == ["GET", "POST"]
    assert trace["episode_count"] == 27
    assert sum(item["status"] == "accepted_evaluation" for item in trace["episodes"]) == 24
    assert sum(item["status"] == "trace_only" for item in trace["episodes"]) == 3
    assert trace["training_eligible"] is False
    assert all(step["fresh_reset"]["transport"] == "httpx_loopback" for step in trace["steps"])
