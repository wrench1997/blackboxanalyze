import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

from app.pg35_independent_fixture import PG35_VARIANTS, SURFACE_SPECS, make_pg35_server


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg35_fixture_has_safe_get_post_encoding_equivalence():
    server = make_pg35_server(0, "beta")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=3.0, follow_redirects=False) as client:
            path = f"/observe/b/surface-01?channel=surface-01&signal_class={quote('markup_candidate', safe='')}"
            identity = client.get(path)
            encoded = client.get("/observe/b/surface-01?channel=surface-01&signal_class=%6D%61%72%6B%75%70%5F%63%61%6E%64%69%64%61%74%65")
            posted = client.post("/observe/b/surface-02", json={"channel": "surface-02", "signal_class": "operator_like"})
            negative = client.post("/observe/b/surface-02", json={"channel": "surface-02", "signal_class": "normal"})
        assert identity.status_code == encoded.status_code == 200
        assert identity.json()["dom_change"] is True
        assert encoded.json()["dom_change"] is True
        assert posted.json()["ast_shape_diff"] is True
        assert negative.json()["candidate_signal"] is False
        for response in (identity, encoded, posted, negative):
            body = response.json()
            assert body["script_execution"] is False
            assert body["database_touched"] is False
            assert body["external_network"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_pg35_catalog_has_real_dual_channel_pairs_and_three_sources():
    catalog = _load("pg35_independent_fixture_catalog_v1.json")
    rows = catalog["samples"]
    assert catalog["independent_target_implementation"] is True
    assert catalog["training_eligible"] is False
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["encodings"] == ["identity", "url_percent"]
    assert len(rows) == 648
    assert catalog["typed_positive_count"] == 288
    assert catalog["negative_control_count"] == 360
    assert catalog["source_count"] == 3
    assert catalog["encoding_pair_count"] == 324
    assert len({row["source_sha256"] for row in rows}) == 3
    assert len({row["evidence"]["evidence_hash"] for row in rows}) == len(rows)
    assert {row["method"] for row in rows} == {"GET", "POST"}
    assert {row["encoding"] for row in rows} == {"identity", "url_percent"}
    assert {row["family"] for row in rows} == {spec["family"] for spec in SURFACE_SPECS.values()}
    assert all(row["reset"]["kind"] == "fresh_pg35_http_server" for row in rows)
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows)
    assert all("payload" not in row["payload_manifest"] for row in rows)
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg35_trace_requires_get_post_and_keeps_negative_pairs():
    trace = _load("pg35_independent_fixture_trace_v1.json")
    assert trace["independent_target_implementation"] is True
    assert trace["training_eligible"] is False
    assert trace["methods"] == ["GET", "POST"]
    assert trace["encodings"] == ["identity", "url_percent"]
    assert trace["episode_count"] == 81
    assert trace["accepted_evaluation_episode_count"] == 72
    assert len(trace["steps"]) == 648
    assert all(step["online_weight_update"] is False for step in trace["steps"])
    assert all(step["long_term_memory_write"] is False for step in trace["steps"])
    assert all("negative_control_pair_id" in step["oracle_projection"] for step in trace["steps"] if step["decision"] == "confirmed_positive")
    assert all(step["action_manifest"]["method"] in {"GET", "POST"} for step in trace["steps"])


def test_pg35_catalog_and_trace_manifests_are_hash_bound():
    catalog = _load("pg35_independent_fixture_catalog_v1.json")
    trace = _load("pg35_independent_fixture_trace_v1.json")
    evidence_hashes = [row["evidence"]["evidence_hash"] for row in catalog["samples"]]
    assert catalog["manifest_sha256"] == hashlib.sha256(
        json.dumps({"dataset_tests": catalog["dataset_tests"], "samples": evidence_hashes}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert trace["catalog_manifest_sha256"] == catalog["manifest_sha256"]
