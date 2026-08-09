import asyncio
from pathlib import Path

import pytest

from app.main import app
from app.payload_catalog import write_catalog, load_catalog, flatten_catalog
from app.replay_collector import LocalReplayCollector, validate_replay_spec


ROOT = Path(__file__).resolve().parents[1]


def _dom_spec():
    probe = '<span data-sift-marker="pg03-probe-a">x</span>'
    return {
        "source_id": "pg03-src-a",
        "lab_id": "maze-dom-replay",
        "family": "xss",
        "surface": "dom_sink",
        "expected_oracle": "controlled_detached_dom_v1",
        "expected_signal": "browser_sink_observed+dom_change",
        "path": "/api/maze/replay/dom",
        "probe_kind": "inert_dom_markup",
        "marker": "pg03-probe-a",
        "probe": probe,
        "params": {"value": probe, "marker": "pg03-probe-a"},
    }


def test_local_replay_collector_captures_bounded_response_and_rule_ir(tmp_path: Path):
    record = asyncio.run(LocalReplayCollector(app).collect(_dom_spec()))
    assert record["safety"]["local_only"] is True
    assert record["safety"]["raw_body_stored"] is False
    assert record["response_projection"]["status_code"] == 200
    assert record["oracle_projection"]["dom_change"] is True
    assert record["rule_ir_result"] is True
    assert "body" not in record["response_projection"]
    assert record["evidence"]["evidence_hash"]

    catalog = {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pg03-test-catalog",
        "sources": [{
            "provenance": {
                "source_id": "pg03-src-a",
                "source_type": "in_repo_synthetic",
                "origin": "research/payload_source_catalog_v1.json",
                "license": "in_repo_synthetic",
                "authorization": "workspace_local_only",
                "scope": ["http://127.0.0.1:3100"],
                "captured_at": "2026-08-02",
                "authorized_for": ["training", "local_replay", "holdout_evaluation"],
                "external_network": False,
                "evaluator_state_visible": False,
            },
            "samples": [record],
        }],
    }
    path = tmp_path / "pg03-catalog.json"
    written = write_catalog(path, catalog)
    assert len(flatten_catalog(load_catalog(path))) == 1
    assert written["catalog_sha256"]


def test_replay_collector_rejects_external_or_mutating_scope():
    bad = dict(_dom_spec(), target="https://example.com")
    with pytest.raises(ValueError, match="127.0.0.1"):
        validate_replay_spec(bad)
    bad = dict(_dom_spec(), method="POST")
    with pytest.raises(ValueError, match="only GET"):
        validate_replay_spec(bad)
    bad = dict(_dom_spec(), path="/api/challenges/secret")
    with pytest.raises(ValueError, match="allow-listed"):
        validate_replay_spec(bad)
