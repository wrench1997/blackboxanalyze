import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_pg33_catalog_has_dual_channel_typed_replay_lineage():
    catalog = _load("pg_pk_33_get_post_typed_replay_catalog_v1.json")
    rows = catalog["samples"]
    assert catalog["runtime_replay"] is True
    assert catalog["methods"] == ["GET", "POST"]
    assert catalog["training_eligible"] is False
    assert catalog["typed_positive_count"] == 36
    assert catalog["negative_control_count"] == 48
    assert len(rows) == 84
    assert {row["dataset_role"] for row in rows} == {
        "train", "dev", "family_holdout", "ood_source", "negative_control"
    }
    assert {row["sampling_seed"] for row in rows} == {331, 337, 347}
    assert {row["method"] for row in rows} == {"GET", "POST"}
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows)
    assert all(row["reset"]["kind"] == "fresh_loopback_http_server" for row in rows)
    assert all(row["reset"]["transport"] == "httpx_loopback" for row in rows)
    assert {source["loopback_scope"]["port"] for source in catalog["sources"]} == {31933}
    assert all(row["evidence"]["evidence_hash"] for row in rows)
    assert all(row["evidence"]["safety"]["raw_body_stored"] is False for row in rows)
    assert all(row["evidence"]["safety"]["attack_string_stored"] is False for row in rows)
    positives = [row for row in rows if row["oracle_projection"]["positive"]]
    assert len(positives) == 36
    assert all(row["decision"]["evidence_status"] == "confirmed_positive" for row in positives)
    assert all(row["negative_control"]["same_source"] for row in positives)
    # Persisted output must contain projections/hashes only, never raw request or
    # response material.  The boolean safety attestations are allowed.
    forbidden = {"body", "raw_body", "request_body", "response_body", "credentials"}
    assert all(key.casefold() not in forbidden for key, _ in _walk(catalog))


def test_pg33_trace_dataset_has_get_post_episodes_and_no_training_promotion():
    trace = _load("pg_pk_33_trace_dataset_v1.json")
    assert trace["methods"] == ["GET", "POST"]
    assert trace["episode_count"] == 21
    assert len(trace["episodes"]) == 21
    assert sum(item["status"] == "accepted_evaluation" for item in trace["episodes"]) == 18
    assert sum(item["status"] == "trace_only" for item in trace["episodes"]) == 3
    assert trace["training_eligible"] is False
    assert all(item["training_candidate"] is False for item in trace["episodes"])
    assert all(item["memory_promotion_allowed"] is False for item in trace["episodes"])
    assert all(step["fresh_reset"]["fresh_target"] for step in trace["steps"])
    assert all(step["evidence_sha256"] for step in trace["steps"])
    assert all("probe" not in step["action_manifest"] for step in trace["steps"])
