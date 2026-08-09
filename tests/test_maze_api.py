from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_maze_catalog_and_dom_oracle_api_are_local_semantic_surfaces():
    labs = client.get("/api/maze/labs")
    assert labs.status_code == 200
    assert len(labs.json()) >= 12

    response = client.post(
        "/api/maze/oracle/dom",
        json={
            "value": '<span data-sift-marker="sift-marker">inert</span>',
            "sink": "template.innerHTML",
            "marker": "sift-marker",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dom_change"] is True
    assert body["script_execution"] is False
    assert body["network_access"] is False


def test_sql_oracle_api_rejects_unknown_class_and_never_executes_database():
    bad = client.post("/api/maze/oracle/sql", json={"fragment_class": "real_sql"})
    assert bad.status_code == 400
    response = client.post("/api/maze/oracle/sql", json={"fragment_class": "time_delay"})
    assert response.status_code == 200
    body = response.json()["evidence"]
    assert body["timeout_observed"] is True
    assert body["database_touched"] is False
    assert body["real_sleep_performed"] is False


def test_local_replay_get_adapters_are_read_only_and_semantic():
    dom = client.get(
        "/api/maze/replay/dom",
        params={
            "value": '<span data-sift-marker="sift-replay">x</span>',
            "marker": "sift-replay",
        },
    )
    assert dom.status_code == 200
    assert dom.json()["dom_change"] is True
    sql = client.get("/api/maze/replay/sql", params={"fragment_class": "time_delay"})
    assert sql.status_code == 200
    assert sql.json()["evidence"]["timeout_observed"] is True
    bad = client.get("/api/maze/replay/sql", params={"fragment_class": "real_sql"})
    assert bad.status_code == 400


def test_maze_run_manifest_api_verifies_engineering_smoke():
    latest = client.get("/api/maze/runs/latest")
    if latest.status_code == 404:
        assert latest.json()["detail"] == "no maze run manifest found"
        return
    assert latest.status_code == 200
    body = latest.json()
    assert body["schema_version"] == "sift-maze-run-v1"
    run = client.get(f"/api/maze/runs/{body['run_id']}")
    assert run.status_code == 200
    assert run.json()["ledger_verification"]["valid"] is True


def test_detection_payload_api_returns_manifest_without_execution():
    response = client.post(
        "/api/maze/detection-payload",
        json={
            "path": "/api/products",
            "marker": "sift-probe-api",
            "probe_kind": "sql_channel_class",
            "probe": "time_delay",
            "expected": {"status_class": "2xx"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "sift-detection-payload-v1"
    assert body["probe"] == "time_delay"
    assert body["safety"]["does_not_execute"] is True
    assert body["payload_sha256"]


def test_model_capability_api_blocks_without_dataset_holdout_evidence():
    response = client.post(
        "/api/research/model-capability/evaluate",
        json={
            "evidence": {
                "unit_tests_passed": True,
                "oracle_validated": True,
                "data_lineage_complete": True,
                "authorized_sources_attested": True,
                "raw_data_retained": False,
                "dataset_tests": [],
                "baseline_metrics": {
                    "typed_recall": .5,
                    "precision": .9,
                    "false_positive_rate": .1,
                    "abstain_precision": .9,
                    "ece": .1,
                    "median_queries": 3,
                },
                "candidate_metrics": {
                    "typed_recall": .5,
                    "precision": .9,
                    "false_positive_rate": .1,
                    "abstain_precision": .9,
                    "ece": .1,
                    "median_queries": 3,
                },
                "false_positive_count": 0,
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "blocked"
    assert body["training_allowed"] is False
    assert body["unit_tests_are_not_capability_evidence"] is True


def test_trace_aligned_api_echoes_a_safe_step_without_training_side_effects():
    import hashlib
    from app.trace_aligned_dataset import sha256_json

    def digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    action = {
        "method": "GET",
        "route_template_id": "fixture-route",
        "placement": "query",
        "encoding_chain": ["identity"],
        "probe_ref": "inert-marker",
        "probe_sha256": digest("probe"),
        "safety": {
            "no_external_network": True,
            "does_not_execute": True,
            "no_database_write": True,
            "no_credential_access": True,
        },
    }
    oracle = {"modality": "negative_control", "positive": False, "positive_authority": False}
    body = {
        "action_manifest": action,
        "baseline_projection": {"status_class": "2xx"},
        "response_projection": {"status_class": "2xx"},
        "oracle_projection": oracle,
        "belief_before": {"xss": .5, "none": .5},
        "belief_after": {"xss": .4, "none": .6},
        "decision": "abstain",
        "next_action": "stop",
    }
    response = client.post(
        "/api/research/trace/step",
        json={"step": {
            "episode_id": "episode-api",
            "step_id": "step-api",
            "sampling_seed": 1,
            "target_instance_id": "target-api",
            "hypothesis": "xss",
            **body,
            "fresh_reset": {"fresh_target": True, "completed": True, "evaluator_state_hidden": True},
            "evidence_sha256": digest("evidence"),
            "echo": {"sha256": sha256_json(body)},
        }},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["online_weight_update"] is False
    assert result["long_term_memory_write"] is False
    assert result["trace_sha256"]


def test_detection_payload_api_accepts_safe_post_form_manifest():
    response = client.post(
        "/api/maze/detection-payload",
        json={
            "method": "POST",
            "path": "/local/readonly",
            "marker": "sift-post-api",
            "form": {"probe": "sift-post-api"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "POST"
    assert body["form"]["probe"] == "sift-post-api"
    assert body["safety"]["does_not_execute"] is True


def test_local_replay_supports_paired_get_and_post_oracle_routes():
    get_dom = client.get(
        "/api/maze/replay/dom",
        params={"value": "<span data-sift-marker='paired'>paired</span>", "marker": "paired"},
    )
    post_dom = client.post(
        "/api/maze/replay/dom",
        json={"value": "<span data-sift-marker='paired'>paired</span>", "marker": "paired"},
    )
    assert get_dom.status_code == 200
    assert post_dom.status_code == 200
    assert get_dom.json()["evidence_hash"] == post_dom.json()["evidence_hash"]

    get_sql = client.get("/api/maze/replay/sql", params={"fragment_class": "operator_like"})
    post_sql = client.post("/api/maze/replay/sql", json={"fragment_class": "operator_like"})
    assert get_sql.status_code == 200
    assert post_sql.status_code == 200
    assert get_sql.json()["evidence"]["evidence_hash"] == post_sql.json()["evidence"]["evidence_hash"]

    get_logic = client.get("/api/maze/replay/logic", params={"probe_class": "boundary_candidate"})
    post_logic = client.post("/api/maze/replay/logic", json={"probe_class": "boundary_candidate"})
    assert get_logic.status_code == 200
    assert post_logic.status_code == 200
    assert get_logic.json()["typed_boundary_observed"] is True
    assert get_logic.json()["evidence_hash"] == post_logic.json()["evidence_hash"]
