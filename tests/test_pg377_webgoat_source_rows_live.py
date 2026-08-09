from __future__ import annotations

import json

import pytest

from scripts.run_pg377_webgoat_source_rows_live import (
    AXES,
    FIELD_COUNT,
    ROUTES,
    SCHEMA_VERSION,
    _fake_capture,
    _incomplete_capture,
    _capture_role,
    _normalize_reset,
    build_pg377_plan,
    collect_pg377_webgoat_source_rows,
    write_artifacts,
)


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_pg377_plan_is_static_seven_axis_get_post_and_107_field_contract() -> None:
    plan = build_pg377_plan()
    assert plan["schema_version"] == SCHEMA_VERSION
    assert plan["status"] == "planning_only"
    assert plan["expected_role_replay_count"] == 3 * 2 * 4
    assert plan["expected_source_row_count"] == 3 * 2 * 4
    assert plan["capture_contract"]["seven_axes"] == list(AXES)
    assert plan["capture_contract"]["field_capture_manifest_count"] == FIELD_COUNT == 107
    assert plan["execution"]["docker_started"] is False
    assert plan["execution"]["network_contacted"] is False
    assert all(value is False for value in plan["promotion"].values())
    text = json.dumps(plan, ensure_ascii=False).casefold()
    assert "/webgoat" not in text and "http://" not in text and "https://" not in text


def test_pg377_planning_never_calls_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_runtime(**_kwargs):
        raise AssertionError("planning lane contacted runtime")

    monkeypatch.setattr("scripts.run_pg377_webgoat_source_rows_live._capture_role", fail_runtime)
    result = collect_pg377_webgoat_source_rows(seeds=(37701,), live=False)
    assert result["report"]["status"] == "blocked_live_gate"
    assert result["rows"] == [] and result["sidecars"] == []


def test_pg377_live_requires_explicit_gate_before_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PG377_LOCAL_DOCKER_EVAL", raising=False)
    with pytest.raises(RuntimeError, match="PG377_LOCAL_DOCKER_EVAL"):
        collect_pg377_webgoat_source_rows(seeds=(37701,), live=True, capture_role=_fake_capture)


def test_pg377_reset_normalizer_drops_relay_diagnostics() -> None:
    reset = _normalize_reset(
        {
            "fresh_reset": True,
            "reset_id": "r",
            "target_instance_digest": "a" * 64,
            "network_mode": "none",
            "external_network": False,
            "loopback_only": True,
            "state_clean": True,
            "volume_mount_count": 0,
            "container_restart_used": False,
            "attestation": {"attested": True},
            "readiness_status_class": "2xx",
        }
    )
    assert set(reset) == {
        "fresh_reset",
        "reset_id",
        "target_instance_digest",
        "network_mode",
        "external_network",
        "loopback_only",
        "state_clean",
        "volume_mount_count",
        "container_restart_used",
    }
    assert "attestation" not in reset and "readiness_status_class" not in reset


def test_pg377_capture_role_normalizes_extra_relay_reset_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_pg377_webgoat_source_rows_live as runner

    class FakeTarget:
        def __init__(self, **_kwargs):
            self.stopped = False

        def start(self):
            return {
                "fresh_reset": True,
                "reset_id": "r",
                "target_instance_digest": "a" * 64,
                "network_mode": "none",
                "external_network": False,
                "loopback_only": True,
                "state_clean": True,
                "attestation": {"attested": True},
                "readiness_status_class": "2xx",
            }

        def request(self, *, method, form_body=b"", path=None):
            _ = form_body, path
            return {
                "method": method,
                "status": 200,
                "status_class": "2xx",
                "content_type_class": "text/html",
                "location_class": "none",
                "body": b"<!doctype html><html><head></head><body>fixture</body></html>",
            }

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(runner, "DisposableWebGoat", FakeTarget)
    monkeypatch.setattr(runner, "build_container_command", lambda **_kwargs: [])
    route = ROUTES[0]
    captured = _capture_role(seed=37701, role="candidate", route=route, route_ref=runner._route_ref(route))
    assert set(captured["reset"]) == {
        "fresh_reset",
        "reset_id",
        "target_instance_digest",
        "network_mode",
        "external_network",
        "loopback_only",
        "state_clean",
        "volume_mount_count",
        "container_restart_used",
    }


def test_pg377_missing_runtime_observation_is_incomplete_and_ask_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG377_LOCAL_DOCKER_EVAL", "1")

    def missing(*, seed, role, route, route_ref):
        return _incomplete_capture(seed=seed, role=role, route=route, route_ref=route_ref)

    result = collect_pg377_webgoat_source_rows(seeds=(37701,), live=True, capture_role=missing)
    assert result["report"]["status"] == "completed_incomplete_source_rows"
    assert result["report"]["counts"]["capture_failure_count"] == 8
    assert result["report"]["counts"]["valid_source_row_count"] == 0
    assert all(row["target_projection"]["safe_to_send"] is False for row in result["rows"])
    assert all(row["training_eligible"] is False for row in result["rows"])


def test_pg377_fake_runtime_materializes_all_roles_without_raw_or_promotion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG377_LOCAL_DOCKER_EVAL", "1")
    result = collect_pg377_webgoat_source_rows(seeds=(37701,), live=True, capture_role=_fake_capture)
    report = result["report"]
    assert report["status"] == "completed_source_row_candidate_only"
    assert report["counts"] == {
        "seed_count": 1,
        "route_count": 2,
        "role_replay_count": 8,
        "source_row_count": 8,
        "valid_source_row_count": 8,
        "strict_incomplete_count": 0,
        "capture_failure_count": 0,
        "training_eligible_count": 0,
        "typed_role_count": 6,
        "negative_violation_count": 0,
        "failure_observed_count": 2,
        "failure_action_changed_count": 2,
        "belief_observed_count": 8,
        "replay_sidecar_count": 2,
        "seven_axis_complete_count": 8,
    }
    assert report["hard_gate"]["seven_axes_present"] is True
    assert report["hard_gate"]["field_manifest_107"] is True
    assert report["hard_gate"]["candidate_reference_negative_replay"] is True
    assert report["hard_gate"]["negative_zero_violation"] is True
    assert report["hard_gate"]["failure_repair_observed"] is True
    assert report["hard_gate"]["belief_replay_observed"] is True
    assert report["promotion"] == {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    assert len(result["rows"]) == 8
    assert len(result["sidecars"]) == 2
    assert all(row["adapter_validation"]["valid"] for row in result["rows"])
    assert all(row["field_capture_manifest_count"] == 107 for row in result["rows"])
    assert all(row["target_projection"]["safe_to_send"] is False for row in result["rows"])
    assert {entry["roles"]["negative"]["typed_effect_confirmed"] for entry in result["sidecars"]} == {False}
    assert {entry["roles"]["candidate"]["typed_effect_confirmed"] for entry in result["sidecars"]} == {True}
    forbidden = {"url", "uri", "path", "payload", "raw_payload", "response_body", "wire", "oracle_answer", "evaluator_answer"}
    assert not any(key.casefold() in forbidden for key, _ in _walk(result))
    text = json.dumps(result, ensure_ascii=False).casefold()
    assert "/webgoat" not in text and "http://" not in text and "https://" not in text


def test_pg377_artifact_dataset_projects_complete_nested_pg331_rows(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG377_LOCAL_DOCKER_EVAL", "1")
    result = collect_pg377_webgoat_source_rows(seeds=(37701,), live=True, capture_role=_fake_capture)
    paths = write_artifacts(
        result,
        output=tmp_path / "report.json",
        dataset_output=tmp_path / "rows.json",
        sidecar_output=tmp_path / "sidecars.json",
    )
    dataset = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert set(paths) == {"report", "dataset", "sidecars"}
    assert dataset["row_projection"].startswith("pg331_source_row")
    assert len(dataset["records"]) == 8
    assert all(record["schema_version"] == "pg331-whole-web-source-row-v1" for record in dataset["records"])
    assert all(record["raw_payload_stored"] is False and record["raw_response_body_stored"] is False for record in dataset["records"])


def test_pg377_route_contract_is_get_post_and_route_path_is_not_serialized() -> None:
    assert {str(route["expected_method"]) for route in ROUTES} == {"GET", "POST"}
    plan = build_pg377_plan(seeds=(37701,))
    assert {route["method"] for route in plan["routes"]} == {"GET", "POST"}
    assert all(route["path_literal_stored"] is False for route in plan["routes"])
