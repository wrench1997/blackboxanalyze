from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import pytest

from scripts.plan_pg379_source_collection import SLOTS, build_pg379_source_collection_plan
from scripts.run_pg379_dynamic_source_rows_live import (
    ALL_ROLES,
    OPERATOR_FLAG,
    SCHEMA_VERSION,
    _DockerWholePageRuntime,
    build_pg379_docker_runtime_factory,
    collect_pg379_dynamic_source_rows_live,
    _typed_from_bounded_projection,
    write_artifacts,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _attestation(lane: str, image: str) -> dict[str, object]:
    return {
        "implementation_id": f"pg379_dynamic_real_{'train_impl_a' if lane == 'train' else 'holdout_impl_b'}",
        "lane": lane,
        "bound": True,
        "attestation_status": "passed",
        "image_digest": f"sha256:{image}",
        "image_built": True,
        "image_attested": True,
        "runtime_module_sha256": _digest(f"{lane}-runtime"),
        "process_boundary_sha256": _digest(f"{lane}-process"),
        "source_digest": _digest(f"{lane}-source"),
        "authorization_id": "pg379-test-local-loopback",
        "network_mode": "none",
        "external_network": False,
        "loopback_only": True,
        "bind_or_volume_mounts_allowed": False,
        "published_ports": False,
        "fresh_reset_contract": True,
        "independent_source_review": True,
        "side_effects_enabled": False,
    }


class _FakeRuntime:
    starts = 0
    stops = 0
    requests = 0

    def __init__(self, *, route: Mapping[str, object], role: str, seed: int, lane: str, **_: object) -> None:
        self.route = route
        self.role = role
        self.seed = seed
        self.lane = lane

    def start(self) -> dict[str, object]:
        type(self).starts += 1
        return {
            "fresh_reset": True,
            "reset_id": f"reset-{self.lane}-{self.seed}-{self.role}",
            "target_instance_digest": _digest(f"target-{self.lane}-{self.seed}-{self.role}"),
            "network_mode": "none",
            "external_network": False,
            "loopback_only": True,
            "state_clean": True,
            "volume_mount_count": 0,
            "container_restart_used": False,
        }

    def request(self, *, method: str, phase: str, **_: object) -> dict[str, object]:
        type(self).requests += 1
        expected = str(self.route["method"]).upper()
        action_method = str(method).upper()
        typed = self.role != "negative" and action_method == expected
        if action_method == "GET":
            return {
                "method": "GET",
                "status": 200,
                "status_class": "2xx",
                "content_type_class": "text/html",
                "location_class": "none",
                "typed_effect": typed,
                "body": "<!doctype html><html><head><title>fixture</title></head><body><form><input name='q'></form></body></html>",
            }
        return {
            "method": "POST",
            "status": 302,
            "status_class": "3xx",
            "content_type_class": "text/html",
            "location_class": "loopback",
            "typed_effect": typed,
            "body": "",
        }

    def stop(self) -> None:
        type(self).stops += 1


def _factory(**kwargs: object) -> _FakeRuntime:
    return _FakeRuntime(**kwargs)  # type: ignore[arg-type]


def _attestations() -> dict[str, object]:
    return {
        "train": _attestation("train", "1" * 64),
        "holdout": _attestation("holdout", "2" * 64),
    }


def _walk(value: object):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_pg379_planning_is_side_effect_free() -> None:
    result = collect_pg379_dynamic_source_rows_live(seeds=(37901,), live=False)
    report = result["report"]
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "planning_only_live_blocked"
    assert result["rows"] == [] and result["sidecars"] == []
    assert report["execution"]["docker_started"] is False
    assert report["objective"]["training_rows_created"] is False


def test_pg379_missing_attestation_fails_closed_before_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPERATOR_FLAG, "1")
    calls: list[object] = []

    def factory(**kwargs: object):
        calls.append(kwargs)
        raise AssertionError("runtime must not be constructed before attestation")

    result = collect_pg379_dynamic_source_rows_live(
        seeds=(37901,),
        live=True,
        attestations=None,
        image_digest="sha256:" + "1" * 64,
        runtime_factory=factory,
    )
    assert result["report"]["status"] == "blocked_preflight"
    assert result["report"]["execution"]["docker_started"] is False
    assert calls == []
    assert any("attestation_missing" in reason for reason in result["report"]["live_gate"]["blocked_reasons"])


def test_pg379_unbuilt_image_fails_closed_without_target_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPERATOR_FLAG, "1")
    attestations = _attestations()
    attestations["train"]["image_built"] = False  # type: ignore[index]
    calls: list[object] = []

    def factory(**kwargs: object):
        calls.append(kwargs)
        raise AssertionError("unbuilt image must not start")

    result = collect_pg379_dynamic_source_rows_live(
        seeds=(37901,),
        live=True,
        attestations=attestations,
        runtime_factory=factory,
    )
    assert result["report"]["status"] == "blocked_preflight"
    assert calls == []
    assert any("image_build_unattested" in reason for reason in result["report"]["live_gate"]["blocked_reasons"])


def test_pg379_fake_runtime_fresh_matrix_adapter_and_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPERATOR_FLAG, "1")
    _FakeRuntime.starts = _FakeRuntime.stops = _FakeRuntime.requests = 0
    result = collect_pg379_dynamic_source_rows_live(
        seeds=(37901,),
        live=True,
        attestations=_attestations(),
        runtime_factory=_factory,
    )
    report = result["report"]
    # Two implementations × one seed × twelve abstract routes × four roles.
    assert report["status"] == "completed_source_row_candidate_only"
    assert report["counts"]["role_episode_observed"] == 96
    assert report["counts"]["source_row_count"] == 72
    assert report["counts"]["target_sidecar_count"] == 24
    assert _FakeRuntime.starts == _FakeRuntime.stops == 96
    assert report["execution"]["runtime_episodes_started"] == 96
    assert report["counts"]["runtime_started_count"] == 96
    assert report["execution"]["docker_started"] is False
    assert report["execution"]["target_contacted"] is False
    assert report["counts"]["training_eligible_count"] == 0
    assert report["hard_gate"]["candidate_reference_negative_replay"] is True
    assert report["hard_gate"]["fresh_reset_per_role"] is True
    assert report["hard_gate"]["target_slots_13"] is True
    assert report["hard_gate"]["context_firewall"] is True
    assert report["hard_gate"]["negative_zero_violation"] is True
    assert all(row["training_eligible"] is False for row in result["rows"])
    assert all(set(sidecar["target_slots"]) == set(SLOTS) for sidecar in result["sidecars"])
    # Evaluator targets are sidecars, never context fields or training rows.
    assert all("target_slots" not in row for row in result["rows"])
    forbidden = {"payload", "raw_payload", "response_body", "raw_response", "wire", "oracle_answer", "evaluator_answer"}
    assert not any(key.casefold() in forbidden for key, _ in _walk(result))
    text = json.dumps(result, ensure_ascii=False).casefold()
    assert "http://" not in text and "https://" not in text
    assert all(value is False for value in report["promotion"].values())


def test_pg379_writer_never_emits_training_dataset(tmp_path) -> None:
    result = collect_pg379_dynamic_source_rows_live(seeds=(37901,), live=False)
    paths = write_artifacts(
        result,
        output=tmp_path / "report.json",
        sidecar_output=tmp_path / "sidecars.json",
    )
    assert set(paths) == {"report", "sidecars"}
    assert not (tmp_path / "rows.json").exists()
    sidecars = json.loads((tmp_path / "sidecars.json").read_text(encoding="utf-8"))
    assert sidecars["training_rows_written"] is False


def test_pg379_docker_factory_requires_digest_bound_route_manifest() -> None:
    with pytest.raises(ValueError, match="image_ref and route map"):
        build_pg379_docker_runtime_factory({"train": {}, "holdout": {}})
    factory = build_pg379_docker_runtime_factory(
        {
            "train": {
                "image_ref": "fixture/train@sha256:" + "1" * 64,
                "runtime_language": "python",
                "port": 8080,
                "routes": [{"route_class": "get_query_html_text", "path": "/internal", "parameter": "q"}],
            },
            "holdout": {
                "image_ref": "fixture/holdout@sha256:" + "2" * 64,
                "runtime_language": "node",
                "port": 8799,
                "routes": [{"route_class": "get_query_html_text", "path": "/internal", "parameter": "q"}],
            },
        }
    )
    assert getattr(factory, "is_docker_runtime") is True
    assert factory.preflight(lane="train", image_digest="sha256:" + "1" * 64) is True
    assert factory.preflight(lane="train", image_digest="sha256:" + "2" * 64) is False
    with pytest.raises(ValueError, match="immutable @sha256"):
        build_pg379_docker_runtime_factory(
            {
                "train": {"image_ref": "fixture/train:latest", "runtime_language": "python", "port": 8080, "routes": [{"route_class": "x", "path": "/x"}]},
                "holdout": {"image_ref": "fixture/holdout:latest", "runtime_language": "node", "port": 8799, "routes": [{"route_class": "x", "path": "/x"}]},
            }
        )


def test_pg379_bounded_html_shape_is_typed_only_for_safe_canary() -> None:
    route = {
        "method": "GET",
        "response_shape": "html_text",
    }
    action = {"status": 200, "content_type_class": "text/html"}
    safe_html = b'<main data-pg379-b-shape="html_text" data-input-class="safe_canary"></main>'
    ordinary_html = b'<main data-pg379-b-shape="html_text" data-input-class="ordinary"></main>'
    assert _typed_from_bounded_projection(action=action, body=safe_html, route=route, role="candidate") is True
    assert _typed_from_bounded_projection(action=action, body=ordinary_html, route=route, role="candidate") is False
    assert _typed_from_bounded_projection(action=action, body=safe_html, route=route, role="negative") is False


def test_pg379_bounded_json_shape_uses_abstract_projection() -> None:
    route = {"method": "POST", "response_shape": "state_delta"}
    action = {"status": 200, "content_type_class": "application/json"}
    body = b'{"response_shape":"state_delta","state_delta":true}'
    assert _typed_from_bounded_projection(action=action, body=body, route=route, role="reference") is True


def test_pg379_python_role_binds_json_value_to_reviewed_fixture_field() -> None:
    runtime = object.__new__(_DockerWholePageRuntime)
    runtime.runtime_language = "python"
    runtime.route_map = {
        "post_json_state_transition": {
            "route_class": "post_json_state_transition",
            "path": "/fixture/json",
            "input_source": "json",
            "parameter_role": "json_value",
        }
    }
    runtime.route = {
        "route_class": "post_json_state_transition",
        "method": "POST",
        "parameter_role": "json_value",
        "encoding_chain": "json_object_then_utf8",
    }
    runtime.seed = 37901
    runtime.role = "candidate"
    body, path = runtime._route_request("POST")
    assert path == "/fixture/json"
    assert body == b'{"value":"PG379_CANARY_37901_candidate"}'


def test_pg379_redirect_shape_uses_bounded_loopback_location() -> None:
    route = {"method": "GET", "response_shape": "redirect_shape"}
    assert _typed_from_bounded_projection(
        action={"status": 302, "location_class": "loopback", "content_type_class": "text/plain"},
        body=b"redirect_shape",
        route=route,
        role="candidate",
    ) is True
    assert _typed_from_bounded_projection(
        action={"status": 302, "location_class": "none", "content_type_class": "text/plain"},
        body=b"redirect_shape",
        route=route,
        role="candidate",
    ) is False
