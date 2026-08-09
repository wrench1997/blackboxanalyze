"""Reviewed-contract shell for future PG-332 DVWA live collection.

This file deliberately has no Docker/HTTP/browser imports.  It documents and
validates the only permitted future runtime: fixed digest, network-none target,
127.0.0.1 relay, no publish/mount, and fresh disposable role targets.  The
actual relay/evaluator remains absent, so ``run`` never contacts a target.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_record, sha256_json  # noqa: E402
from app.pg331_source_row import RESET_KEYS, collect_pg331_source_row  # noqa: E402
from app.pg331_vulnerableapp_adapter import capture_vulnerableapp_projection  # noqa: E402
from app.pg332_dvwa_docker_relay import DisposableDvwa  # noqa: E402
from scripts.plan_pg332_dvwa_source_rows import IMAGE, ROLES, SEEDS, _BY_ID, _route_ref  # noqa: E402

SCHEMA_VERSION = "pg332-dvwa-source-row-live-contract-v1"
SOURCE_ROLES = ("candidate", "reference", "negative")
RELAY_HOST = "127.0.0.1"
CONTAINER_NAME_RE = re.compile(r"^pg332-dvwa-[0-9]+-[a-f0-9]{12}-(candidate|reference|negative|replay)$")


def _identity(seed: int, route: Mapping[str, Any], role: str) -> str:
    return sha256_json({"schema": SCHEMA_VERSION, "seed": seed, "route_ref": _route_ref(route), "role": role, "image": IMAGE, "network": "none", "fresh": True})


def container_name(*, seed: int, route_ref_sha256: str, role: str) -> str:
    """Create the only runtime container identity; never accepts caller names."""
    if role not in ROLES or not re.fullmatch(r"[a-f0-9]{64}", str(route_ref_sha256)):
        raise ValueError("PG-332 requires an allowlisted role and route digest")
    return f"pg332-dvwa-{int(seed)}-{route_ref_sha256[:12]}-{role}"


def build_container_command(*, seed: int, route_ref_sha256: str, role: str) -> tuple[str, ...]:
    """Pure fixed-digest Docker invocation for a future PHP docker-exec relay.

    The target has no published host port: a PHP process would run through
    ``docker exec`` and only talk to the target's own loopback.  This helper
    does not execute Docker.
    """
    name = container_name(seed=seed, route_ref_sha256=route_ref_sha256, role=role)
    return (
        "docker", "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--label", "sift.pg332=true", "--label", f"sift.pg332.reset_epoch={int(seed)}:{route_ref_sha256[:12]}:{role}",
        "--network", "none", "--cap-drop", "ALL",
        # DVWA's legacy Apache/MariaDB entrypoint needs only these file/UID
        # setup capabilities.  They do not create a network or persistence
        # channel and match the already-reviewed Pikachu relay contract.
        "--cap-add", "DAC_OVERRIDE", "--cap-add", "CHOWN", "--cap-add", "FOWNER",
        "--cap-add", "SETUID", "--cap-add", "SETGID", "--security-opt", "no-new-privileges",
        # The writable layer is disposable and is removed with ``--rm``.  Do
        # not add tmpfs/bind/volume mounts here: the PG-332 attestation treats
        # every mount as a separate persistence surface and rejects it.
        "--pids-limit", "256", "--memory", "1g", IMAGE,
    )


def attest_container_inspection(inspection: Mapping[str, Any], *, expected_name: str) -> dict[str, Any]:
    """Fail closed unless a just-created target is fixed-image/network-none/no-mount."""
    if not CONTAINER_NAME_RE.fullmatch(expected_name):
        raise ValueError("PG-332 expected container identity is invalid")
    failures: list[str] = []
    if str(inspection.get("name", "")).lstrip("/") != expected_name: failures.append("container_name")
    if str(inspection.get("image", "")) != IMAGE: failures.append("image_digest")
    if str(inspection.get("network_mode", "")) != "none": failures.append("network_mode")
    if list(inspection.get("mounts") or []): failures.append("mounts")
    if list(inspection.get("published_ports") or []): failures.append("published_ports")
    if inspection.get("relay_host") != RELAY_HOST: failures.append("relay_host")
    if inspection.get("legacy_bridge_reclassified") is not False: failures.append("legacy_bridge_reclassified")
    return {"valid": not failures, "failures": failures, "container_name": expected_name, "network_mode": "none", "loopback_relay": RELAY_HOST}


def build_live_contract(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Return static episodes only; paths/probes/evaluator answers stay private."""
    episodes = []
    for seed in (int(value) for value in seeds):
        for route in _BY_ID.values():
            method = str(route["method"])
            stateful = bool(route.get("stateful_disposable"))
            episodes.append({"seed": seed, "route_ref_sha256": _route_ref(route), "method": method, "roles": {role: {"target_identity_sha256": _identity(seed, route, role), "fresh_reset_required": True, "fresh_reset_observed": False, "database_clean_attestation_required": True, "teardown_required": True, "source_row_allowed": role in SOURCE_ROLES} for role in ROLES}, "transport": {"network_mode": "none", "relay_host": RELAY_HOST, "loopback_only": True, "published_ports_allowed": False, "bind_or_volume_mounts_allowed": False, "legacy_bridge_reclassification_allowed": False}, "evaluator": {"candidate_reference_negative_required": True, "replay_required": True, "role_bound_evidence_required": True, "mode": "state_delta_only" if stateful else "typed_response_or_dom", "post_typed_available": "unknown_until_evaluator" if method == "POST" else "unknown_until_evaluator"}, "model_projection": {"next_action": "ask_typed", "safe_to_send": False}, "training_eligible": False})
    result = {"schema_version": SCHEMA_VERSION, "status": "planning_only_reviewed_contract", "execution": {"docker_started": False, "network_contacted": False, "image": IMAGE, "runtime_gate": "PG332_LOCAL_DOCKER_EVAL=1", "network_mode": "none", "relay_host": RELAY_HOST}, "episodes": episodes, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    result["contract_sha256"] = sha256_json(result)
    return result


def build_evaluator_sidecar(*, seed: int, route_ref_sha256: str, reset: Mapping[str, Any], candidate: Mapping[str, Any], reference: Mapping[str, Any], negative: Mapping[str, Any], replay_consistent: bool) -> dict[str, Any]:
    """Bind bounded role projections; rejects raw material in shared sidecar code."""
    record = build_pg331_evaluator_record(record_id=f"pg332:{seed}:{route_ref_sha256[:16]}", reset=reset, candidate=candidate, reference=reference, negative=negative, replay_consistent=replay_consistent, evaluator_id="pg332-dvwa-reviewed-evaluator-v1")
    return {"evaluator_sidecar": record["evaluator_sidecar"], "model_context": record["model_context"], "record_sha256": record["record_sha256"], "training_eligible": False}


def bind_source_row(*, seed: int, route_id: str, role: str, observation: Mapping[str, Any], field_capture_manifest: Mapping[str, Any], reset: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    """Pure strict-row binding; missing evaluator/fields becomes ASK, not send."""
    if route_id not in _BY_ID or role not in SOURCE_ROLES: raise ValueError("allowlisted route and candidate/reference/negative role required")
    route = _BY_ID[route_id]
    target = {"question": "ask_typed", "next_action": "ask_typed", "repair_action": "observe", "transport_ref": "unknown", "field_role_ref": "unknown", "encoding_ref": "unknown", "probe_variant_ref": "none", "safe_to_send": False}
    # The Docker attestation contains useful evaluator-only details (container
    # name, mount/port counts, teardown flag), but PG-331 rows accept only the
    # small reset projection declared by the source-row schema.
    row_reset = {str(key): reset[key] for key in RESET_KEYS if key in reset}
    return collect_pg331_source_row(record_id=f"pg332:{seed}:{_route_ref(route)[:16]}:{role}", observation=observation, source_meta={"source_id": "pg332-dvwa-local", "implementation": "vulnerables-web-dvwa", "collector_id": SCHEMA_VERSION, "authorization_id": "operator-authorized-local-network-none", "image_digest": IMAGE.split("@sha256:", 1)[1], "source_digest": sha256_json({"seed": seed, "route_ref": _route_ref(route), "role": role, "evidence": evaluator.get("evidence_hash", "")})}, reset=row_reset, evaluator=evaluator, field_capture_manifest=field_capture_manifest, target_projection=target, split="implementation_holdout", operator_reviewed=False, hard_negative=role == "negative")


def run(*, transport_factory: Any | None = None) -> dict[str, Any]:
    """Run only when explicitly armed; absent reviewed evaluator is an ASK artifact.

    ``transport_factory`` is injection-only for future reviewed runtime code and
    tests.  It is never constructed implicitly, so an environment flag alone
    cannot accidentally start Docker.  The historic bridge is deliberately
    not accepted as a factory or attestation source.
    """
    if os.environ.get("PG332_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-332 live collection requires PG332_LOCAL_DOCKER_EVAL=1")
    contract = build_live_contract()
    if transport_factory is not None:
        raise RuntimeError("PG-332 reviewed PHP docker-exec relay/evaluator is not wired; refusing injected transport until review")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete_environment_failure",
        "environment_failure": "reviewed_dvwa_login_and_typed_evaluator_unavailable",
        "target_contacted": False,
        "docker_started": False,
        "legacy_bridge_reclassified": False,
        "contract_sha256": contract["contract_sha256"],
        "model_projection": {"question": "ask_typed", "next_action": "ask_typed", "safe_to_send": False},
        "training_eligible": False,
        "promotion": contract["promotion"],
    }


def _baseline_evaluator(*, route_ref_sha256: str, role: str, reset: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Create a diagnostic baseline sidecar; it is never a positive label."""

    evidence = sha256_json({"route_ref_sha256": route_ref_sha256, "role": role, "reset_id": reset.get("reset_id"), "observation": observation})
    return {
        "typed_available": False,
        "negative_control": False,
        "reference_present": False,
        "candidate_present": False,
        "fresh_reset": bool(reset.get("fresh_reset")),
        "evidence_hash": evidence,
        "confirmed_positive": False,
        "effect_class": "baseline_structure_only",
        "evaluator_version": "pg332-dvwa-baseline-no-typed-oracle-v1",
    }


def _route_request(
    target: Any,
    route: Mapping[str, Any],
    *,
    probe_values: Mapping[str, str] | None = None,
    belief_projection: Mapping[str, Any] | None = None,
    failure_projection: Mapping[str, Any] | None = None,
    effect_probe: Any | None = None,
    post_supported: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Send only an empty/neutral request and reduce it in memory."""

    method = str(route["method"]).upper()
    fields = [str(route.get("field", ""))] if route.get("field") else []
    values = {str(key): str(value) for key, value in dict(probe_values or {}).items()}
    if not values and fields:
        values = {field: "" for field in fields}
    parameters = [{"role": "query_term" if method == "GET" else "form_field", "value_type": "text", "presence": "present"} for _ in values]
    if method == "GET":
        path = str(route["path"])
        if values:
            path += "?" + urlencode(values)
        response = target.request("GET", path, body=b"")
        content_type = str(dict(response.get("headers") or {}).get("content-type", ""))
        location = str(dict(response.get("headers") or {}).get("location", ""))
        request_projection = {"method": "GET", "parameters": parameters, "csrf_presence_class": "absent", "cookie_presence_class": "present", "content_length": len(urlencode(values).encode("utf-8"))}
    else:
        body = urlencode(values).encode("utf-8")
        response = target.request("POST", str(route["path"]), body=body)
        content_type = str(dict(response.get("headers") or {}).get("content-type", ""))
        location = str(dict(response.get("headers") or {}).get("location", ""))
        request_projection = {"method": "POST", "parameters": parameters, "csrf_presence_class": "absent", "cookie_presence_class": "present", "content_length": len(body)}
    body_bytes = bytes(response.get("body") or b"")
    status = int(response.get("status", 0) or 0)
    body_shape = "html" if "html" in content_type.casefold() or body_bytes.lstrip().lower().startswith((b"<!doctype", b"<html", b"<body")) else "text" if body_bytes else "empty"
    response_headers = dict(response.get("headers") or {})
    cache_names = {"cache-control", "pragma", "expires", "etag", "age", "last-modified", "vary"}
    header_names = {str(key).casefold() for key in response_headers}
    content_type_lower = content_type.casefold()
    charset_class = "utf8" if "charset=utf-8" in content_type_lower.replace(" ", "") or "charset=utf8" in content_type_lower.replace(" ", "") else "other" if "charset=" in content_type_lower else "absent"
    response_projection = {
        "status": status,
        "body_length": len(body_bytes),
        "body_shape": body_shape,
        "connection_outcome": "complete" if status else "transport_error",
        "failure_class": "none" if status else "connection_error",
        "failure_stage": "none" if status else "transport",
        "error_shape": "empty" if status else "connection_error",
        "charset_class": charset_class,
        "cache_shape": "present" if header_names & cache_names else "absent",
        "csrf_presence_class": "absent" if status else "unknown",
    }
    # Only bounded header classes enter the adapter; the cookie value and raw
    # body stay inside this evaluator-side call and are discarded afterwards.
    headers = {"content-type": content_type} if content_type else {}
    if location:
        headers["location"] = location if location.startswith("/") else "external_or_unknown"
    captured = capture_vulnerableapp_projection(
        html=body_bytes.decode("utf-8", errors="replace"),
        headers=headers,
        request_projection=request_projection,
        response_projection=response_projection,
        post_supported=bool(post_supported),
        failure_projection=failure_projection,
        belief_projection=belief_projection,
    )
    effect_projection = None
    if effect_probe is not None:
        candidate_effect = effect_probe(body_bytes, status, response_headers)
        if not isinstance(candidate_effect, Mapping):
            raise ValueError("PG-332 effect probe must return an abstract mapping")
        effect_projection = {str(key): value for key, value in candidate_effect.items()}
    content_type_class = "html" if "html" in content_type.casefold() else "json" if "json" in content_type.casefold() else "text" if "text" in content_type.casefold() else "other" if content_type else "absent"
    return captured, {"status": status, "content_type_class": content_type_class, "body_length": len(body_bytes), "body_shape": body_shape, "location_present": bool(location)}, effect_projection


def run_baseline(*, seeds: Sequence[int] = (33299,), route_ids: Sequence[str] = ("dvwa-xss-reflected-get",), roles: Sequence[str] = ROLES) -> dict[str, Any]:
    """Collect real whole-page GET/POST structure, with typed oracle absent.

    This is intentionally a diagnostic lane.  It exercises the real fresh
    target/relay/HTML parser but cannot create training rows or vulnerability
    labels until a reviewed candidate/reference/negative evaluator is bound.
    """

    if os.environ.get("PG332_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-332 baseline requires PG332_LOCAL_DOCKER_EVAL=1")
    requested_routes = [str(value) for value in route_ids]
    requested_roles = [str(value) for value in roles]
    if not requested_routes or any(value not in _BY_ID for value in requested_routes):
        raise ValueError("PG-332 baseline route is not allowlisted")
    if not requested_roles or any(value not in ROLES for value in requested_roles):
        raise ValueError("PG-332 baseline role is not allowlisted")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in (int(value) for value in seeds):
        for route_id in requested_routes:
            route = _BY_ID[route_id]
            route_ref = _route_ref(route)
            for role in requested_roles:
                name = container_name(seed=seed, route_ref_sha256=route_ref, role=role)
                target = DisposableDvwa(name=name, seed=seed, index=len(episodes), command=build_container_command(seed=seed, route_ref_sha256=route_ref, role=role))
                try:
                    reset = target.start()
                    capture, response_shape, _ = _route_request(target, route)
                    observation = dict(capture["observation"])
                    evaluator = _baseline_evaluator(route_ref_sha256=route_ref, role=role, reset=reset, observation=observation)
                    if role in SOURCE_ROLES:
                        row = bind_source_row(seed=seed, route_id=route_id, role=role, observation=observation, field_capture_manifest=capture["field_capture_manifest"], reset=reset, evaluator=evaluator)
                        rows.append(row)
                    episodes.append({"seed": seed, "route_ref_sha256": route_ref, "role": role, "method": str(route["method"]).upper(), "target_contacted": True, "fresh_reset": True, "response_shape": response_shape, "typed_available": False, "training_eligible": False})
                except Exception as error:
                    errors.append({"seed": seed, "route_ref_sha256": route_ref, "role": role, "error_class": type(error).__name__})
                finally:
                    target.stop()
    report = {
        "schema_version": "pg332-dvwa-source-row-live-report-v1",
        "status": "completed_diagnostic_only" if episodes and not errors else "incomplete",
        "runtime": {"image": IMAGE, "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False, "target_contacted": bool(episodes), "elapsed_seconds": round(time.monotonic() - started, 3)},
        "counts": {"episode_count": len(episodes), "source_row_count": len(rows), "get_count": sum(int(item["method"] == "GET") for item in episodes), "post_count": sum(int(item["method"] == "POST") for item in episodes), "typed_count": 0, "ask_count": len(rows), "errors": len(errors), "training_eligible": 0},
        "episodes": episodes,
        "errors": errors,
        "information_status": "diagnostic_only_typed_evaluator_missing",
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "真实 DVWA 页面/GET/POST 结构已回放；candidate/reference/negative typed evaluator 尚未接线，所有行只能 ASK/diagnostic，不能训练或声明漏洞。",
    }
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "episodes": episodes}


if __name__ == "__main__":
    if os.environ.get("PG332_BASELINE_LIVE") == "1":
        print(json.dumps(run_baseline(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_live_contract(), ensure_ascii=False, indent=2))
