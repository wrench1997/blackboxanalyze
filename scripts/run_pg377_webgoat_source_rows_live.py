"""PG-377 WebGoat whole-page source-row collector.

The collector is deliberately split into a planning lane and an explicitly
authorised live lane.  Planning never imports a target or opens a socket.  A
live run reuses the reviewed PG-333 network-none relay, starts one disposable
container for every seed/route/role, and keeps HTML/request bytes in memory
only long enough for :mod:`app.pg377_webgoat_source_row_adapter` to produce a
de-identified PG-331 source row.

This is a source-row/evaluator evidence collector, not a vulnerability or
payload generator.  The adapter emits all seven ontology axes and the 107
field manifest; missing observations become ``ASK``/``incomplete``.  Typed
method-shape evidence, reset attestations and role-bound SHA-256 evidence stay
in an evaluator sidecar and all promotion flags remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_evaluator_sidecar import build_pg331_evaluator_sidecar, sha256_json  # noqa: E402
from app.pg333_webgoat_docker_relay import (  # noqa: E402
    IMAGE,
    IMAGE_DIGEST,
    ROUTE_PATH,
    DisposableWebGoat,
    build_container_command,
    container_name,
)
from app.pg377_webgoat_source_row_adapter import (  # noqa: E402
    AXES,
    FIELD_COUNT,
    ROLES,
    capture_pg377_webgoat_source_row,
    validate_pg377_webgoat_source_row,
)
from scripts.run_pg333_webgoat_typed_get_post_source_rows import (  # noqa: E402
    _abstract_projection,
    _belief,
    _failure,
    _form_body,
    _typed_effect,
)


SCHEMA_VERSION = "pg377-webgoat-whole-page-source-rows-v1"
SEEDS = (37701, 37702, 37703)
SOURCE_ROLES = ("candidate", "reference", "negative")
ALL_ROLES = (*SOURCE_ROLES, "replay")
ROUTES: tuple[dict[str, str], ...] = (
    {"route_id": "webgoat_login_get", "expected_method": "GET", "surface_id": "login_page_shape"},
    {"route_id": "webgoat_login_post", "expected_method": "POST", "surface_id": "login_redirect_shape"},
)
_AXIS_PRESENCE_KEYS = {
    "document_structure": "document_presence",
    "navigation": "navigation_presence",
    "request_transport": "request_transport_presence",
    "response_transport": "response_transport_presence",
    "javascript_surface": "javascript_presence",
    "failure_feedback": "failure_feedback_presence",
    "belief_and_replay": "belief_replay_presence",
}
_RESET_FIELDS = (
    "fresh_reset",
    "reset_id",
    "target_instance_digest",
    "network_mode",
    "external_network",
    "loopback_only",
    "state_clean",
    "volume_mount_count",
    "container_restart_used",
)

PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}
_RAW_KEYS = frozenset(
    {
        "url",
        "uri",
        "path",
        "location",
        "payload",
        "raw_payload",
        "request_body",
        "request_value",
        "query_value",
        "form_value",
        "response_body",
        "raw_response",
        "body_text",
        "markup",
        "source_code",
        "oracle_answer",
        "evaluator_answer",
        "wire",
    }
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_ref(route: Mapping[str, Any]) -> str:
    """Hash the allowlisted route; never serialize its path literal."""

    return sha256_json(
        {
            "schema": SCHEMA_VERSION,
            "route_id": str(route["route_id"]),
            "method": str(route["expected_method"]),
            "surface_id": str(route["surface_id"]),
            "path_digest": sha256_json(ROUTE_PATH),
            "image_digest": IMAGE_DIGEST,
        }
    )


def _action_method(expected_method: str, role: str) -> str:
    if role == "negative":
        return "POST" if str(expected_method).upper() == "GET" else "GET"
    return str(expected_method).upper()


def _normalize_reset(reset: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only the PG-331 reset contract at the adapter boundary.

    Relay diagnostics such as ``attestation`` and ``readiness_status_class``
    are evaluator/runtime metadata, not source-row reset fields.  Dropping
    them here prevents an otherwise valid fresh reset from being rejected as
    an unknown-field contract violation.
    """

    if not isinstance(reset, Mapping):
        return {}
    return {key: reset[key] for key in _RESET_FIELDS if key in reset}


def _abstract_headers(action: Mapping[str, Any]) -> dict[str, str]:
    """Keep only a safe content-type class; no location/header literal."""

    content_type = str(action.get("content_type_class", "unknown")).casefold()
    return {"Content-Type": content_type} if content_type and content_type != "unknown" else {}


def _request_projection(*, method: str, body_length: int, html_body: bytes) -> dict[str, Any]:
    method = str(method).upper()
    csrf = "present" if any(fragment in html_body.lower() for fragment in (b"csrf", b"xsrf")) else "absent"
    cookie = "absent"
    return {
        "method": method,
        "parameters": (
            [
                {"role": "identity", "value_type": "text", "presence": "present"},
                {"role": "secret", "value_type": "text", "presence": "present"},
            ]
            if method == "POST"
            else []
        ),
        "csrf_presence_class": csrf,
        "cookie_presence_class": cookie,
        "content_length": max(0, min(int(body_length), 2 * 1024 * 1024)),
    }


def _response_projection(*, action: Mapping[str, Any], body: bytes, typed: bool, role: str) -> dict[str, Any]:
    status = int(action.get("status", 0) or 0)
    return {
        "status": status if 100 <= status < 600 else 500,
        "body_length": min(len(body), 2 * 1024 * 1024),
        "body_shape": "empty" if not body else "html",
        "connection_outcome": "complete" if status else "transport_error",
        "charset_class": "utf8" if body else "absent",
        "cache_shape": "absent",
        "csrf_presence_class": "absent",
        "failure_class": "none" if typed or role == "negative" else "response_shape_mismatch",
        "failure_stage": "none" if typed or role == "negative" else "evaluator",
        "error_shape": "empty" if typed or role == "negative" else "shape_difference",
    }


def _failure_projection_for_role(*, expected_method: str, action_method: str, typed: bool, role: str) -> dict[str, Any]:
    """Keep deliberate negative controls visible as a changed action."""

    if role == "negative" and not typed:
        return {
            "failure_class": "response_shape_mismatch",
            "failure_stage": "evaluator",
            "error_shape": "shape_difference",
            "parse_error_class": "none",
            "encoding_error_class": "none",
            "redirect_error_class": "unexpected_method_surface",
            "blocked_reason_class": "none",
            "previous_action": "baseline_observe",
            "next_action": "ask_typed",
            "repair_delta_axis": "response_shape",
            "repair_outcome": "abstain_until_typed",
            "new_observation": "present",
            "retry_count": 0,
            "timeout_bucket": "none",
            "environment_failure_class": "none",
        }
    return _failure(expected_method=expected_method, action_method=action_method, typed=typed)


def _role_input(*, role: str, expected_method: str, action_method: str, action: Mapping[str, Any], body: bytes, typed: bool, evidence: str) -> dict[str, Any]:
    abstract, effect_class = _abstract_projection(action=action, body=body, typed=typed, expected_method=expected_method)
    projection = dict(abstract)
    projection["shape_sha256"] = str(projection.get("shape_sha256", evidence))
    return {
        "sent": True,
        "available": True,
        "executed": True,
        "typed_effect_confirmed": bool(typed),
        "effect_class": effect_class,
        "projection": projection,
        "evidence_sha256": evidence,
        "non_destructive": True,
    }


def _scrub(value: Any, path: str = "$") -> None:
    """Reject raw keys/literals in serialized artifacts, recursively."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in _RAW_KEYS:
                raise ValueError(f"PG-377 raw key at {path}.{key}")
            _scrub(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _scrub(child, f"{path}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        if "http://" in folded or "https://" in folded or "/webgoat" in folded:
            raise ValueError(f"PG-377 raw route/origin at {path}")


def build_pg377_plan(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError("PG-377 requires at least one seed")
    routes = [
        {
            "route_ref_sha256": _route_ref(route),
            "method": str(route["expected_method"]),
            "response_shape": "html_page" if route["expected_method"] == "GET" else "loopback_redirect",
            "path_literal_stored": False,
        }
        for route in ROUTES
    ]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only",
        "implementation": "webgoat",
        "image_digest": IMAGE_DIGEST,
        "relay_module_sha256": _sha(ROOT / "app" / "pg333_webgoat_docker_relay.py"),
        "adapter_module_sha256": _sha(ROOT / "app" / "pg377_webgoat_source_row_adapter.py"),
        "seeds": list(normalized),
        "routes": routes,
        "roles": list(ALL_ROLES),
        "expected_role_replay_count": len(normalized) * len(ROUTES) * len(ALL_ROLES),
        "expected_source_row_count": len(normalized) * len(ROUTES) * len(ALL_ROLES),
        "capture_contract": {
            "seven_axes": list(AXES),
            "field_capture_manifest_count": FIELD_COUNT,
            "get_post_required": True,
            "candidate_reference_negative_replay": True,
            "fresh_reset_per_role": True,
            "failure_repair_belief_replay": True,
            "raw_response_persistence": False,
            "model_context_sidecars_off": True,
        },
        "execution": {
            "live_gate": "PG377_LOCAL_DOCKER_EVAL=1",
            "network_mode": "none",
            "loopback_only": True,
            "published_ports": False,
            "bind_or_volume_mounts": False,
            "docker_started": False,
            "network_contacted": False,
        },
        "promotion": dict(PROMOTION),
        "interpretation": "PG-377 is a fresh whole-page source-row candidate; no vulnerability or payload claim is made.",
    }
    plan["plan_sha256"] = sha256_json(plan)
    _scrub(plan)
    return plan


def _capture_role(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    """Capture one fresh role using the reviewed PG-333 relay."""

    expected = str(route["expected_method"]).upper()
    action_method = _action_method(expected, role)
    name = container_name(seed=int(seed), route_ref_sha256=route_ref, role=role)
    target = DisposableWebGoat(name=name, seed=int(seed), role=role, command=build_container_command(name=name, seed=int(seed), role=role))
    try:
        reset = _normalize_reset(target.start())
        reset.setdefault("volume_mount_count", 0)
        reset.setdefault("container_restart_used", False)
        baseline = target.request(method="GET")
        action = baseline if action_method == "GET" else target.request(method="POST", form_body=_form_body("POST"))
        body = bytes(action.get("body") or b"")
        typed = bool(_typed_effect(expected_method=expected, action_method=action_method, action=action, body=body))
        baseline_body = bytes(baseline.get("body") or b"")
        html_body = body if action_method == "GET" else baseline_body
        evidence = sha256_json(
            {
                "schema": SCHEMA_VERSION,
                "seed": int(seed),
                "route_ref_sha256": route_ref,
                "role": role,
                "reset_id": str(reset.get("reset_id", "")),
                "typed": typed,
                "status_class": str(action.get("status_class", "transport_error")),
                "content_type_class": str(action.get("content_type_class", "unknown")),
                "body_length": len(body),
            }
        )
        request = _request_projection(method=action_method, body_length=len(_form_body("POST")) if action_method == "POST" else 0, html_body=html_body)
        response = _response_projection(action=action, body=body, typed=typed, role=role)
        failure = _failure_projection_for_role(expected_method=expected, action_method=action_method, typed=typed, role=role)
        belief = _belief(expected_method=expected, action_method=action_method, typed=typed, role=role, csrf_class=request["csrf_presence_class"], cookie_class=request["cookie_presence_class"])
        role_input = _role_input(role=role, expected_method=expected, action_method=action_method, action=action, body=body, typed=typed, evidence=evidence)
        return {
            "seed": int(seed),
            "role": role,
            "route_ref_sha256": route_ref,
            "expected_method": expected,
            "action_method": action_method,
            "reset": reset,
            "html": html_body.decode("utf-8", errors="replace"),
            "headers": _abstract_headers(action),
            "request_projection": request,
            "response_projection": response,
            "failure_projection": failure,
            "belief_projection": belief,
            "role_input": role_input,
            "typed": typed,
            "evidence_sha256": evidence,
        }
    finally:
        target.stop()


def _materialize_role(*, captured: Mapping[str, Any], sidecar: Mapping[str, Any], route: Mapping[str, Any], role: str, operator_reviewed: bool = False) -> dict[str, Any]:
    seed = int(captured["seed"])
    route_ref = str(captured["route_ref_sha256"])
    evidence = str(captured["evidence_sha256"])
    source_meta = {
        "source_id": "pg377-webgoat-local",
        "implementation": "webgoat",
        "family_id": "webgoat_login_surface",
        "surface_id": str(route["surface_id"]),
        "collector_id": SCHEMA_VERSION,
        "authorization_id": "operator-authorized-local-network-none",
        "image_digest": IMAGE_DIGEST,
        "source_digest": sha256_json({"seed": seed, "route_ref_sha256": route_ref, "role": role, "evidence_sha256": evidence}),
    }
    html = captured.get("html")
    html_value = html if isinstance(html, str) else None
    request_value = captured.get("request_projection")
    response_value = captured.get("response_projection")
    failure_value = captured.get("failure_projection")
    belief_value = captured.get("belief_projection")
    row = capture_pg377_webgoat_source_row(
        html=html_value,
        headers=dict(captured.get("headers") or {}) if isinstance(captured.get("headers"), Mapping) else None,
        request_projection=dict(request_value) if isinstance(request_value, Mapping) else None,
        response_projection=dict(response_value) if isinstance(response_value, Mapping) else None,
        role=role,
        reset=dict(captured["reset"]) if isinstance(captured.get("reset"), Mapping) else None,
        evaluator_sidecar=dict(sidecar),
        failure_projection=dict(failure_value) if isinstance(failure_value, Mapping) else None,
        belief_projection=dict(belief_value) if isinstance(belief_value, Mapping) else None,
        post_supported=True,
        source_meta=source_meta,
        record_id=f"pg377-webgoat-{seed}-{route['route_id']}-{role}",
        split="implementation_holdout",
        operator_reviewed=bool(operator_reviewed),
        hard_negative=role == "negative",
    )
    validation = validate_pg377_webgoat_source_row(row)
    row["adapter_validation"] = {"valid": bool(validation["valid"]), "failures": list(validation.get("failures") or [])}
    # The adapter hash includes its full output.  Add validation only as a
    # bounded audit field after hashing; the strict nested source row remains
    # hash-verifiable on its own.
    row["record_sha256"] = sha256_json({key: value for key, value in row.items() if key != "record_sha256"})
    return row


def _incomplete_capture(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str, reason: str = "runtime_incomplete") -> dict[str, Any]:
    """Represent a failed/missing observation without guessing any field."""

    evidence = sha256_json({"schema": SCHEMA_VERSION, "seed": int(seed), "route_ref_sha256": route_ref, "role": role, "reason": reason})
    return {
        "seed": int(seed),
        "role": role,
        "route_ref_sha256": route_ref,
        "expected_method": str(route.get("expected_method", "unknown")).upper(),
        "action_method": "unknown",
        "reset": None,
        "html": None,
        "headers": None,
        "request_projection": None,
        "response_projection": None,
        "failure_projection": None,
        "belief_projection": None,
        "role_input": {"sent": False, "available": False, "executed": False, "typed_effect_confirmed": False, "effect_class": "unknown", "projection": {}, "evidence_sha256": evidence, "non_destructive": True},
        "typed": False,
        "evidence_sha256": evidence,
        "capture_failure": reason,
    }


def _sidecar_for_roles(*, record_id: str, reset: Mapping[str, Any] | None, roles: Mapping[str, Mapping[str, Any]], replay: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(reset, Mapping):
        # Preserve an explicit incomplete evaluator state; do not manufacture
        # a fresh reset or typed evidence when the runtime did not return one.
        sidecar = {
            "schema_version": "pg331-evaluator-sidecar-v1",
            "record_id": str(record_id),
            "evaluator_id": "pg377-webgoat-method-shape-evaluator-v1",
            "checks": {key: False for key in ("candidate_present", "reference_present", "negative_present", "candidate_available", "reference_available", "negative_available", "typed_effect", "negative_control_clean", "reference_agreement", "replay_consistent", "fresh_reset", "evidence_hashes", "non_destructive")},
            "evidence_sha256": "",
            "evidence_hash": "",
            "evidence_hash_valid": False,
            "confirmed_positive": False,
            "raw_payload_stored": False,
            "raw_response_stored": False,
            "oracle_answer_in_context": False,
            "replay": {"present": False, "typed_effect_confirmed": False, "evidence_sha256": str(replay.get("evidence_sha256", "")), "fresh_reset": False},
        }
        return sidecar
    sidecar = build_pg331_evaluator_sidecar(
        record_id=record_id,
        reset=dict(reset),
        candidate=dict(roles["candidate"]),
        reference=dict(roles["reference"]),
        negative=dict(roles["negative"]),
        replay_consistent=bool(roles["candidate"].get("typed_effect_confirmed") == replay.get("typed_effect_confirmed") and roles["candidate"].get("typed_effect_confirmed")),
        reference_agreement=bool(roles["candidate"].get("typed_effect_confirmed") and roles["reference"].get("typed_effect_confirmed")),
        negative_control_clean=not bool(roles["negative"].get("typed_effect_confirmed")),
        evaluator_id="pg377-webgoat-method-shape-evaluator-v1",
    )
    # Replay is evaluator-only metadata; it is not copied into source context.
    sidecar["replay"] = {
        "present": True,
        "typed_effect_confirmed": bool(replay.get("typed_effect_confirmed")),
        "evidence_sha256": str(replay.get("evidence_sha256", "")),
        "fresh_reset": True,
    }
    return sidecar


def _fake_capture(*, seed: int, role: str, route: Mapping[str, Any], route_ref: str) -> dict[str, Any]:
    """Small abstract runtime used by tests; it never contacts a target."""

    expected = str(route["expected_method"]).upper()
    action_method = _action_method(expected, role)
    typed = action_method == expected
    page = b"<!doctype html><html><head><title>fixture</title></head><body><form><input name='q'></form></body></html>"
    body = page if action_method == "GET" else b""
    action = {"status": 200 if action_method == "GET" else 302, "status_class": "2xx" if action_method == "GET" else "3xx", "content_type_class": "text/html", "location_class": "loopback" if action_method == "POST" else "none"}
    evidence = sha256_json({"seed": seed, "role": role, "route_ref_sha256": route_ref, "typed": typed})
    request = _request_projection(method=action_method, body_length=40 if action_method == "POST" else 0, html_body=page)
    return {
        "seed": int(seed),
        "role": role,
        "route_ref_sha256": route_ref,
        "expected_method": expected,
        "action_method": action_method,
        "reset": {"reset_id": sha256_json((seed, role, "reset")), "fresh_reset": True, "target_instance_digest": IMAGE_DIGEST, "network_mode": "none", "external_network": False, "loopback_only": True, "state_clean": True, "volume_mount_count": 0, "container_restart_used": False},
        "html": page.decode("utf-8"),
        "headers": {"Content-Type": "text/html"},
        "request_projection": request,
        "response_projection": _response_projection(action=action, body=body, typed=typed, role=role),
        "failure_projection": _failure_projection_for_role(expected_method=expected, action_method=action_method, typed=typed, role=role),
        "belief_projection": _belief(expected_method=expected, action_method=action_method, typed=typed, role=role, csrf_class=request["csrf_presence_class"], cookie_class=request["cookie_presence_class"]),
        "role_input": _role_input(role=role, expected_method=expected, action_method=action_method, action=action, body=body, typed=typed, evidence=evidence),
        "typed": typed,
        "evidence_sha256": evidence,
    }


def collect_pg377_webgoat_source_rows(*, seeds: Sequence[int] = SEEDS, live: bool = False, capture_role: Callable[..., Mapping[str, Any]] | None = None, operator_reviewed: bool = False) -> dict[str, Any]:
    """Return a static plan or collect fresh rows under the explicit gate.

    ``capture_role`` is an injection point for deterministic contract tests;
    production calls leave it unset and use the reviewed Docker relay.
    """

    plan = build_pg377_plan(seeds=seeds)
    if not live:
        return {"report": {**plan, "status": "blocked_live_gate", "interpretation": "Planning only; no WebGoat target was contacted."}, "rows": [], "sidecars": []}
    if os.environ.get("PG377_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-377 live source-row collection requires PG377_LOCAL_DOCKER_EVAL=1")
    capture = capture_role or _capture_role
    rows: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    capture_failures: list[str] = []
    started = time.monotonic()
    for seed_value in seeds:
        seed = int(seed_value)
        for route in ROUTES:
            route_ref = _route_ref(route)
            captured: dict[str, dict[str, Any]] = {}
            for role in ALL_ROLES:
                try:
                    value = capture(seed=seed, role=role, route=route, route_ref=route_ref)
                    captured[role] = dict(value) if isinstance(value, Mapping) else _incomplete_capture(seed=seed, role=role, route=route, route_ref=route_ref)
                    if not isinstance(value, Mapping):
                        capture_failures.append(f"capture:{seed}:{route_ref}:{role}:runtime_incomplete")
                    elif value.get("capture_failure"):
                        capture_failures.append(f"capture:{seed}:{route_ref}:{role}:{str(value.get('capture_failure'))}")
                except Exception:
                    # A target/evaluator failure is an incomplete observation,
                    # never a reason to infer a positive or abort into a
                    # partially trusted dataset.
                    captured[role] = _incomplete_capture(seed=seed, role=role, route=route, route_ref=route_ref)
                    capture_failures.append(f"capture:{seed}:{route_ref}:{role}:runtime_error")
            role_inputs = {role: dict(captured[role]["role_input"]) for role in ALL_ROLES}
            candidate_reset = captured["candidate"].get("reset")
            sidecar = _sidecar_for_roles(record_id=f"pg377-webgoat-{seed}-{route['route_id']}", reset=dict(candidate_reset) if isinstance(candidate_reset, Mapping) else None, roles={role: role_inputs[role] for role in SOURCE_ROLES}, replay=role_inputs["replay"])
            role_rows: dict[str, dict[str, Any]] = {}
            for role in ALL_ROLES:
                role_rows[role] = _materialize_role(captured=captured[role], sidecar=sidecar, route=route, role=role, operator_reviewed=operator_reviewed and role in SOURCE_ROLES)
                rows.append(role_rows[role])
            sidecars.append({"seed": seed, "route_ref_sha256": route_ref, "roles": {role: {"typed_effect_confirmed": bool(role_inputs[role].get("typed_effect_confirmed")), "evidence_sha256": str(role_inputs[role].get("evidence_sha256", ""))} for role in ALL_ROLES}, "evaluator_sidecar": sidecar})
            route_summaries.append({"seed": seed, "route_ref_sha256": route_ref, "method": str(route["expected_method"]), "typed": {role: bool(captured[role]["typed"]) for role in ALL_ROLES}, "source_row_valid": {role: bool(role_rows[role]["adapter_validation"]["valid"]) for role in ALL_ROLES}})
    axis_counts = {
        axis: sum(int(dict(row.get("source_row", row)).get("axis_presence", {}).get(_AXIS_PRESENCE_KEYS[axis]) == "observed") for row in rows)
        for axis in AXES
    }
    seven_axis_complete_count = sum(
        int(
            all(
                dict(row.get("source_row", row)).get("axis_presence", {}).get(_AXIS_PRESENCE_KEYS[axis]) == "observed"
                for axis in AXES
            )
        )
        for row in rows
    )
    valid_count = sum(int(bool(row.get("adapter_validation", {}).get("valid"))) for row in rows)
    strict_failures = sorted({failure for row in rows for failure in list(dict(row.get("source_row") or {}).get("failures") or [])})
    failures = sorted(set(capture_failures) | {failure for row in rows for failure in list(row.get("adapter_validation", {}).get("failures") or [])} | {f"source_row:{failure}" for failure in strict_failures})
    typed_count = sum(int(bool(item.get("typed_effect_confirmed"))) for item in (role for entry in sidecars for role in entry["roles"].values()))
    negative_violations = sum(int(bool(entry["roles"]["negative"]["typed_effect_confirmed"])) for entry in sidecars)
    failure_observed_count = sum(
        int(
            str(
                dict(dict(row.get("observation") or {}).get("failure_feedback") or {}).get("failure_class", "none")
            )
            not in {"", "none", "unknown"}
        )
        for row in rows
    )
    failure_action_changed_count = sum(
        int(
            str(
                dict(dict(row.get("observation") or {}).get("failure_feedback") or {}).get("failure_class", "none")
            )
            not in {"", "none", "unknown"}
            and str(dict(dict(row.get("observation") or {}).get("failure_feedback") or {}).get("previous_action", ""))
            != str(dict(dict(row.get("observation") or {}).get("failure_feedback") or {}).get("next_action", ""))
        )
        for row in rows
    )
    belief_observed_count = sum(int(isinstance(dict(row.get("observation") or {}).get("belief_and_replay"), Mapping)) for row in rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_source_row_candidate_only" if not failures else "completed_incomplete_source_rows",
        "mode": "live_fake_runtime" if capture_role is not None else "live_disposable_webgoat",
        "plan_sha256": plan["plan_sha256"],
        "source_attestation": {"image_digest": IMAGE_DIGEST, "relay_module_sha256": plan["relay_module_sha256"], "adapter_module_sha256": plan["adapter_module_sha256"], "network_mode": "none", "loopback_only": True, "published_ports": False, "bind_or_volume_mounts": False},
        "counts": {"seed_count": len(tuple(seeds)), "route_count": len(ROUTES), "role_replay_count": len(rows), "source_row_count": len(rows), "valid_source_row_count": valid_count, "strict_incomplete_count": sum(int(bool(dict(row.get("source_row") or {}).get("failures"))) for row in rows), "capture_failure_count": len(capture_failures), "training_eligible_count": sum(int(row.get("training_eligible") is True) for row in rows), "typed_role_count": typed_count, "negative_violation_count": negative_violations, "failure_observed_count": failure_observed_count, "failure_action_changed_count": failure_action_changed_count, "belief_observed_count": belief_observed_count, "replay_sidecar_count": len(sidecars), "seven_axis_complete_count": seven_axis_complete_count},
        "axis_presence_counts": axis_counts,
        "field_capture_manifest_count": FIELD_COUNT,
        "failures": failures[:32],
        "route_summaries": route_summaries,
        "hard_gate": {"seven_axes_present": seven_axis_complete_count == len(rows), "field_manifest_107": all(int(row.get("field_capture_manifest_count", 0)) == FIELD_COUNT for row in rows), "candidate_reference_negative_replay": len(rows) == len(tuple(seeds)) * len(ROUTES) * len(ALL_ROLES), "negative_zero_violation": negative_violations == 0, "failure_repair_observed": failure_observed_count > 0 and failure_action_changed_count > 0, "belief_replay_observed": belief_observed_count == len(rows) and len(sidecars) == len(tuple(seeds)) * len(ROUTES), "fresh_reset_per_role": all(bool(row.get("reset_attestation", {}).get("attested")) for row in rows), "context_firewall": all(dict(row.get("context_firewall") or {}).get("forbidden_token_count") == 0 for row in rows), "network_none": True, "no_bind_or_volume": True, "training_promotion": False},
        "execution": {"target_contacted": capture_role is None, "docker_started": capture_role is None, "network_contacted": capture_role is None, "raw_response_persisted": False},
        "promotion": dict(PROMOTION),
        "interpretation": "Fresh WebGoat whole-page source rows are diagnostic/evaluator evidence only; no payload or vulnerability claim is made.",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _scrub(report)
    # Rows are deliberately scrubbed after adapter materialization.  The
    # evaluator sidecar stays separate from context tokens.
    _scrub(rows)
    _scrub(sidecars)
    report["report_sha256"] = sha256_json(report)
    return {"report": report, "rows": rows, "sidecars": sidecars}


def write_artifacts(result: Mapping[str, Any], *, output: Path, dataset_output: Path, sidecar_output: Path) -> dict[str, str]:
    report = dict(result["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [
        dict(row.get("source_row")) if isinstance(row, Mapping) and isinstance(row.get("source_row"), Mapping) else dict(row)
        for row in list(result.get("rows") or [])
    ]
    dataset = {"schema_version": SCHEMA_VERSION, "status": "diagnostic_source_rows", "row_projection": "pg331_source_row_when_complete_else_incomplete_wrapper", "records": records, "promotion": dict(PROMOTION)}
    dataset["dataset_sha256"] = sha256_json(dataset)
    sidecars = {"schema_version": SCHEMA_VERSION, "status": "evaluator_sidecar_only", "sidecars": list(result.get("sidecars") or []), "promotion": dict(PROMOTION)}
    sidecars["sidecars_sha256"] = sha256_json(sidecars)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset_output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sidecar_output.write_text(json.dumps(sidecars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"report": str(output), "dataset": str(dataset_output), "sidecars": str(sidecar_output)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="opt into disposable local WebGoat capture")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg377_webgoat_source_rows_report_v1.json")
    parser.add_argument("--dataset-output", type=Path, default=ROOT / "research" / "pg377_webgoat_source_rows_v1.json")
    parser.add_argument("--sidecar-output", type=Path, default=ROOT / "research" / "pg377_webgoat_source_rows_sidecars_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = collect_pg377_webgoat_source_rows(live=bool(args.live))
    paths = write_artifacts(result, output=args.output, dataset_output=args.dataset_output, sidecar_output=args.sidecar_output)
    summary = {"status": result["report"]["status"], "counts": result["report"].get("counts", {}), "report_sha256": result["report"].get("report_sha256", ""), "artifacts": paths}
    print(json.dumps(summary if not args.json else {"summary": summary, "report": result["report"]}, ensure_ascii=False, indent=2))
    return 0 if result["report"]["status"] in {"planning_only", "blocked_live_gate", "completed_source_row_candidate_only"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALL_ROLES",
    "AXES",
    "FIELD_COUNT",
    "ROUTES",
    "SCHEMA_VERSION",
    "SEEDS",
    "SOURCE_ROLES",
    "build_pg377_plan",
    "collect_pg377_webgoat_source_rows",
    "write_artifacts",
    "_capture_role",
    "_fake_capture",
    "_normalize_reset",
]
