"""PG-368 WebGoat Rule-IR adapter (fail-closed, evaluator-only).

PG-368 is the second-implementation bridge for the current research model.
The reviewed plan deliberately projects ``ASK``/``safe_to_send=0`` because
there is no model-side typed evaluator for WebGoat yet.  Consequently the
default command is a dry-run: it records the abstract slots that are missing
and never starts Docker.  It is not a neural positive and it cannot promote a
payload or a vulnerability claim.

An operator may opt into a *structural method-shape canary* with
``--live`` and ``PG368_LOCAL_DOCKER_EVAL=1``.  That lane uses only the
source-attested PG-333 WebGoat relay, a disposable ``network=none`` container,
invalid non-secret login form values, and a GET/POST response-shape oracle.
The lane is evaluator-only: its typed result is reported as a method-shape
observation, while ``confirmed_positive`` remains false because the model
abstained.  Raw route literals, values, headers, response bytes and wire are
never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg333_webgoat_docker_relay import (  # noqa: E402
    IMAGE,
    IMAGE_DIGEST,
    DisposableWebGoat,
    build_container_command,
    container_name,
)
from scripts.plan_pg368_second_implementation import (  # noqa: E402
    ROUTES,
    SEEDS,
    build_pg368_second_implementation_plan,
    route_ref_sha256,
    sha256_file,
    sha256_json,
)
from scripts.run_pg333_webgoat_typed_get_post_source_rows import (  # noqa: E402
    _form_body,
    _typed_effect,
)


SCHEMA_VERSION = "pg368-webgoat-model-binder-replay-v1"
ROLES = ("candidate", "reference", "negative", "replay")
_RAW_KEY_NAMES = {"url", "uri", "body", "response_body", "payload", "raw_payload", "raw_value", "wire"}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _opposite(method: str) -> str:
    return "POST" if str(method).upper() == "GET" else "GET"


def _route_shape(route: Mapping[str, Any]) -> dict[str, str]:
    """Return only the canonical abstract route shape (no path literal)."""

    method = str(route.get("method", "")).upper()
    return {
        "route_ref_sha256": route_ref_sha256(route),
        "transport_ref": "get_query" if method == "GET" else "post_form" if method == "POST" else "unknown",
        "response_shape_ref": str(route.get("response_shape", "unknown")),
        "method": method,
    }


def _ask_projection(route: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Canonical model projection.  It is intentionally non-sendable."""

    method = str(route.get("method", "")).upper()
    return {
        "question": "ask_typed",
        "next_action": "ask_typed",
        "repair_action": "observe",
        "transport_ref": "request_method",
        "field_role_ref": "credential_pair" if method == "POST" else "none",
        "encoding_ref": "form_urlencoded" if method == "POST" else "none",
        "probe_variant_ref": {
            "candidate": "source_attested_candidate",
            "reference": "reference",
            "negative": "negative_control",
            "replay": "fresh_replay",
        }[role],
        "safe_to_send": False,
        "model_selected": False,
        "model_status": "ASK_missing_typed_evaluator",
    }


def bind_rule_ir(rule_ir: Mapping[str, Any], *, route: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Validate a canonical abstract Rule-IR without creating a wire.

    The PG-368 plan has no typed oracle, therefore *every* current projection
    is fail-closed.  This function exists so the adapter contract is explicit
    and testable rather than silently treating a method shape as a payload.
    """

    if str(role) not in ROLES:
        raise ValueError("invalid role")
    if str(rule_ir.get("safe_to_send")).casefold() in {"true", "1"} or rule_ir.get("safe_to_send") is True:
        raise ValueError("PG-368 cannot bind safe Rule-IR before typed evaluator")
    required = {"question", "next_action", "transport_ref", "encoding_ref", "probe_variant_ref", "safe_to_send"}
    if not required.issubset(set(rule_ir)):
        raise ValueError("PG-368 Rule-IR missing abstract slot")
    shape = _route_shape(route)
    return {
        "status": "ASK",
        "safe_to_send": False,
        "model_selected": False,
        "route_ref_sha256": shape["route_ref_sha256"],
        "transport_ref": shape["transport_ref"],
        "response_shape_ref": shape["response_shape_ref"],
        "role": str(role),
        "reason": "typed_evaluator_unavailable",
    }


def _projection(action: Mapping[str, Any]) -> dict[str, Any]:
    """Project relay response metadata without body/header values."""

    status = int(action.get("status", 0) or 0)
    content_type = str(action.get("content_type_class", "unknown"))
    return {
        "status_class": str(action.get("status_class", "transport_error")),
        "status_shape": "numeric" if status else "absent",
        "content_type_class": content_type,
        "location_class": str(action.get("location_class", "none")),
        "body_shape": "html" if len(bytes(action.get("body") or b"")) > 100 else "empty_or_small",
        "body_length_bucket": "empty" if not action.get("body") else "small_or_medium",
        "connection_outcome": "complete" if status else "transport_error",
    }


def _scrub(value: Any, *, path: str = "$") -> None:
    """Reject raw wire-shaped keys and obvious route/value leakage."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _RAW_KEY_NAMES:
                raise ValueError(f"raw_key:{path}.{key}")
            _scrub(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scrub(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        for fragment in ("http://", "https://", "/webgoat", "probe.invalid"):
            if fragment in folded:
                raise ValueError(f"raw_text:{fragment}")


def _dry_run(plan: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for episode in list(plan.get("episodes") or []):
        method = str(episode.get("method", "")).upper()
        # Reuse only the reviewed in-memory route shape.  The route id/path
        # never leaves the process; preserving the same route hash keeps the
        # binder contract role/route-bound without leaking a literal.
        route = next((candidate for candidate in ROUTES if str(candidate.get("method", "")).upper() == method), None)
        if route is None:
            raise ValueError("PG-368 plan contains an unknown method shape")
        for role in ROLES:
            model_projection = _ask_projection(route, role)
            binding = bind_rule_ir(model_projection, route=route, role=role)
            rows.append(
                {
                    "seed": int(episode.get("seed", 0)),
                    "route_ref_sha256": str(episode.get("route_ref_sha256", "")),
                    "method": str(episode.get("method", "")),
                    "response_shape_ref": str(episode.get("response_shape", "")),
                    "role": role,
                    "model_projection": model_projection,
                    "binding": binding,
                    "typed_method_shape_confirmed": False,
                    "confirmed_positive": False,
                    "abstain": True,
                    "reason": "ASK_missing_typed_evaluator",
                    "target_contacted": False,
                }
            )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked_model_ask",
        "mode": "dry_run",
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "source_attestation": {
            "image_digest": IMAGE_DIGEST,
            "relay_module_sha256": sha256_file(ROOT / "app" / "pg333_webgoat_docker_relay.py"),
            "network_mode": "none",
            "loopback_only": True,
        },
        "counts": {
            "episodes": len(list(plan.get("episodes") or [])),
            "roles": len(rows),
            "ask_rows": len(rows),
            "target_contacted": 0,
            "typed_method_shape_confirmed": 0,
            "confirmed_positive": 0,
            "unsafe_allow": 0,
        },
        "rows": rows,
        "scientific_scope": {
            "model_selected_abstract_slots": False,
            "model_projection_is_ask": True,
            "webgoat_method_shape_evaluator": False,
            "general_vulnerability_claim": False,
        },
        "raw_persistence": {
            "model_context_raw": False,
            "request_target_stored": False,
            "request_value_stored": False,
            "response_bytes_stored": False,
            "wire_stored": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "PG-368 has no model-selected typed WebGoat action; all rows are ASK and remain diagnostic.",
    }
    _scrub(report)
    report["report_sha256"] = _digest(report)
    return report


def _live_role(*, seed: int, route: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Run one structural method-shape role in a fresh WebGoat container."""

    method = str(route.get("method", "")).upper()
    route_ref = route_ref_sha256(route)
    name = container_name(seed=int(seed), route_ref_sha256=route_ref, role=role)
    target = DisposableWebGoat(
        name=name,
        seed=int(seed),
        role=role,
        command=build_container_command(name=name, seed=int(seed), role=role),
    )
    started = time.monotonic()
    try:
        reset = target.start()
        action_method = _opposite(method) if role == "negative" else method
        action = target.request(method=action_method, form_body=_form_body(action_method))
        action_body = bytes(action.get("body") or b"")
        typed = bool(_typed_effect(expected_method=method, action_method=action_method, action=action, body=action_body))
        projection = _projection(action)
        evidence = _digest(
            {
                "seed": int(seed),
                "role": role,
                "route_ref_sha256": route_ref,
                "reset_id": str(reset.get("reset_id", "")),
                "typed": typed,
                "projection": projection,
            }
        )
        return {
            "seed": int(seed),
            "route_ref_sha256": route_ref,
            "method": method,
            "role": role,
            "action_method": action_method,
            "fresh_reset": bool(reset.get("fresh_reset")),
            "target_attested": bool(dict(reset.get("attestation") or {}).get("attested")),
            "typed_method_shape_confirmed": typed,
            "projection": projection,
            "evidence_sha256": evidence,
            "elapsed_bucket": "short" if (time.monotonic() - started) < 10 else "bounded",
            "model_selected": False,
            "confirmed_positive": False,
            "target_contacted": True,
        }
    finally:
        target.stop()


def replay(*, live: bool = False, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Build a dry-run report or explicitly authorized structural replay."""

    plan = build_pg368_second_implementation_plan(seeds=seeds)
    if not live:
        return _dry_run(plan)
    if os.environ.get("PG368_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG368 live replay requires PG368_LOCAL_DOCKER_EVAL=1")
    rows: list[dict[str, Any]] = []
    for route in ROUTES:
        for seed in seeds:
            for role in ROLES:
                result = _live_role(seed=int(seed), route=route, role=role)
                # The model remains ASK, so typed method-shape evidence is
                # never promoted to a model positive.
                result["model_projection"] = _ask_projection(route, role)
                result["abstain"] = True
                rows.append(result)
    typed = sum(bool(row.get("typed_method_shape_confirmed")) for row in rows)
    negative_violations = sum(bool(row.get("role") == "negative" and row.get("typed_method_shape_confirmed")) for row in rows)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_evaluator_only_method_shape",
        "mode": "live_structural_canary",
        "plan_sha256": str(plan.get("plan_sha256", "")),
        "source_attestation": {
            "image_digest": IMAGE_DIGEST,
            "relay_module_sha256": sha256_file(ROOT / "app" / "pg333_webgoat_docker_relay.py"),
            "network_mode": "none",
            "loopback_only": True,
            "external_network": False,
            "published_ports": False,
            "bind_or_volume_mounts": False,
        },
        "counts": {
            "episodes": len(rows) // len(ROLES),
            "roles": len(rows),
            "target_contacted": len(rows),
            "typed_method_shape_confirmed": typed,
            "negative_violation": negative_violations,
            "confirmed_positive": 0,
            "model_selected": 0,
            "ask_rows": len(rows),
        },
        "rows": rows,
        "scientific_scope": {
            "model_selected_abstract_slots": False,
            "model_projection_is_ask": True,
            "webgoat_method_shape_evaluator": True,
            "general_vulnerability_claim": False,
            "method_shape_is_not_vulnerability": True,
        },
        "raw_persistence": {
            "model_context_raw": False,
            "request_target_stored": False,
            "request_value_stored": False,
            "response_bytes_stored": False,
            "wire_stored": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": "Evaluator-only GET/POST method-shape observations; the model abstained and no vulnerability positive was claimed.",
    }
    _scrub(report)
    report["report_sha256"] = _digest(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="opt into the disposable local method-shape canary")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg368_webgoat_binder_replay_report_v1.json")
    args = parser.parse_args()
    report = replay(live=bool(args.live))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"blocked_model_ask", "completed_evaluator_only_method_shape"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "ROLES", "bind_rule_ir", "replay"]
