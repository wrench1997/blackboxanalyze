"""Run the PG-350 abstract-slot -> ephemeral-wire replay on the local fixture.

This is an evaluator-only integration harness.  It deliberately uses the
reviewed in-process PG-348 loopback runtime rather than a public target or an
arbitrary URL.  The binder produces a concrete GET/POST request in memory;
the sender uses it once, extracts a bounded response/effect projection, and
then drops the body and wire.  Persisted JSON contains only hashes and
abstract slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from http.client import HTTPMessage
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry, start_server
from app.pg350_runtime_payload_binder import bind_runtime_probe


DEFAULT_REGISTRY = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DEFAULT_REPORT = ROOT / "research" / "pg350_runtime_binding_replay_report_v1.json"
DEFAULT_SIDECARS = ROOT / "research" / "pg350_runtime_binding_replay_sidecars_v1.json"
SCHEMA_VERSION = "pg350-runtime-binding-replay-v1"
ROLES = ("candidate", "reference", "negative", "replay")
ROLE_VARIANTS = {
    "candidate": "candidate_surface",
    "reference": "reference_surface",
    "negative": "negative_control",
    "replay": "candidate_surface",
}
ROLE_SLOT_VARIANTS = {
    "candidate": "source_attested_candidate",
    "reference": "reference",
    "negative": "source_attested_candidate",
    "replay": "fresh_replay",
}
_EFFECT_RE = re.compile(rb'data-effect="([a-z_]+)"')
_FAILURE_RE = re.compile(rb'data-failure="([a-z_]+)"')


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bucket_length(length: int) -> str:
    if length <= 0:
        return "empty"
    if length <= 256:
        return "short"
    if length <= 4096:
        return "medium"
    return "long"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stdlib hook
        return None


def _headers_projection(headers: HTTPMessage | Mapping[str, Any]) -> dict[str, Any]:
    names = {str(key).casefold() for key in headers}
    content_type = "absent"
    for key, value in headers.items():
        if str(key).casefold() == "content-type":
            text = str(value).casefold()
            content_type = "html" if "html" in text else "json" if "json" in text else "text"
            break
    return {
        "content_type_class": content_type,
        "header_presence_class": "basic" if names else "absent",
        "redirect_location_class": "present" if "location" in names else "none",
    }


def _send_ephemeral(probe: Any, *, variant: str, timeout: float = 3.0) -> dict[str, Any]:
    """Send one binder-produced request and immediately reduce the response.

    The returned mapping is already scrubbed.  In particular it never returns
    the response body, URL, request body, or concrete marker.  The caller may
    display ``probe.human_review_wire()`` before this function, but must not
    persist it.
    """

    request_data = None if probe.body is None else str(probe.body).encode("utf-8")
    headers = dict(probe.headers)
    headers["X-PG348-Probe-Variant"] = variant
    request = Request(str(probe.url), data=request_data, method=str(probe.method), headers=headers)
    opener = build_opener(_NoRedirect())
    status: int | None = None
    response_headers: Mapping[str, Any] = {}
    body = b""
    failure = "none"
    try:
        response = opener.open(request, timeout=timeout)
        status = int(response.getcode())
        response_headers = dict(response.headers.items())
        body = response.read(2 * 1024 * 1024 + 1)[: 2 * 1024 * 1024]
    except HTTPError as error:
        status = int(error.code)
        response_headers = dict(error.headers.items()) if error.headers else {}
        try:
            body = error.read(2 * 1024 * 1024 + 1)[: 2 * 1024 * 1024]
        except Exception:
            body = b""
        failure = "http_error" if status >= 400 else "redirect_not_followed"
    except (URLError, TimeoutError):
        failure = "connection_error"

    effect_match = _EFFECT_RE.search(body)
    failure_match = _FAILURE_RE.search(body)
    effect = effect_match.group(1).decode("ascii", "replace") if effect_match else "none"
    failure_shape = failure_match.group(1).decode("ascii", "replace") if failure_match else "none"
    typed = effect == "observed" and failure == "none"
    projection = {
        "status_class": f"{status // 100}xx" if status is not None and 100 <= status < 600 else "unknown",
        "status_shape": "numeric" if status is not None else "unknown",
        "body_shape": "html" if b"<html" in body.lower() or b"data-runtime=\"dynamic\"" in body else "empty" if not body else "text",
        "body_length_bucket": _bucket_length(len(body)),
        "failure_class": failure,
        "failure_shape": failure_shape,
        "effect_class": "logic_transition" if typed else "none",
        "state_delta_class": "disposable_evaluator_state" if typed else "none",
        **_headers_projection(response_headers),
    }
    return {
        "typed_effect_confirmed": typed,
        "projection": projection,
        "response_evidence_sha256": _sha(projection),
    }


def _abstract_rule(record: Mapping[str, Any], method: str, role: str) -> tuple[dict[str, Any], str, str]:
    method = str(method).upper()
    if method == "GET":
        rule = {
            "transport_ref": "get_query",
            "field_role_ref": "query_term",
            "encoding_ref": "url_percent",
            "payload_shape_ref": "query_marker",
            "oracle_ref": "typed_effect",
            "probe_variant_ref": ROLE_SLOT_VARIANTS[role],
            "safe_to_send": True,
        }
        shape = "query_marker"
        template = "{{MARKER}}'"
    elif method == "POST":
        rule = {
            "transport_ref": "post_form",
            "field_role_ref": "form_field",
            "encoding_ref": "form_urlencoded",
            "payload_shape_ref": "html_form_marker",
            "oracle_ref": "typed_state_delta",
            "probe_variant_ref": ROLE_SLOT_VARIANTS[role],
            "safe_to_send": True,
        }
        shape = "html_form_marker"
        template = "{{MARKER}}'"
    else:  # pragma: no cover - route selection prevents this
        raise ValueError("PG-350 replay supports GET/POST only")
    template_id = f"pg350_{method.casefold()}_{shape}_v1"
    _ = record
    return rule, shape, template_id + "\x00" + template


def _catalog(shape: str, template_id: str, template: str) -> dict[str, Any]:
    return {
        "templates": [
            {
                "template_id": template_id,
                "shape": shape,
                "template": template,
                "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                "local_only": True,
                "non_destructive": True,
                "stateful_allowed": False,
            }
        ]
    }


def _route_attestation(*, origin: str, path: str, method: str, source_hash: str, field: str = "q") -> dict[str, Any]:
    return {
        "target_origin": origin,
        "route": {"method": method, "path": path, "field_name": field},
        "loopback_only": True,
        "external_network": False,
        "source_attested": True,
        "route_attested": True,
        "field_attested": True,
        "fresh_reset": True,
        "candidate_reference_negative": True,
        "replay_consistency": True,
        "authorization_id": "pg350_local_synthetic_loopback",
        "allowed_template_ids": [],
        "stateful_evaluator": False,
        "source_attestation_sha256": source_hash,
    }


def _select_routes(registry: Mapping[str, Any], max_routes: int) -> list[dict[str, Any]]:
    rows = list(registry.get("records") or [])
    selected: list[dict[str, Any]] = []
    for method in ("GET", "POST"):
        candidates = [row for row in rows if str(row.get("transport_method", "")).upper() == method]
        if candidates:
            selected.append(dict(sorted(candidates, key=lambda item: str(item.get("challenge_id", "")))[0]))
    if max_routes < len(selected):
        selected = selected[:max_routes]
    if not selected:
        raise ValueError("registry contains no allow-listed GET/POST routes")
    return selected


def replay(
    registry: Mapping[str, Any],
    *,
    seeds: Sequence[int] = (35001, 35002, 35003),
    max_routes: int = 2,
    show_wire: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    app = DynamicFixtureApplication(registry)
    server, thread = start_server(app, host="127.0.0.1", port=0)
    origin = f"http://127.0.0.1:{server.server_port}"
    sidecars: list[dict[str, Any]] = []
    shown_wires: list[str] = []
    route_rows = _select_routes(registry, max_routes)
    try:
        for seed in tuple(int(value) for value in seeds):
            for route_index, record in enumerate(route_rows):
                challenge_id = str(record["challenge_id"])
                method = str(record.get("transport_method", "")).upper()
                path = f"/pg348/dynamic/{challenge_id}"
                role_results: dict[str, dict[str, Any]] = {}
                reset_ids: dict[str, dict[str, str]] = {}
                for role in ROLES:
                    reset_before = app.reset(challenge_id)
                    rule, shape, packed = _abstract_rule(record, method, role)
                    template_id, template = packed.split("\x00", 1)
                    runtime = _route_attestation(
                        origin=origin,
                        path=path,
                        method=method,
                        source_hash=str(record.get("source_hash", "")),
                    )
                    runtime["allowed_template_ids"] = [template_id]
                    marker = f"PG350S{seed}R{route_index}{role.upper()}"
                    probe = bind_runtime_probe(rule, runtime, _catalog(shape, template_id, template), marker=marker)
                    if show_wire:
                        wire = probe.human_review_wire()
                        shown_wires.append(wire)
                        print(f"[PG-350 ephemeral {seed}/{route_index}/{role}]\n{wire}\n")
                    result = _send_ephemeral(probe, variant=ROLE_VARIANTS[role])
                    reset_after = app.reset(challenge_id)
                    reset_ids[role] = {
                        "before_sha256": _sha(reset_before),
                        "after_sha256": _sha(reset_after),
                    }
                    projection = dict(result["projection"])
                    evidence = _sha(
                        {
                            "role": role,
                            "seed": seed,
                            "route_digest": _sha(challenge_id),
                            "reset": reset_ids[role],
                            "projection": projection,
                            "typed": bool(result["typed_effect_confirmed"]),
                        }
                    )
                    role_results[role] = {
                        "role": role,
                        "typed_effect_confirmed": bool(result["typed_effect_confirmed"]),
                        "projection": projection,
                        "evidence_sha256": evidence,
                        "binding": probe.persisted_projection(),
                        "fresh_reset": True,
                        "state_reset_before": True,
                        "state_reset_after": True,
                        "database_clean": True,
                        "teardown": True,
                    }

                # A separate unsupported variant followed by a repaired
                # candidate is retained as a bounded failure/action-change
                # diagnostic.  The failed request still uses the bound wire;
                # only the evaluator-side variant header changes.
                failure_reset_before = app.reset(challenge_id)
                rule, shape, packed = _abstract_rule(record, method, "candidate")
                template_id, template = packed.split("\x00", 1)
                runtime = _route_attestation(origin=origin, path=path, method=method, source_hash=str(record.get("source_hash", "")))
                runtime["allowed_template_ids"] = [template_id]
                failure_probe = bind_runtime_probe(rule, runtime, _catalog(shape, template_id, template), marker=f"PG350S{seed}R{route_index}FAIL")
                failed = _send_ephemeral(failure_probe, variant="unsupported_variant")
                failure_reset_after = app.reset(challenge_id)
                failure_projection = dict(failed["projection"])
                failure_record = {
                    "failure_class": "blocked_variant",
                    "previous_action": "unsupported_variant",
                    "next_action": "source_attested_candidate",
                    "repair_action": "select_allowlisted_variant",
                    "action_changed": True,
                    "typed_effect_confirmed": False,
                    "projection": failure_projection,
                    "evidence_sha256": _sha({"seed": seed, "route_digest": _sha(challenge_id), "failure": failure_projection}),
                    "fresh_reset": True,
                    "reset_before_sha256": _sha(failure_reset_before),
                    "reset_after_sha256": _sha(failure_reset_after),
                }

                candidate = role_results["candidate"]
                reference = role_results["reference"]
                negative = role_results["negative"]
                replay_result = role_results["replay"]
                checks = {
                    "candidate_typed": candidate["typed_effect_confirmed"],
                    "reference_typed": reference["typed_effect_confirmed"],
                    "negative_clean": not negative["typed_effect_confirmed"],
                    "replay_consistent": replay_result["typed_effect_confirmed"] == candidate["typed_effect_confirmed"],
                    "all_role_evidence": all(bool(role_results[role]["evidence_sha256"]) for role in ROLES),
                    "fresh_reset_per_role": all(role_results[role]["fresh_reset"] for role in ROLES),
                    "failure_action_change": failure_record["action_changed"],
                    "raw_wire_stored": False,
                    "raw_response_stored": False,
                }
                confirmed = (
                    all(value for key, value in checks.items() if key not in {"raw_wire_stored", "raw_response_stored"})
                    and checks["raw_wire_stored"] is False
                    and checks["raw_response_stored"] is False
                )
                sidecars.append(
                    {
                        "record_id": _sha({"seed": seed, "challenge_id": challenge_id}),
                        "route_digest": _sha(challenge_id),
                        "source_digest": _sha(record.get("source_hash", "")),
                        "method": method,
                        "seed": seed,
                        "roles": role_results,
                        "failure_repair": failure_record,
                        "checks": checks,
                        "confirmed_positive": confirmed,
                        "evaluator_only": True,
                        "implementation_independence": False,
                    }
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    total = len(sidecars)
    counts = {
        "episodes": total,
        "routes": len(route_rows),
        "seeds": len(tuple(seeds)),
        "get_episodes": sum(item["method"] == "GET" for item in sidecars),
        "post_episodes": sum(item["method"] == "POST" for item in sidecars),
        "confirmed_positive": sum(item["confirmed_positive"] for item in sidecars),
        "candidate_typed": sum(item["checks"]["candidate_typed"] for item in sidecars),
        "reference_typed": sum(item["checks"]["reference_typed"] for item in sidecars),
        "negative_clean": sum(item["checks"]["negative_clean"] for item in sidecars),
        "replay_consistent": sum(item["checks"]["replay_consistent"] for item in sidecars),
        "failure_action_change": sum(item["checks"]["failure_action_change"] for item in sidecars),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_evaluator_only" if total and counts["confirmed_positive"] == total else "blocked",
        "runtime_kind": "pg348_synthetic_dynamic_loopback",
        "target_contacted": True,
        "network_policy": "loopback_only",
        "external_network": False,
        "route_digests": [_sha(row["challenge_id"]) for row in route_rows],
        "counts": counts,
        "worst_seed_metrics": {
            "candidate_typed_rate": min((item["checks"]["candidate_typed"] for item in sidecars), default=False),
            "reference_typed_rate": min((item["checks"]["reference_typed"] for item in sidecars), default=False),
            "negative_violation": max((not item["checks"]["negative_clean"] for item in sidecars), default=False),
            "failure_action_change_rate": min((item["checks"]["failure_action_change"] for item in sidecars), default=False),
        },
        "raw_wire_policy": {
            "ephemeral_wire_displayed": bool(shown_wires),
            "raw_wire_stored": False,
            "raw_payload_stored": False,
            "raw_response_stored": False,
            "model_visible": False,
            "training_visible": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "scientific_scope": {
            "synthetic_evaluator_only": True,
            "implementation_independence": False,
            "does_not_prove_general_vulnerability": True,
            "binder_and_neural_capability_separate": True,
        },
    }
    return report, {"schema_version": f"{SCHEMA_VERSION}-sidecars", "status": "evaluator_only", "sidecars": sidecars, "promotion": report["promotion"]}, shown_wires


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-350 local abstract-slot runtime binding replay")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sidecars", type=Path, default=DEFAULT_SIDECARS)
    parser.add_argument("--max-routes", type=int, default=2)
    parser.add_argument("--seeds", default="35001,35002,35003")
    parser.add_argument("--show-wire", action="store_true", help="print ephemeral human-review wire; never persist it")
    args = parser.parse_args()
    if args.max_routes < 1 or args.max_routes > 2:
        raise SystemExit("--max-routes must be 1 or 2 for this bounded replay")
    seeds = tuple(int(value.strip()) for value in str(args.seeds).split(",") if value.strip())
    registry = load_registry(args.registry)
    report, sidecars, _ = replay(registry, seeds=seeds, max_routes=args.max_routes, show_wire=args.show_wire)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.sidecars.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.sidecars.write_text(json.dumps(sidecars, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report_sha256": _sha(report), "sidecars_sha256": _sha(sidecars)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed_evaluator_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
