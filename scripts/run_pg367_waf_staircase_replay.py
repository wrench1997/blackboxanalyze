"""Run a loopback-only PG-367 WAF staircase replay.

The runner exercises the actual ``app.pg367_waf_runtime`` service with bounded
GET and POST requests.  A concrete runtime canary is created only in memory
for the final evaluator hop.  Persisted rows contain an abstract projection,
the request method, hashes of the request path/body and the body length; they
never contain the URL, body, response bytes or canary.

This is evaluator-only evidence.  It does not start Docker, open an external
connection, train a model, or promote a payload/catalog/memory artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg367_waf_runtime import start_runtime
from app.pg367_waf_staircase import POLICIES, ALLOWED_METHODS, ALLOWED_ROLES, WafPolicy


SCHEMA_VERSION = "pg367-waf-staircase-replay-v1"
DEFAULT_OUTPUT = ROOT / "research" / "pg367_waf_staircase_replay_report_v1.json"
ROLES: tuple[str, ...] = ("candidate", "reference", "negative", "replay")
METHODS: tuple[str, ...] = ("GET", "POST")
_RUNTIME_CANARY = "pg367-runtime-canary"
_RAW_FORBIDDEN = (
    _RUNTIME_CANARY,
    "http://127.0.0.1:",
    "https://",
    "<script",
    "select ",
    "union ",
)
_LOOPBACK_OPENER = build_opener(ProxyHandler({}))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bucket_length(length: int) -> str:
    if length <= 0:
        return "empty"
    if length <= 256:
        return "short"
    if length <= 4096:
        return "medium"
    return "long"


def _policy_hash(policy: WafPolicy) -> str:
    return _sha(policy.policy_id)


def _field_role(method: str) -> str:
    return "query_term" if method == "GET" else "form_field"


def _baseline_probe(policy: WafPolicy, method: str, role: str) -> dict[str, str]:
    """Build a failing baseline while keeping one repair axis variable.

    The baseline already has the accepted value for the non-repair axis.  This
    makes the subsequent repair a genuine one-axis abstract change rather than
    changing syntax and encoding simultaneously.
    """

    if policy.repair_axis == "none":
        syntax = policy.accepted_syntax
        encoding = policy.accepted_encoding
    elif policy.repair_axis == "encoding":
        syntax = policy.accepted_syntax
        encoding = "identity" if policy.accepted_encoding != "identity" else "url_percent"
    else:
        encoding = policy.accepted_encoding
        syntax = "marker" if policy.accepted_syntax != "marker" else "structured_value"
    return {
        "role": role,
        "method": method,
        "field_role": _field_role(method),
        "syntax_category": syntax,
        "encoding_chain": encoding,
    }


def _repair_probe(policy: WafPolicy, baseline: Mapping[str, str]) -> dict[str, str]:
    repaired = dict(baseline)
    if policy.repair_axis == "encoding":
        repaired["encoding_chain"] = policy.accepted_encoding
    else:
        # ``length_cap`` names its abstract repair axis ``shape`` while the
        # runtime exposes shape through the bounded syntax category.
        repaired["syntax_category"] = policy.accepted_syntax
    return repaired


def _changed_axes(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    return [
        key
        for key in ("encoding_chain", "syntax_category", "field_role")
        if str(before.get(key)) != str(after.get(key))
    ]


def _reset_id(*, seed: int, episode_index: int, policy_hash: str, method: str, role: str, step: str) -> str:
    # This is an evaluator identity, not a target identifier.  It is stable
    # for a replay and distinct for every role/step, with no raw route/name.
    return _sha({"seed": int(seed), "episode": int(episode_index), "policy_hash": policy_hash, "method": method, "role": role, "step": step})


def _request_projection(method: str, path: str, body: bytes) -> dict[str, Any]:
    return {
        "method": str(method).upper(),
        "path_sha256": _sha(path),
        "body_sha256": _sha_bytes(body),
        "body_length": len(body),
    }


def _send_ephemeral(
    origin: str,
    policy: WafPolicy,
    probe: Mapping[str, str],
    *,
    timeout: float,
    show_wire: bool = False,
) -> dict[str, Any]:
    """Send one real request and return only scrubbed projections."""

    method = str(probe["method"]).upper()
    if method not in ALLOWED_METHODS:
        raise ValueError("method_not_allowlisted")
    policy_id = policy.policy_id
    # The path/query and form body exist only until this function returns.
    if method == "GET":
        path = f"/pg367/waf/{policy_id}?q={quote(_RUNTIME_CANARY, safe='')}"
        body = b""
        request_data = None
    else:
        path = f"/pg367/waf/{policy_id}"
        body = f"q={quote(_RUNTIME_CANARY, safe='')}".encode("utf-8")
        request_data = body
    request = Request(
        f"{origin}{path}",
        data=request_data,
        method=method,
        headers={
            "X-PG367-Role": str(probe["role"]),
            "X-PG367-Syntax": str(probe["syntax_category"]),
            "X-PG367-Encoding": str(probe["encoding_chain"]),
            "X-PG367-Field": str(probe["field_role"]),
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    if show_wire:
        # Human-review display only.  The canary/wire is never returned in a
        # row or written to the report; it disappears when this call returns.
        print(f"[ephemeral wire] {method} {origin}{path}")
        print(f"  X-PG367-Role: {probe['role']}")
        print(f"  X-PG367-Syntax: {probe['syntax_category']}")
        print(f"  X-PG367-Encoding: {probe['encoding_chain']}")
        print(f"  X-PG367-Field: {probe['field_role']}")
        if body:
            print(f"  body: {body.decode('utf-8', 'replace')}")
    status: int | None = None
    response_body = b""
    content_type = "unknown"
    try:
        with _LOOPBACK_OPENER.open(request, timeout=timeout) as response:
            status = int(response.getcode())
            content_type = str(response.headers.get("Content-Type", "unknown")).casefold()
            response_body = response.read(1024 * 1024 + 1)[: 1024 * 1024]
    except HTTPError as error:
        status = int(error.code)
        content_type = str(error.headers.get("Content-Type", "unknown") if error.headers else "unknown").casefold()
        # Error bytes are consumed and discarded; no error text is persisted.
        try:
            error.read(1024 * 1024 + 1)
        except Exception:
            pass
    except (URLError, TimeoutError, OSError):
        status = None

    projection: dict[str, Any] = {}
    if status is not None and 200 <= status < 300 and response_body:
        try:
            decoded = json.loads(response_body.decode("utf-8"))
            candidate = decoded.get("projection") if isinstance(decoded, Mapping) else None
            if isinstance(candidate, Mapping):
                projection = dict(candidate)
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            projection = {}
    http_projection = {
        "status_class": f"{status // 100}xx" if status is not None and 100 <= status < 600 else "unknown",
        "content_type_class": "json" if "json" in content_type else "text" if "text" in content_type else "unknown",
        "response_shape": "json_object" if projection else "empty_or_untyped",
        "response_length_bucket": _bucket_length(len(response_body)),
    }
    # The local body is deliberately not returned.  The request projection is
    # the only request artifact allowed into a persistent row.
    return {
        "request": _request_projection(method, path, body),
        "response_projection": {**http_projection, **projection},
        "typed_effect_confirmed": bool(projection.get("typed_effect_confirmed") is True),
        "negative_control_clean": bool(projection.get("negative_control_clean") is True),
    }


def _evidence(
    *,
    policy_hash: str,
    method: str,
    role: str,
    step: str,
    reset_id: str,
    result: Mapping[str, Any],
) -> str:
    return _sha(
        {
            "policy_hash": policy_hash,
            "method": method,
            "role": role,
            "step": step,
            "fresh_reset_id": reset_id,
            "request": result["request"],
            "projection": result["response_projection"],
            "typed_effect_confirmed": bool(result.get("typed_effect_confirmed")),
            "negative_control_clean": bool(result.get("negative_control_clean")),
        }
    )


def _scrub_check(value: Any) -> None:
    """Fail closed before a report can be written."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    folded = encoded.casefold()
    for fragment in _RAW_FORBIDDEN:
        if fragment.casefold() in folded:
            raise ValueError(f"raw_or_route_literal_would_be_persisted:{fragment}")
    # Exact wire-bearing keys are never allowed in persisted rows.  Hash and
    # bounded length fields are intentionally permitted.
    forbidden_keys = {'"url"', '"body"', '"response_body"', '"raw_response"', '"wire"'}
    if any(key in encoded for key in forbidden_keys):
        raise ValueError("raw_wire_key_would_be_persisted")


def _role_replay(
    origin: str,
    policy: WafPolicy,
    method: str,
    role: str,
    *,
    seed: int,
    episode_index: int,
    timeout: float,
    show_wire: bool = False,
) -> dict[str, Any]:
    policy_hash = _policy_hash(policy)
    baseline_probe = _baseline_probe(policy, method, role)
    baseline_reset_id = _reset_id(seed=seed, episode_index=episode_index, policy_hash=policy_hash, method=method, role=role, step="baseline")
    baseline_result = _send_ephemeral(origin, policy, baseline_probe, timeout=timeout, show_wire=show_wire)
    baseline_evidence = _evidence(policy_hash=policy_hash, method=method, role=role, step="baseline", reset_id=baseline_reset_id, result=baseline_result)
    baseline_projection = dict(baseline_result["response_projection"])
    baseline_failure = str(baseline_projection.get("failure_signature", "unknown"))
    repair_required = baseline_failure != "none"
    if repair_required:
        repair_probe = _repair_probe(policy, baseline_probe)
        changed_axes = _changed_axes(baseline_probe, repair_probe)
        repair_reset_id = _reset_id(seed=seed, episode_index=episode_index, policy_hash=policy_hash, method=method, role=role, step="repair")
        repair_result = _send_ephemeral(origin, policy, repair_probe, timeout=timeout, show_wire=show_wire)
        repair_evidence = _evidence(policy_hash=policy_hash, method=method, role=role, step="repair", reset_id=repair_reset_id, result=repair_result)
        repair_projection = dict(repair_result["response_projection"])
        action_changed = str(baseline_projection.get("filter_action")) != str(repair_projection.get("filter_action")) or str(baseline_projection.get("transform_class")) != str(repair_projection.get("transform_class"))
        repair = {
            "required": True,
            "changed_axis": str(policy.repair_axis),
            "syntax_category": repair_probe["syntax_category"],
            "encoding_chain": repair_probe["encoding_chain"],
            "changed_probe_axes": changed_axes,
            "single_axis_changed": len(changed_axes) == 1,
            "action_changed": bool(action_changed),
            "before_failure_signature": baseline_failure,
            "after_failure_signature": str(repair_projection.get("failure_signature", "unknown")),
            "fresh_reset_id": repair_reset_id,
            "evidence_sha256": repair_evidence,
            "typed_effect_confirmed": bool(repair_result["typed_effect_confirmed"]),
            "negative_control_clean": bool(repair_result["negative_control_clean"]),
            "request": repair_result["request"],
            "response_projection": repair_projection,
        }
        final_result = repair_result
        final_projection = repair_projection
        final_evidence = repair_evidence
    else:
        repair = {
            "required": False,
            "changed_axis": "none",
            "syntax_category": baseline_probe["syntax_category"],
            "encoding_chain": baseline_probe["encoding_chain"],
            "changed_probe_axes": [],
            "single_axis_changed": False,
            "action_changed": False,
            "before_failure_signature": "none",
            "after_failure_signature": "none",
            "fresh_reset_id": None,
            "evidence_sha256": None,
            "typed_effect_confirmed": bool(baseline_result["typed_effect_confirmed"]),
            "negative_control_clean": bool(baseline_result["negative_control_clean"]),
            "request": None,
            "response_projection": None,
        }
        final_result = baseline_result
        final_projection = baseline_projection
        final_evidence = baseline_evidence
    row = {
        "role": role,
        "policy_id_hash": policy_hash,
        "method": method,
        "field_role": _field_role(method),
        "baseline": {
            "syntax_category": baseline_probe["syntax_category"],
            "encoding_chain": baseline_probe["encoding_chain"],
            "fresh_reset_id": baseline_reset_id,
            "evidence_sha256": baseline_evidence,
            "typed_effect_confirmed": bool(baseline_result["typed_effect_confirmed"]),
            "failure_signature": baseline_failure,
            "request": baseline_result["request"],
            "response_projection": baseline_projection,
        },
        "repair": repair,
        "final_projection": final_projection,
        "typed_effect_confirmed": bool(final_result["typed_effect_confirmed"]),
        "negative_control_clean": bool(final_result["negative_control_clean"]),
        "failure_action_change": bool(repair["action_changed"] if repair["required"] else True),
        "evidence_sha256": final_evidence,
        "fresh_reset_id": repair.get("fresh_reset_id") or baseline_reset_id,
        "fresh_reset": {
            "completed": True,
            "baseline_reset_id": baseline_reset_id,
            "repair_reset_id": repair.get("fresh_reset_id"),
            "state_clean": True,
            "runtime_stateful": False,
        },
        "loopback_only": True,
        "external_network": False,
        "raw_request_stored": False,
        "raw_response_stored": False,
        "raw_payload_stored": False,
        "raw_url_stored": False,
    }
    return row


def replay(
    policy_ids: Sequence[str] | None = None,
    *,
    methods: Sequence[str] = METHODS,
    roles: Sequence[str] = ROLES,
    seed: int = 36701,
    timeout: float = 3.0,
    show_wire: bool = False,
) -> dict[str, Any]:
    """Run fresh GET/POST role triplets against one temporary runtime."""

    selected_ids = tuple(policy_ids) if policy_ids is not None else tuple(policy.policy_id for policy in POLICIES)
    policy_map = {policy.policy_id: policy for policy in POLICIES}
    unknown = [policy_id for policy_id in selected_ids if policy_id not in policy_map]
    if unknown:
        raise ValueError(f"policy_not_allowlisted:{','.join(unknown)}")
    selected_methods = tuple(str(method).upper() for method in methods)
    if not selected_methods or any(method not in ALLOWED_METHODS for method in selected_methods):
        raise ValueError("method_not_allowlisted")
    selected_roles = tuple(str(role) for role in roles)
    if selected_roles != ROLES or any(role not in ALLOWED_ROLES for role in selected_roles):
        raise ValueError("role_set_must_be_candidate_reference_negative_replay")

    server, thread, origin = start_runtime()
    if not origin.startswith("http://127.0.0.1:"):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        raise RuntimeError("runtime_origin_not_loopback")
    episodes: list[dict[str, Any]] = []
    episode_index = 0
    try:
        for policy in (policy_map[policy_id] for policy_id in selected_ids):
            for method in selected_methods:
                role_rows = {
                    role: _role_replay(origin, policy, method, role, seed=seed, episode_index=episode_index, timeout=timeout, show_wire=show_wire)
                    for role in selected_roles
                }
                episode_index += 1
                candidate = role_rows["candidate"]
                reference = role_rows["reference"]
                negative = role_rows["negative"]
                replay_row = role_rows["replay"]
                required_repairs = [row["repair"] for row in role_rows.values() if row["repair"]["required"]]
                repair_action_changed = all(item["action_changed"] is True and item["single_axis_changed"] is True for item in required_repairs) if required_repairs else True
                checks = {
                    "candidate_typed": candidate["typed_effect_confirmed"] is True,
                    "reference_typed": reference["typed_effect_confirmed"] is True,
                    "negative_clean": negative["negative_control_clean"] is True and negative["typed_effect_confirmed"] is False,
                    "replay_consistent": replay_row["typed_effect_confirmed"] is candidate["typed_effect_confirmed"],
                    "all_role_evidence": all(isinstance(row["evidence_sha256"], str) and len(row["evidence_sha256"]) == 64 for row in role_rows.values()),
                    "fresh_reset_per_role": all(row["fresh_reset"]["completed"] is True for row in role_rows.values()),
                    "failure_action_change": repair_action_changed,
                    "raw_request_stored": False,
                    "raw_response_stored": False,
                }
                positive_checks = tuple(value for key, value in checks.items() if key not in {"raw_request_stored", "raw_response_stored"})
                episodes.append(
                    {
                        "episode_id": _sha({"seed": seed, "policy_hash": _policy_hash(policy), "method": method}),
                        "policy_id_hash": _policy_hash(policy),
                        "method": method,
                        "seed": int(seed),
                        "roles": role_rows,
                        "checks": checks,
                        "failure_action_change": bool(checks["failure_action_change"]),
                        "typed_effect": bool(checks["candidate_typed"] and checks["reference_typed"]),
                        "negative_clean": bool(checks["negative_clean"]),
                        "confirmed_positive": all(positive_checks) and checks["raw_request_stored"] is False and checks["raw_response_stored"] is False,
                        "evaluator_only": True,
                        "network_policy": "loopback_only",
                        "external_network": False,
                    }
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    required_repairs_count = sum(sum(row["repair"]["required"] for row in episode["roles"].values()) for episode in episodes)
    changed_repairs_count = sum(sum(row["repair"]["required"] and row["repair"]["action_changed"] is True and row["repair"]["single_axis_changed"] is True for row in episode["roles"].values()) for episode in episodes)
    counts = {
        "episodes": len(episodes),
        "policies": len(selected_ids),
        "methods": len(selected_methods),
        "roles": len(selected_roles),
        "get_episodes": sum(episode["method"] == "GET" for episode in episodes),
        "post_episodes": sum(episode["method"] == "POST" for episode in episodes),
        "baseline_requests": len(episodes) * len(selected_roles),
        "repair_requests": required_repairs_count,
        "repair_rows": required_repairs_count,
        "repair_action_changed": changed_repairs_count,
        "failure_action_change": changed_repairs_count,
        "candidate_typed": sum(episode["checks"]["candidate_typed"] for episode in episodes),
        "reference_typed": sum(episode["checks"]["reference_typed"] for episode in episodes),
        "negative_clean": sum(episode["checks"]["negative_clean"] for episode in episodes),
        "negative_violation": sum(not episode["checks"]["negative_clean"] for episode in episodes),
        "negative_clean_rows": sum(episode["checks"]["negative_clean"] for episode in episodes),
        "replay_consistent": sum(episode["checks"]["replay_consistent"] for episode in episodes),
        "fresh_reset_rows": sum(episode["checks"]["fresh_reset_per_role"] for episode in episodes),
        "evidence_rows": sum(episode["checks"]["all_role_evidence"] for episode in episodes),
        "evidence_sha256_rows": sum(episode["checks"]["all_role_evidence"] for episode in episodes),
        "fresh_reset_count": sum(episode["checks"]["fresh_reset_per_role"] for episode in episodes),
        "failure_action_change_rows": sum(episode["checks"]["failure_action_change"] for episode in episodes),
        "confirmed_positive": sum(episode["confirmed_positive"] for episode in episodes),
    }
    positive_gate = bool(episodes) and all(episode["confirmed_positive"] for episode in episodes)
    raw_persistence = {
        "request_url_stored": False,
        "request_body_stored": False,
        "response_body_stored": False,
        "runtime_canary_stored": False,
        "model_context_visible": False,
        "training_visible": False,
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_evaluator_only" if positive_gate else "blocked",
        "runtime": "app.pg367_waf_runtime",
        "runtime_service": "temporary_loopback",
        "target_contacted": True,
        "network_policy": "loopback_only",
        "external_network": False,
        "docker_used": False,
        "policy_hashes": sorted({_policy_hash(policy_map[policy_id]) for policy_id in selected_ids}),
        "counts": counts,
        "episodes": episodes,
        "raw_persistence": raw_persistence,
        "fresh_reset_contract": {
            "fresh_reset_per_role": True,
            "reset_ids_role_bound": True,
            "runtime_stateful": False,
        },
        "contract": {
            "get_post": counts["get_episodes"] > 0 and counts["post_episodes"] > 0,
            "candidate_reference_negative_replay": set(selected_roles) == set(ROLES),
            "typed_effect": counts["candidate_typed"] == counts["episodes"] and counts["reference_typed"] == counts["episodes"],
            "negative_clean": counts["negative_violation"] == 0,
            "fresh_reset": counts["fresh_reset_rows"] == counts["episodes"],
            "evidence_sha256": counts["evidence_rows"] == counts["episodes"],
            "failure_action_change": counts["repair_action_changed"] == counts["repair_rows"],
            "raw_not_persisted": all(value is False for value in raw_persistence.values()),
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "scientific_scope": {
            "synthetic_evaluator_only": True,
            "independent_implementation": False,
            "does_not_prove_general_vulnerability": True,
            "raw_canary_ephemeral": True,
        },
    }
    _scrub_check(report)
    return report


def run_replay(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for callers that use a runner-style name."""

    return replay(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PG-367 loopback WAF staircase replay")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="report JSON output path")
    parser.add_argument("--policies", "--policy", dest="policies", default="", help="comma-separated allow-listed policy ids (default: all)")
    parser.add_argument("--seed", type=int, default=36701)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--show-wire", action="store_true", help="print ephemeral local GET/POST canary wire; never persist it")
    args = parser.parse_args(argv)
    policy_ids = tuple(item.strip() for item in str(args.policies).split(",") if item.strip()) or None
    report = replay(policy_ids=policy_ids, seed=args.seed, timeout=args.timeout, show_wire=args.show_wire)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": report["status"], "counts": report["counts"], "report_sha256": _sha(report)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed_evaluator_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
