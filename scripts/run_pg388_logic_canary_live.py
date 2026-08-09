"""Run the bounded PG-388 logic canary through the local demo HTTP stack.

The live lane is intentionally explicit and local-only.  It accepts only the
enum request contract already exposed by the fixture and stores an abstract
projection of the response.  No raw request, response body, business value,
identifier, wire or external target is copied into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA_VERSION = "pg388-logic-canary-live-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:3000/pg388-api"
CASES = ("nonce_replay", "coupon_reuse_boundary", "subject_resource_scope")
SEQUENCES = (
    ("candidate", "baseline"),
    ("candidate", "candidate"),
    ("reference", "reference"),
    ("negative", "negative"),
    ("replay", "replay"),
)
_OBSERVATION_KEYS = (
    "status",
    "case_ref",
    "role",
    "phase",
    "state_before",
    "state_after",
    "state_delta",
    "effect_shape",
    "action_shape",
    "invariant_holds",
    "vulnerable_effect",
    "typed_observation",
    "negative_control_clean",
    "safe_to_send",
    "target_contacted",
    "external_network",
    "persistent_storage",
    "fresh_reset_required",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _local_base_url(value: str) -> str | None:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if parsed.port not in {None, 80, 3000, 8088}:
        return None
    return value.rstrip("/")


def _request_json(base_url: str, path: str, *, method: str = "GET", payload: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310 - local origin is validated above
            decoded = response.read(128 * 1024).decode("utf-8")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("local_canary_transport_failure") from exc
    try:
        parsed = json.loads(decoded)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("local_canary_invalid_json") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("local_canary_non_object_response")
    return parsed


def _project_observation(document: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _OBSERVATION_KEYS if key not in document]
    if missing:
        raise RuntimeError("local_canary_missing_typed_fields")
    projection = {key: document[key] for key in _OBSERVATION_KEYS}
    if projection["status"] != "typed_local_canary_result" or projection["typed_observation"] is not True:
        raise RuntimeError("local_canary_not_typed")
    if projection["safe_to_send"] is not False or projection["external_network"] is not False:
        raise RuntimeError("local_canary_safety_contract_failure")
    projection["evidence_sha256"] = _sha(projection)
    return projection


def _blocked(reason: str, *, base_url: str, validation: str = "blocked_preflight") -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": validation,
        "base_url": base_url,
        "reason": reason,
        "rows": [],
        "counts": {"fresh_resets": 0, "typed_observations": 0, "candidate_effects": 0, "negative_control_clean": 0, "unsafe_allow": 0},
        "execution": {"local_frontend_contacted": False, "target_contacted": False, "external_network": False, "wire_created": False},
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha(report)
    return report


def run(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 5.0, environ: dict[str, str] | None = None, request: Callable[..., dict[str, Any]] = _request_json) -> dict[str, Any]:
    """Run all three cases or return a fail-closed preflight report."""

    env = os.environ if environ is None else environ
    normalized = _local_base_url(base_url)
    if normalized is None:
        return _blocked("local_origin_required", base_url=base_url)
    if env.get("PG388_LOCAL_EVAL") != "1":
        return _blocked("PG388_LOCAL_EVAL=1_required", base_url=normalized, validation="planning_only_live_blocked")

    rows: list[dict[str, Any]] = []
    reset_count = 0
    candidate_effects = 0
    negative_clean = 0
    unsafe_allow = 0
    try:
        health = request(normalized, "/health", timeout=timeout)
        manifest = request(normalized, "/api/manifest", timeout=timeout)
        if health.get("status") != "ok" or manifest.get("status") != "dynamic_fixture_only_unbound":
            return _blocked("local_fixture_contract_mismatch", base_url=normalized)
        for case_ref in CASES:
            reset = request(normalized, "/api/reset", method="POST", payload={}, timeout=timeout)
            if reset.get("status") != "fresh_reset" or reset.get("state_clean") is not True:
                raise RuntimeError("local_canary_reset_contract_failure")
            reset_count += 1
            for role, phase in SEQUENCES:
                response = request(
                    normalized,
                    "/api/canary",
                    method="POST",
                    payload={"case_ref": case_ref, "role": role, "phase": phase},
                    timeout=timeout,
                )
                projection = _project_observation(response)
                rows.append(projection)
                if projection["vulnerable_effect"] is True:
                    candidate_effects += 1
                if projection["negative_control_clean"] is True:
                    negative_clean += 1
                if projection["safe_to_send"] is True:
                    unsafe_allow += 1
    except RuntimeError as exc:
        report = _blocked(str(exc), base_url=normalized, validation="completed_incomplete_live_canary")
        report["execution"]["local_frontend_contacted"] = True
        report["execution"]["target_contacted"] = False
        report["execution"]["external_network"] = False
        report["counts"]["fresh_resets"] = reset_count
        report["counts"]["typed_observations"] = len(rows)
        return report

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_live_local_canary_only",
        "base_url": normalized,
        "implementation_id": manifest.get("implementation_id", "abstract_local_fixture"),
        "manifest_case_count": manifest.get("case_count"),
        "rows": rows,
        "counts": {
            "fresh_resets": reset_count,
            "typed_observations": len(rows),
            "candidate_effects": candidate_effects,
            "negative_control_clean": negative_clean,
            "unsafe_allow": unsafe_allow,
        },
        "execution": {"local_frontend_contacted": True, "target_contacted": False, "external_network": False, "wire_created": False},
        "model_boundary": {"raw_request_stored": False, "raw_response_stored": False, "evaluator_answer_in_context": False, "safe_to_send": False},
        "training_eligible": False,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha(report)
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", default="research/pg388_logic_canary_live_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run(args.base_url, timeout=args.timeout)
    write_report(args.output, report)
    print(json.dumps({"output": args.output, "status": report["status"], "counts": report["counts"], "report_sha256": report["report_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
