"""Collect bounded dynamic GET/POST shape traces for PG-348.

The collector exercises the loopback runtime, but stores only abstract
response/state projections.  Candidate/reference/negative/replay roles are
kept as evaluator-side identities; typed effects are deliberately unavailable
until a reviewed oracle is attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg348_dynamic_runtime import DynamicFixtureApplication, load_registry


DEFAULT_REGISTRY = ROOT / "fixtures" / "pg348" / "registry_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg348_dynamic_shape_trace_v1.json"
ROLES = ("candidate", "reference", "negative", "replay")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bucket(value: int) -> str:
    return "zero" if value <= 0 else "one" if value == 1 else "two" if value == 2 else "few" if value <= 5 else "many"


def _abstract_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "status_class": "redirect" if int(response.get("status", 0)) in {301, 302, 303, 307, 308} else "success" if int(response.get("status", 0)) < 400 else "error",
        "content_type_class": "json" if "json" in str(response.get("content_type", "")) else "html",
        "body_length_bucket": _bucket(int(response.get("body_length", 0))),
        "redirect_shape": str(response.get("redirect_shape", "none")),
        "input_presence": str(response.get("input_presence", "unknown")),
        "state_delta": str(response.get("state_delta", "none")),
        "state_event_count": _bucket(int(response.get("state_event_count", 0))),
    }


def collect(registry: dict[str, Any]) -> dict[str, Any]:
    app = DynamicFixtureApplication(registry)
    rows: list[dict[str, Any]] = []
    for record in registry.get("records") or []:
        challenge_id = str(record.get("challenge_id"))
        method = str(record.get("transport_method", "GET")).upper()
        for role in ROLES:
            reset = app.reset(challenge_id)
            if method == "GET":
                response = app.handle(method, challenge_id, {"probe": ["opaque"]}, None)
            else:
                response = app.handle(method, challenge_id, None, {"probe": ["opaque"]})
            projection = _abstract_response(response)
            evidence = {"record_id": hashlib.sha256(challenge_id.encode("utf-8")).hexdigest(), "role": role, "reset_id": reset["reset_id"], "method": method, "projection": projection}
            rows.append({
                "record_id": evidence["record_id"],
                "role": role,
                "method": method,
                "response_projection": projection,
                "fresh_reset": True,
                "reset_id": reset["reset_id"],
                "persistent_storage": False,
                "external_network": False,
                "typed_available": False,
                "candidate_present": role == "candidate",
                "reference_present": role == "reference",
                "negative_control": role == "negative",
                "replay": role == "replay",
                "evidence_sha256": _hash(evidence),
                "target": {"question": "ask_typed", "next_action": "ask_typed", "safe_to_send": False},
                "training_eligible": False,
            })
    return {
        "schema_version": "pg348-dynamic-shape-trace-v1",
        "status": "completed_dynamic_diagnostic_only",
        "counts": {"rows": len(rows), "records": len(registry.get("records") or []), "roles": len(ROLES), "get_rows": sum(row["method"] == "GET" for row in rows), "post_rows": sum(row["method"] == "POST" for row in rows), "fresh_reset_rows": sum(row["fresh_reset"] for row in rows), "typed_positive": 0, "training_eligible": 0},
        "runtime": {"module": "app/pg348_dynamic_runtime.py", "loopback_only": True, "persistent_storage": False, "external_network": False, "raw_response_stored": False, "raw_input_stored": False},
        "records": rows,
        "failures": ["typed_evaluator_not_attached", "candidate_reference_negative_effect_not_typed"],
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PG-348 dynamic abstract shape traces")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    trace = collect(registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": trace["status"], "counts": trace["counts"], "failures": trace["failures"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
