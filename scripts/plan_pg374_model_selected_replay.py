"""PG-374 staged-candidate model-slot → binder → replay contract.

PG-373 produced a remote candidate checkpoint and aggregate metrics, but did
not materialize decoder outputs for the WebGoat second implementation.  This
module therefore plans the next boundary without pretending that inference or
typed replay happened.  A future caller may pass a full 13-slot abstract
target sequence to :func:`select_staged_candidate`; that function delegates
allowlist checks to the PG-371/PG-350 contract and stops at ``ASK`` until
fresh candidate/reference/negative/replay evidence is supplied.

No Docker, GPU, browser, network, template expansion, payload, URL, response
body, or evaluator answer is read or generated here.  The persisted artifact
contains only bounded abstract slots, flags, hashes, and route/method shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg371_model_slot_binder_contract import (  # noqa: E402
    MODEL_SLOTS,
    REQUIRED_SLOTS,
    ROLES,
    parse_target_slots,
    select_and_bind_model_slots,
    sha256_json,
)
from scripts.plan_pg368_second_implementation import (  # noqa: E402
    IMAGE,
    ROUTES,
    SEEDS as PG368_SEEDS,
    route_ref_sha256,
)

SCHEMA_VERSION = "pg374-model-selected-replay-plan-v1"
PG373_REPORT_PATH = ROOT / "research" / "pg373_staged_pretrain_candidate_v1.json"
PG368_PLAN_PATH = ROOT / "research" / "pg368_second_implementation_plan_v1.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = {
    "url",
    "uri",
    "body",
    "response_body",
    "payload",
    "raw_payload",
    "raw_value",
    "wire",
    "evaluator_answer",
}


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scrub(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS or any(part in key_text for part in ("raw_", "response_body", "evaluator_")):
                raise ValueError(f"forbidden_key:{path}.{key}")
            _scrub(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scrub(item, f"{path}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        for fragment in ("http://", "https://", "<script", "document.cookie", "/webgoat"):
            if fragment in folded:
                raise ValueError(f"forbidden_text:{fragment}")


def _staged_candidate_attestation(report: Mapping[str, Any]) -> dict[str, Any]:
    candidates = list(report.get("candidates") or [])
    hashes: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        seed = str(candidate.get("seed", ""))
        checkpoint = candidate.get("checkpoint")
        digest = str(checkpoint.get("sha256", "")).casefold() if isinstance(checkpoint, Mapping) else ""
        if seed and _HEX64.fullmatch(digest):
            hashes[seed] = digest
    return {
        "schema_version": str(report.get("schema_version", "")),
        "report_file_sha256": _file_sha(PG373_REPORT_PATH),
        "report_declared_sha256": str(report.get("report_sha256", "")),
        "candidate_seed_count": len(candidates),
        "candidate_checkpoint_sha256": hashes,
        "output_materialized": False,
        "full_13_slot_output_materialized": False,
        "typed_live_replay_with_model_selected_wire": bool(dict(report.get("scientific_gate") or {}).get("typed_live_replay_with_model_selected_wire")),
        "promotion_closed": all(value is False for key, value in dict(report.get("promotion") or {}).items() if key.endswith("_allowed")),
    }


def select_staged_candidate(
    target_tokens: Sequence[str],
    *,
    expected_method: str,
    role: str,
) -> dict[str, Any]:
    """Select a PG-373 13-slot output and bind only through PG-371 gates."""

    parsed = parse_target_slots(target_tokens)
    if parsed is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "candidate_decode_incomplete",
            "model_selected": False,
            "typed_effect_confirmed": False,
            "wire_created": False,
            "safe_to_send": False,
            "target_contacted": False,
            "role": str(role),
            "reason": "missing_or_invalid_13_slot_sequence",
        }
    result = select_and_bind_model_slots(parsed, expected_method=expected_method, role=role)
    result = dict(result)
    result["target_contacted"] = False
    result["source"] = "pg373_staged_13_slot_abstract_output"
    result["typed_effect_confirmed"] = False
    result["wire_created"] = False
    result["safe_to_send"] = False
    _scrub(result)
    return result


def _plan_row(*, seed: int, route: Mapping[str, Any], role: str) -> dict[str, Any]:
    method = str(route["method"]).upper()
    return {
        "seed": int(seed),
        "route_ref_sha256": route_ref_sha256(route),
        "method": method,
        "response_shape_ref": str(route["response_shape"]),
        "role": str(role),
        "model_selected": False,
        "candidate_output_status": "not_materialized",
        "binding_status": "ASK_missing_staged_candidate_output",
        "safe_to_send": False,
        "typed_effect_confirmed": False,
        "wire_created": False,
        "target_contacted": False,
        "fresh_reset_required": True,
        "fresh_reset_observed": False,
        "candidate_reference_negative_replay_required": True,
        "candidate_reference_negative_replay_observed": False,
        "typed_evidence_sha256_required": True,
        "typed_evidence_sha256_observed": False,
        "context_firewall_closed": True,
    }


def build_pg374_plan(*, seeds: Sequence[int] = PG368_SEEDS) -> dict[str, Any]:
    pg373 = _load(PG373_REPORT_PATH)
    pg368 = _load(PG368_PLAN_PATH)
    if pg373.get("status") not in {"remote_candidate_only", "cpu_smoke_candidate_only", "plan_only"}:
        raise ValueError("PG-373 candidate report has an unexpected status")
    if pg368.get("status") != "planning_only":
        raise ValueError("PG-368 plan is not planning_only")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("PG-374 requires at least one seed")
    rows = [_plan_row(seed=seed, route=route, role=role) for seed in normalized_seeds for route in ROUTES for role in ROLES]
    staged = _staged_candidate_attestation(pg373)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "planning_only_blocked",
        "implementation": {
            "implementation_id": "pg374_webgoat_second_implementation",
            "image_digest": IMAGE.split("@sha256:", 1)[1],
            "pg368_plan_file_sha256": _file_sha(PG368_PLAN_PATH),
            "pg368_plan_declared_sha256": str(pg368.get("plan_sha256", "")),
            "independent_implementation_required": True,
        },
        "staged_candidate": staged,
        "rule_ir_schema": {
            "model_slots_pg370": list(MODEL_SLOTS),
            "binder_gate_slots_pg371": list(REQUIRED_SLOTS),
            "output_source": "pg373_staged_13_slot_abstract_output",
            "context_firewall_closed": True,
        },
        "execution": {
            "docker_started": False,
            "gpu_started": False,
            "network_contacted": False,
            "network_mode": "none_required_for_future_live_run",
            "loopback_only_required": True,
            "published_ports_allowed": False,
            "bind_or_volume_mounts_allowed": False,
        },
        "fresh_typed_replay_contract": {
            "candidate_reference_negative_replay_required": True,
            "fresh_reset_per_seed_route_role": True,
            "typed_evidence_sha256_required": True,
            "negative_violation_zero_required": True,
            "model_selected_separate_from_typed_effect": True,
            "wire_creation_separate_from_model_selected": True,
            "observed_in_this_plan": False,
        },
        "rows": rows,
        "counts": {
            "seeds": len(normalized_seeds),
            "routes": len(ROUTES),
            "episodes": len(normalized_seeds) * len(ROUTES),
            "roles": len(rows),
            "get_rows": sum(row["method"] == "GET" for row in rows),
            "post_rows": sum(row["method"] == "POST" for row in rows),
            "candidate_rows": sum(row["role"] == "candidate" for row in rows),
            "reference_rows": sum(row["role"] == "reference" for row in rows),
            "negative_rows": sum(row["role"] == "negative" for row in rows),
            "replay_rows": sum(row["role"] == "replay" for row in rows),
            "model_selected": 0,
            "typed_effect_confirmed": 0,
            "wire_created": 0,
            "target_contacted": 0,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "blocked_reasons": [
            "pg373_decoder_outputs_not_materialized_for_webgoat",
            "typed_live_replay_with_model_selected_wire=false",
            "fresh_evidence_sha256_unobserved",
            "candidate_reference_negative_replay_unobserved",
        ],
        "interpretation": (
            "PG-374 is a planning-only bridge. A future 13-slot model output may set model_selected=true, "
            "but the allowlisted binder remains ASK until fresh typed evidence exists; no wire or target contact is claimed."
        ),
    }
    _scrub(report)
    report["report_sha256"] = sha256_json(report)
    return report


def validate_pg374_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "planning_only_blocked":
        failures.append("schema_or_status")
    rows = list(report.get("rows") or [])
    if len(rows) != 24:
        failures.append("row_count")
    if {row.get("method") for row in rows} != {"GET", "POST"}:
        failures.append("get_post_pair")
    if {row.get("role") for row in rows} != set(ROLES):
        failures.append("role_set")
    for row in rows:
        if any(row.get(key) is not False for key in ("model_selected", "typed_effect_confirmed", "wire_created", "target_contacted")):
            failures.append("state_separation")
        if row.get("fresh_reset_required") is not True or row.get("typed_evidence_sha256_required") is not True:
            failures.append("fresh_evidence_contract")
        if row.get("context_firewall_closed") is not True:
            failures.append("context_firewall")
    staged = dict(report.get("staged_candidate") or {})
    if staged.get("output_materialized") is not False or staged.get("full_13_slot_output_materialized") is not False:
        failures.append("staged_output_unexpectedly_materialized")
    if staged.get("typed_live_replay_with_model_selected_wire") is not False:
        failures.append("typed_live_replay_unexpectedly_true")
    for key, value in dict(report.get("promotion") or {}).items():
        if key.endswith("_allowed") and value is not False:
            failures.append("promotion_open")
    return {"status": "passed" if not failures else "blocked", "failures": failures, "row_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "research" / "pg374_model_selected_replay_plan_v1.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_pg374_plan()
    validation = validate_pg374_plan(report)
    _scrub(report)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"report": report, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
