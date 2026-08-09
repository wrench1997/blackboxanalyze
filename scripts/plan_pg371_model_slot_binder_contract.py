"""PG-371 planning contract for model-slot selection on WebGoat.

This is a read-only adapter contract.  It reuses the PG-368 WebGoat plan and
the PG-350 allowlist at the abstract Rule-IR boundary, but it never starts a
container, opens a socket, expands an evaluator template, or writes a wire.
The report separates ``model_selected`` from ``typed_effect_confirmed`` so a
model decision cannot be mistaken for a fresh evaluator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg371_model_slot_binder_contract import (
    MODEL_SLOTS,
    REQUIRED_SLOTS,
    ROLES,
    SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION,
    planning_binding_row,
    select_and_bind_model_slots,
    sha256_json,
)
from scripts.plan_pg368_second_implementation import (
    IMAGE,
    PG368_IMPLEMENTATION_ID,
    ROUTES,
    SEEDS,
    route_ref_sha256,
)

SCHEMA_VERSION = "pg371-model-slot-binder-plan-v1"
PLAN_PATH = ROOT / "research" / "pg368_second_implementation_plan_v1.json"
PG367_DATASET_PATH = ROOT / "research" / "pg367_waf_staircase_dataset_v2.json"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _scrub(value: Any, path: str = "$") -> None:
    forbidden = {"url", "uri", "body", "response_body", "payload", "raw_payload", "raw_value", "wire", "evaluator_answer"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in forbidden or any(part in str(key).casefold() for part in ("raw_", "response_body", "evaluator_")):
                raise ValueError(f"raw_or_evaluator_key:{path}.{key}")
            _scrub(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _scrub(item, f"{path}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        for fragment in ("http://", "https://", "<script", "document.cookie"):
            if fragment in folded:
                raise ValueError(f"raw_or_evaluator_text:{fragment}")


def _reference_slot_digest() -> str:
    """Hash only the reference target-slot vocabulary, not source rows."""

    dataset = _load(PG367_DATASET_PATH)
    vocabulary = dataset.get("vocabulary")
    target = list(vocabulary.get("target_tokens") or []) if isinstance(vocabulary, Mapping) else []
    names = sorted(str(token) for token in target if any(str(token).startswith(slot + "=") for slot in REQUIRED_SLOTS))
    return sha256_json(names)


def build_pg371_model_slot_binder_plan(*, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    plan = _load(PLAN_PATH)
    if plan.get("status") != "planning_only":
        raise ValueError("PG-368 plan is not planning_only")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("PG-371 requires at least one seed")
    rows: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        for route in ROUTES:
            for role in ROLES:
                rows.append(
                    planning_binding_row(
                        seed=seed,
                        route_ref_sha256=route_ref_sha256(route),
                        method=route["method"],
                        response_shape=route["response_shape"],
                        role=role,
                    )
                )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "planning_only_blocked",
        "implementation": {
            "implementation_id": "pg371_webgoat_model_slot_binder",
            "independent_from": PG368_IMPLEMENTATION_ID,
            "image_digest": IMAGE.split("@sha256:", 1)[1],
            "plan_sha256": str(plan.get("plan_sha256", "")),
            "reference_slot_vocabulary_sha256": _reference_slot_digest(),
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
        "required_slot_contract": {
            "model_slots_pg370": list(MODEL_SLOTS),
            "binder_gate_slots": list(REQUIRED_SLOTS),
            "allowlisted_binder": "app.pg350_runtime_payload_binder._validate_model_slots",
            "model_selected_distinct_from_evaluator": True,
            "wire_creation_in_this_contract": False,
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
            "model_selected": sum(bool(row["model_selected"]) for row in rows),
            "typed_effect_confirmed": sum(bool(row["typed_effect_confirmed"]) for row in rows),
            "wire_created": sum(bool(row["wire_created"]) for row in rows),
            "target_contacted": sum(bool(row["target_contacted"]) for row in rows),
        },
        "evidence_contract": {
            "candidate_reference_negative_replay_required": True,
            "fresh_reset_required": True,
            "typed_evidence_sha256_required": True,
            "context_firewall_required": True,
            "observed_in_this_plan": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": (
            "PG-371 preserves the model-selected state separately from typed evaluator state. "
            "All current rows are planning ASK; no model output, fresh replay, evidence digest, or wire is claimed."
        ),
    }
    _scrub(report)
    report["report_sha256"] = sha256_json(report)
    return report


def validate_pg371_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION or report.get("status") != "planning_only_blocked":
        failures.append("schema_or_status")
    rows = list(report.get("rows") or [])
    if len(rows) != 24:
        failures.append("role_row_count")
    if {row.get("method") for row in rows} != {"GET", "POST"}:
        failures.append("get_post_pair")
    if {row.get("role") for row in rows} != set(ROLES):
        failures.append("role_set")
    for row in rows:
        if row.get("model_selected") is not False or row.get("typed_effect_confirmed") is not False or row.get("wire_created") is not False:
            failures.append("state_separation")
        if row.get("fresh_reset_required") is not True or row.get("typed_evidence_sha256_required") is not True:
            failures.append("fresh_or_evidence_contract")
        if row.get("context_firewall_closed") is not True or row.get("target_contacted") is not False:
            failures.append("context_or_target_contract")
    for key, value in dict(report.get("promotion") or {}).items():
        if key.endswith("_allowed") and value is not False:
            failures.append("promotion_open")
    return {"status": "passed" if not failures else "blocked", "failures": failures, "row_count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "research" / "pg371_model_slot_binder_plan_v1.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_pg371_model_slot_binder_plan()
    validation = validate_pg371_plan(report)
    _scrub(report)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"report": report, "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
