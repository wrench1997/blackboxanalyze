"""PG-389 local typed replay for abstract JS decode/filter chains.

The reviewed PG-385 loopback fixture is used as a bounded evaluator for three
chain projections.  The fixture-bound marker values are created and consumed
inside :func:`run_demo`; this report keeps only abstract projections and
evidence hashes.  It is diagnostic evidence, not a payload catalogue or a
claim about arbitrary web applications.
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

from app.pg389_js_chain_projection import project_chain_case  # noqa: E402
from scripts.run_pg385_filter_repair_demo import run_demo  # noqa: E402


SCHEMA_VERSION = "pg389-js-chain-local-replay-v1"
REPLAY_CASES = ("query_decode_then_filter", "query_filter_then_decode", "double_decode_order")
ROLES = ("candidate", "reference", "negative", "replay")
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _scrub(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).casefold()
    for marker in ("http://", "https://", "wire=", "payload=", "response_body=", "raw_value=", "pg385_cand", "pg385_ref", "pg385_neg", "pg385_replay"):
        if marker in text:
            raise ValueError("raw_or_wire_material_in_pg389_report")


def run_local_replay() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for case_ref in REPLAY_CASES:
        projection = project_chain_case({"case_ref": case_ref}, variant="chain_order_a")
        source_report, _ephemeral_wires = run_demo(show_wire=False)
        source_report_hash = str(source_report["report_sha256"])
        steps = source_report.get("steps", [])
        if len(steps) < 3:
            raise ValueError("pg385_replay_contract_incomplete")
        baseline_projection = steps[0].get("response")
        role_projections = steps[2].get("roles")
        if not isinstance(baseline_projection, dict) or not isinstance(role_projections, dict):
            raise ValueError("pg385_replay_projection_missing")
        action_changed = bool(steps[1].get("action_changed"))
        case_rows: list[dict[str, Any]] = []
        for role in ROLES:
            typed = role_projections.get(role)
            if not isinstance(typed, dict):
                raise ValueError("pg385_role_projection_missing")
            row_core = {
                "record_ref_sha256": _sha({"case_ref": case_ref, "role": role, "source_report": source_report_hash}),
                "case_ref": case_ref,
                "role": role,
                "context_tokens": projection["context_tokens"],
                "target_tokens": projection["target_tokens"],
                "decode_filter_context": projection["decode_filter_context"],
                "baseline_projection": {
                    "filter_state": baseline_projection.get("filter_state"),
                    "failure_shape": baseline_projection.get("failure_shape"),
                    "typed_effect_confirmed": bool(baseline_projection.get("typed_effect_confirmed")),
                },
                "typed_projection": {
                    "filter_state": typed.get("filter_state"),
                    "failure_shape": typed.get("failure_shape"),
                    "effect_class": typed.get("effect_class"),
                    "typed_effect_confirmed": bool(typed.get("typed_effect_confirmed")),
                    "encoding_acceptance": typed.get("encoding_acceptance"),
                    "response_shape": typed.get("response_shape"),
                    "response_evidence_sha256": typed.get("response_evidence_sha256"),
                },
                "fresh_reset": True,
                "failure_observed": baseline_projection.get("filter_state") == "filtered",
                "action_changed": action_changed,
                "local_fixture_contacted": True,
                "ephemeral_wire_used": True,
                "raw_wire_stored": False,
                "raw_value_stored": False,
                "typed_evaluator_observed": bool(typed.get("typed_effect_confirmed")) or role == "negative",
                "training_eligible": False,
                "promotion": dict(PROMOTION),
            }
            row = dict(row_core)
            row["evidence_sha256"] = _sha({"case_ref": case_ref, "role": role, "source_report": source_report_hash, "projection": row_core["typed_projection"]})
            row["row_sha256"] = _sha(row)
            case_rows.append(row)
            rows.append(row)
        case_reports.append({
            "case_ref": case_ref,
            "source_report_sha256": source_report_hash,
            "rows": len(case_rows),
            "baseline_filtered": all(item["baseline_projection"]["filter_state"] == "filtered" for item in case_rows),
            "candidate_typed": bool(case_rows[0]["typed_projection"]["typed_effect_confirmed"]),
            "reference_typed": bool(case_rows[1]["typed_projection"]["typed_effect_confirmed"]),
            "negative_clean": not bool(case_rows[2]["typed_projection"]["typed_effect_confirmed"]),
            "replay_typed": bool(case_rows[3]["typed_projection"]["typed_effect_confirmed"]),
            "action_changed": all(item["action_changed"] for item in case_rows),
        })
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_local_fixture_typed_diagnostic",
        "scope": "PG385_loopback_fixture_mapped_to_PG389_abstract_chain",
        "interpretation": "typed fixture-shape evidence only; not a JS implementation holdout, arbitrary target test, vulnerability claim or payload catalogue",
        "counts": {
            "cases": len(REPLAY_CASES),
            "roles": len(ROLES),
            "rows": len(rows),
            "fresh_resets": len(rows),
            "typed_effect": sum(int(row["typed_projection"]["typed_effect_confirmed"]) for row in rows),
            "baseline_filtered": sum(int(row["baseline_projection"]["filter_state"] == "filtered") for row in rows),
            "failure_action_changed": sum(int(row["action_changed"]) for row in rows),
            "negative_violation": sum(int(row["role"] == "negative" and row["typed_projection"]["typed_effect_confirmed"]) for row in rows),
            "evidence_rows": sum(int(bool(row["evidence_sha256"])) for row in rows),
        },
        "case_reports": case_reports,
        "rows": rows,
        "execution": {
            "local_fixture_contacted": True,
            "target_contacted": False,
            "external_network": False,
            "ephemeral_wire_used": True,
            "raw_wire_stored": False,
            "raw_value_stored": False,
            "docker_started": False,
            "gpu_touched": False,
            "training_started": False,
        },
        "model_boundary": {
            "context_abstract_only": True,
            "model_raw_value": False,
            "model_wire": False,
            "evaluator_last_hop_binding": True,
            "evaluator_answers_in_context": False,
        },
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
    }
    _scrub(report)
    report["report_sha256"] = _sha(report)
    return report


def write_report(path: str | Path = ROOT / "research/pg389_js_chain_local_replay_v1.json") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run_local_replay()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "research/pg389_js_chain_local_replay_v1.json"))
    args = parser.parse_args()
    output = write_report(args.output)
    report = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": report["status"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
