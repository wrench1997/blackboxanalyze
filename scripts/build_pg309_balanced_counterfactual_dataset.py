"""Add balanced abstract counterfactuals for the PG-308 failure modes.

PG-309 does not import new routes or payloads.  It adds paired observable
contexts for the same bounded surface slots: complete, one missing
observation, candidate failure repair, and visible surface-slot mismatch.  The
new rows are training-only counterfactuals; PG-308's source-held-out and
hard-negative lanes remain untouched for evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import canonical_assembly_context  # noqa: E402
from app.pg302_symbolic_assembly import audit_symbolic_records, symbolic_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg308_multisource_slot_dataset_v1.json"
SOURCE_AUDIT = RESEARCH / "pg308_multisource_slot_dataset_audit_v1.json"
OUT = RESEARCH / "pg309_balanced_counterfactual_dataset_v1.json"
AUDIT = RESEARCH / "pg309_balanced_counterfactual_dataset_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _context(
    *,
    method: str,
    field_role: str,
    encoding: str,
    typed_available: str,
    feedback_state: str,
    replay_ready: str,
    evidence_present: str,
    negative_control: str,
    fresh_reset: str,
    history_action: str,
    failure_class: str,
    step_budget: str = "present",
) -> list[str]:
    return canonical_assembly_context(
        [
            "[BOS]",
            f"typed_available={typed_available}",
            f"feedback_state={feedback_state}",
            f"replay_ready={replay_ready}",
            f"evidence_present={evidence_present}",
            f"negative_control={negative_control}",
            f"fresh_reset={fresh_reset}",
            f"surface_method={method}",
            f"surface_field_role={field_role}",
            f"surface_encoding={encoding}",
            f"history_action={history_action}",
            f"failure_class={failure_class}",
            f"step_budget={step_budget}",
            "[EOS]",
        ]
    )


def _row(record_id: str, context: list[str], evidence: str) -> dict[str, Any]:
    target = symbolic_target_for_context(context)
    values = {str(token).split("=", 1)[0]: str(token).split("=", 1)[1] for token in target if "=" in str(token)}
    row = {
        "schema_version": "pg309-balanced-counterfactual-record-v1",
        "record_id": record_id,
        "source": "pg309_balanced_abstract_counterfactual",
        "split": "train",
        "training_eligible": True,
        "context_tokens": context,
        "target_tokens": target,
        "question": values.get("question", "none"),
        "safe_to_send": values.get("safe_to_send") == "1",
        "hard_negative": False,
        "provenance": "abstract_counterfactual_from_pg308_audited_slots",
        "source_dataset_sha256": SOURCE.read_bytes(),
        "source_audit_sha256": "",
        "source_evidence_sha256": evidence,
        "source_authorized_loopback": True,
        "oracle_target_off_input": True,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "memory_promotion_allowed": False,
        "record_sha256": "",
    }
    # Keep provenance scalar and deterministic rather than serializing bytes.
    row["source_dataset_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    row["source_audit_sha256"] = hashlib.sha256(SOURCE_AUDIT.read_bytes()).hexdigest()
    row["record_sha256"] = _digest(row)
    return row


def main() -> int:
    source = _load(SOURCE)
    audit_source = _load(SOURCE_AUDIT)
    if audit_source.get("status") != "passed":
        raise RuntimeError("PG-309 requires a passed PG-308 dataset audit")
    records = [copy.deepcopy(row) for row in source.get("records", [])]
    surfaces = (
        ("GET", "query_param", "url_percent"),
        ("GET", "query_param", "identity"),
        ("POST", "form_field", "form_urlencoded"),
        ("POST", "form_field", "json_string"),
    )
    missing_keys = (
        "typed_available",
        "replay_ready",
        "evidence_present",
        "feedback_state",
        "negative_control",
        "fresh_reset",
    )
    generated: list[dict[str, Any]] = []
    for surface_index, (method, field_role, encoding) in enumerate(surfaces):
        # A complete positive pair for each surface representation.
        generated.append(
            _row(
                f"pg309:complete:{surface_index}",
                _context(
                    method=method,
                    field_role=field_role,
                    encoding=encoding,
                    typed_available="1",
                    feedback_state="negative_control_clear",
                    replay_ready="1",
                    evidence_present="1",
                    negative_control="1",
                    fresh_reset="1",
                    history_action="none",
                    failure_class="none",
                ),
                f"pg309-complete-{surface_index}",
            )
        )
        # Every missing slot is paired with the same surface and all other
        # observations complete.  The first missing slot wins by contract.
        for missing_index, missing_key in enumerate(missing_keys):
            values = {
                "typed_available": "1",
                "feedback_state": "negative_control_clear",
                "replay_ready": "1",
                "evidence_present": "1",
                "negative_control": "1",
                "fresh_reset": "1",
            }
            values[missing_key] = "unknown"
            generated.append(
                _row(
                    f"pg309:missing:{surface_index}:{missing_index}",
                    _context(
                        method=method,
                        field_role=field_role,
                        encoding=encoding,
                        history_action="observe",
                        failure_class="none",
                        **values,
                    ),
                    f"pg309-missing-{surface_index}-{missing_index}",
                )
            )
        # A failed candidate has enough observation to choose a repair, but is
        # never safe to send as-is.
        for failure_index in range(2):
            generated.append(
                _row(
                    f"pg309:repair:{surface_index}:{failure_index}",
                    _context(
                        method=method,
                        field_role=field_role,
                        encoding=encoding,
                        typed_available="1",
                        feedback_state="candidate_failed",
                        replay_ready="1",
                        evidence_present="1",
                        negative_control="1",
                        fresh_reset="1",
                        history_action="candidate_failed",
                        failure_class="candidate_failed",
                    ),
                    f"pg309-repair-{surface_index}-{failure_index}",
                )
            )
        # Keep mismatch rows in training too; they teach the decoder that a
        # complete-looking observation can still require repair.
        mismatch_field = "form_field" if field_role == "query_param" else "query_param"
        mismatch_encoding = "form_urlencoded" if encoding in {"url_percent", "identity"} else "url_percent"
        generated.append(
            _row(
                f"pg309:mismatch:{surface_index}",
                _context(
                    method=method,
                    field_role=mismatch_field,
                    encoding=mismatch_encoding,
                    typed_available="1",
                    feedback_state="negative_control_clear",
                    replay_ready="1",
                    evidence_present="1",
                    negative_control="1",
                    fresh_reset="1",
                    history_action="surface_mismatch",
                    failure_class="surface_slot_mismatch",
                ),
                f"pg309-mismatch-{surface_index}",
            )
        )
    records.extend(generated)
    base_audit = audit_symbolic_records(records)
    counts = {
        "total": len(records),
        "train": sum(int(row.get("split") == "train") for row in records),
        "training_eligible": sum(int(row.get("training_eligible")) for row in records),
        "generated_counterfactual": len(generated),
        "generated_complete": sum(int(row.get("record_id", "").startswith("pg309:complete")) for row in generated),
        "generated_missing": sum(int(row.get("record_id", "").startswith("pg309:missing")) for row in generated),
        "generated_repair": sum(int(row.get("record_id", "").startswith("pg309:repair")) for row in generated),
        "generated_mismatch": sum(int(row.get("record_id", "").startswith("pg309:mismatch")) for row in generated),
    }
    checks = dict(base_audit.get("checks") or {})
    checks.update(
        {
            "source_audit_pass": True,
            "counterfactuals_present": counts["generated_counterfactual"] > 0,
            "complete_missing_pairs": counts["generated_complete"] == len(surfaces) and counts["generated_missing"] == len(surfaces) * len(missing_keys),
            "repair_pairs_present": counts["generated_repair"] == len(surfaces) * 2,
            "mismatch_pairs_present": counts["generated_mismatch"] == len(surfaces),
            "payload_strings_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
            "promotion_blocked": True,
        }
    )
    dataset = {
        "schema_version": "pg309-balanced-counterfactual-dataset-v1",
        "purpose": "balanced missing/complete/failure/mismatch counterfactuals for PG-308 symbolic slot-copy",
        "source": {"dataset": str(SOURCE.relative_to(ROOT)), "dataset_sha256": source.get("dataset_sha256"), "audit": str(SOURCE_AUDIT.relative_to(ROOT)), "audit_sha256": audit_source.get("audit_sha256")},
        "records": records,
        "counts": counts,
        "contract": {"causal_next_token_targets": True, "symbolic_slot_references": True, "paired_missing_complete": True, "failure_repair_pairs": True, "slot_mismatch_pairs": True, "route_family_not_in_context": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "oracle_target_off_input": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {"schema_version": "pg309-balanced-counterfactual-dataset-audit-v1", "dataset": str(OUT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "checks": checks, "base_audit": base_audit, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset": str(OUT.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
