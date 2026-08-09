"""Build PG-306 process-supervised data from the PG-305 live evaluator.

The new rows add real GET/POST surface observations and the missing-slot
counterfactuals that the frozen model failed.  They contain no executable
value, route, response body, oracle verdict or wire in the model context.
PG-305 remains an evaluator-only source; this builder marks rows as candidate
training data for the next experiment, but the promotion gate stays closed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import (  # noqa: E402
    FORBIDDEN_KEYS,
    OBSERVATION_KEYS,
    SURFACE_KEYS,
    assembly_target_for_context,
    canonical_assembly_context,
    sha256_json,
    target_map,
)
from app.pg305_live_evaluator import MISSING_ORDER, context_tokens, missing_question_contexts  # noqa: E402


RESEARCH = ROOT / "research"
PG301 = RESEARCH / "pg301_payload_assembly_dataset_v1.json"
PG305 = RESEARCH / "pg305_live_loopback_replay_training_dataset_v1.json"
PG305_CATALOG = RESEARCH / "pg305_live_loopback_replay_human_catalog_v1.json"
OUT = RESEARCH / "pg306_real_process_dataset_v1.json"
AUDIT = RESEARCH / "pg306_real_process_dataset_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record(record_id: str, context: list[str], target: list[str], split: str, *, source: str, training_eligible: bool, hard_negative: bool, counterfactual_group: str) -> dict[str, Any]:
    values = target_map(target)
    row = {
        "schema_version": "pg306-real-process-record-v1",
        "record_id": str(record_id),
        "source": str(source),
        "split": str(split),
        "training_eligible": bool(training_eligible),
        "context_tokens": list(context),
        "target_tokens": list(target),
        "question": values.get("question", "none"),
        "safe_to_send": values.get("safe_to_send") == "1",
        "hard_negative": bool(hard_negative),
        "counterfactual_group": str(counterfactual_group),
        "oracle_target_off_input": True,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "memory_promotion_allowed": False,
    }
    row["record_sha256"] = sha256_json(row)
    return row


def _audit_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    context = [str(token) for token in row.get("context_tokens") or []]
    target = [str(token) for token in row.get("target_tokens") or []]
    failures: list[str] = []
    keys = {token.split("=", 1)[0] for token in context + target if "=" in token}
    if not all(key in keys for key in OBSERVATION_KEYS + SURFACE_KEYS):
        failures.append("observable_slots_incomplete")
    if any(key in keys for key in FORBIDDEN_KEYS):
        failures.append("forbidden_key_present")
    joined = " ".join(context + target).casefold()
    if any(fragment in joined for fragment in ("http://", "https://", "<loopback_origin>", "response_body", "raw_payload")):
        failures.append("raw_or_wire_material_present")
    if row.get("raw_payload_stored") or row.get("raw_response_body_stored"):
        failures.append("raw_storage_flag")
    if len(target) != 12:
        failures.append("target_shape")
    return not failures, failures


def main() -> int:
    pg301 = _load(PG301)
    pg305 = _load(PG305)
    catalog = _load(PG305_CATALOG)
    route_by_record = {str(row.get("record_id")): str(row.get("route", {}).get("id", "unknown")) for row in catalog.get("entries", [])}
    method_by_record = {str(row.get("record_id")): str(row.get("route", {}).get("method", "GET")).upper() for row in catalog.get("entries", [])}

    records: list[dict[str, Any]] = []
    # Keep the audited PG-301 process split as a controlled synthetic baseline.
    for original in pg301.get("records", []):
        split = str(original.get("split", "train"))
        records.append(_record(
            str(original.get("record_id", "pg301")),
            list(original.get("context_tokens") or []),
            list(original.get("target_tokens") or []),
            split,
            source="pg301_abstract_baseline",
            training_eligible=split == "train",
            hard_negative=split == "hard_negative_eval",
            counterfactual_group=str(original.get("counterfactual_group", "pg301")),
        ))

    real_route_ids = sorted(route_by_record.values())
    real_train_routes = set(real_route_ids[:-1])
    real_holdout_routes = set(real_route_ids[-1:])
    real_rows = {str(row.get("record_id")): row for row in pg305.get("records", [])}
    for record_id, original in sorted(real_rows.items()):
        route_id = route_by_record.get(record_id, "unknown")
        method = method_by_record.get(record_id, "GET")
        split = "real_live_holdout" if route_id in real_holdout_routes else "train"
        records.append(_record(
            record_id,
            list(original.get("context_tokens") or []),
            list(original.get("target_tokens") or []),
            split,
            source="pg305_real_live_evaluator",
            training_eligible=split == "train" and bool(original.get("typed_effect_confirmed")),
            hard_negative=False,
            counterfactual_group=f"pg305-real:{route_id}",
        ))
        # Same live surface, but hide exactly one critical observation at a
        # time.  These are process-supervision rows, not final-only labels.
        for missing in missing_question_contexts(method):
            missing_slot = str(missing["missing_slot"])
            counter_split = "real_live_holdout" if route_id in real_holdout_routes else "train"
            records.append(_record(
                f"{record_id}:missing:{missing_slot}",
                list(missing["context_tokens"]),
                list(missing["target_tokens"]),
                counter_split,
                source="pg305_real_missing_counterfactual",
                training_eligible=counter_split == "train",
                hard_negative=False,
                counterfactual_group=f"pg305-real:{route_id}:missing",
            ))
        failure_values = {
            "typed_available": "1",
            "feedback_state": "candidate_failed",
            "replay_ready": "1",
            "evidence_present": "1",
            "negative_control": "1",
            "fresh_reset": "1",
        }
        failure_context = context_tokens(method, **failure_values, history_action="candidate_failed", failure_class="candidate_failed")
        records.append(_record(
            f"{record_id}:failure",
            failure_context,
            assembly_target_for_context(failure_context),
            "train",
            source="pg305_real_failure_repair_train",
            training_eligible=True,
            hard_negative=False,
            counterfactual_group=f"pg305-real:{route_id}:failure",
        ))

    good_indices: list[int] = []
    failures: dict[str, int] = {}
    for index, row in enumerate(records):
        good, reasons = _audit_row(row)
        if good:
            good_indices.append(index)
        for reason in reasons:
            failures[reason] = failures.get(reason, 0) + 1
    splits = {str(row.get("split")) for row in records}
    counts = {
        "total": len(records),
        "train": sum(int(row.get("split") == "train") for row in records),
        "implementation_holdout": sum(int(row.get("split") == "implementation_holdout") for row in records),
        "real_live_holdout": sum(int(row.get("split") == "real_live_holdout") for row in records),
        "hard_negative_eval": sum(int(row.get("split") == "hard_negative_eval") for row in records),
        "real_process_rows": sum(int(row.get("source") == "pg305_real_live_evaluator") for row in records),
        "missing_counterfactual_rows": sum(int("missing_counterfactual" in str(row.get("source"))) for row in records),
        "failure_counterfactual_rows": sum(int("failure" in str(row.get("source"))) for row in records),
        "training_eligible": sum(int(bool(row.get("training_eligible"))) for row in records),
    }
    dataset = {
        "schema_version": "pg306-real-process-dataset-v1",
        "purpose": "real evaluator grounded missing-observation question and abstract assembly next-token training",
        "sources": {"pg301_dataset": str(PG301.relative_to(ROOT)), "pg305_training_projection": str(PG305.relative_to(ROOT)), "pg305_human_catalog": str(PG305_CATALOG.relative_to(ROOT)), "pg305_training_projection_sha256": pg305.get("dataset_sha256")},
        "records": records,
        "counts": counts,
        "splits": sorted(splits),
        "contract": {"causal_next_token_targets": True, "process_question_supervision": True, "real_get_post_rows": counts["real_process_rows"] > 0, "missing_counterfactuals": counts["missing_counterfactual_rows"] > 0, "failure_repair_counterfactuals": counts["failure_counterfactual_rows"] > 0, "route_and_family_not_in_context": True, "oracle_target_off_input": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    audit = {
        "schema_version": "pg306-real-process-dataset-audit-v1",
        "dataset": str(OUT.relative_to(ROOT)),
        "dataset_sha256": dataset["dataset_sha256"],
        "source_live_report": "research/pg305_live_loopback_replay_report_v1.json",
        "source_live_report_sha256": _load(RESEARCH / "pg305_live_loopback_replay_report_v1.json").get("report_sha256"),
        "counts": counts,
        "checks": {"records_present": bool(records), "all_rows_well_formed": len(good_indices) == len(records), "train_present": "train" in splits, "holdout_present": bool({"implementation_holdout", "real_live_holdout"}.intersection(splits)), "hard_negative_present": "hard_negative_eval" in splits, "get_post_real_present": any(method_by_record.values()) and {"GET", "POST"}.issubset(set(method_by_record.values())), "missing_question_present": counts["missing_counterfactual_rows"] > 0, "failure_repair_present": counts["failure_counterfactual_rows"] > 0, "raw_material_absent": not failures, "promotion_blocked": True},
        "failures": failures,
        "status": "passed" if len(good_indices) == len(records) and failures == {} else "failed",
        "audit_sha256": "",
    }
    audit["audit_sha256"] = _digest(audit)
    _write(OUT, dataset)
    _write(AUDIT, audit)
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset": str(OUT.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
