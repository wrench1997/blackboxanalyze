"""Build PG-322 cross-implementation/third-surface Rule-IR records.

The VulnerableApp route names and literal probes stay outside model-visible
tokens.  The decoder sees only bounded transport/field/encoding observations,
history actions, failure state, and the six identifiability observations.  A
record with missing observations is an ASK example, never a guessed answer.
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

from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context, target_map  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
OUTPUT = RESEARCH / "pg322_cross_impl_decoy_dataset_v1.json"
AUDIT = RESEARCH / "pg322_cross_impl_decoy_dataset_audit_v1.json"

SURFACES = (
    {
        "surface_id": "vapp_html_query",
        "source_id": "pg322_vulnerableapp_java_spring",
        "implementation": "sasanlabs_vulnerableapp_java_spring",
        "method": "GET",
        "field_role": "query_param",
        "encoding": "url_percent",
        "typed_available": "1",
        "expected_lane": "positive",
    },
    {
        "surface_id": "vapp_img_query",
        "source_id": "pg322_vulnerableapp_java_spring",
        "implementation": "sasanlabs_vulnerableapp_java_spring",
        "method": "GET",
        "field_role": "query_param",
        "encoding": "url_percent",
        "typed_available": "1",
        "expected_lane": "positive",
    },
    {
        "surface_id": "vapp_post_405",
        "source_id": "pg322_vulnerableapp_java_spring",
        "implementation": "sasanlabs_vulnerableapp_java_spring",
        "method": "POST",
        "field_role": "form_field",
        "encoding": "form_urlencoded",
        "typed_available": "0",
        "expected_lane": "unsupported",
    },
    {
        "surface_id": "blind_path_decoy",
        "source_id": "pg322_blind_third_surface",
        "implementation": "abstract_path_decoy",
        "method": "GET",
        "field_role": "path_segment",
        "encoding": "url_percent",
        "typed_available": "unknown",
        "expected_lane": "decoy",
    },
    {
        "surface_id": "blind_header_decoy",
        "source_id": "pg322_blind_third_surface",
        "implementation": "abstract_header_decoy",
        "method": "GET",
        "field_role": "header_value",
        "encoding": "identity",
        "typed_available": "unknown",
        "expected_lane": "decoy",
    },
)

ROLE_ACTIONS = ("candidate_request", "candidate_probe", "reference_request", "reference_probe", "negative_control", "negative_probe")
FAILURE_ACTIONS = ("candidate_failed", "repair_requested")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _base_context(surface: dict[str, str], action: str, *, failure: str = "none", missing: set[str] | None = None) -> list[str]:
    missing = missing or set()
    values: dict[str, str] = {
        "typed_available": str(surface["typed_available"]),
        "feedback_state": "negative_control_clear",
        "replay_ready": "1",
        "evidence_present": "1",
        "negative_control": "1",
        "fresh_reset": "1",
        "surface_method": str(surface["method"]),
        "surface_field_role": str(surface["field_role"]),
        "surface_encoding": str(surface["encoding"]),
        "history_action": action,
        "failure_class": failure,
        "step_budget": "present",
    }
    for key in missing:
        values[key] = "unknown"
    return canonical_assembly_context([f"{key}={value}" for key, value in values.items()])


def _record(surface: dict[str, str], action: str, *, split: str, eligible: bool, record_id: str, failure: str = "none", missing: set[str] | None = None, hard_negative: bool = False) -> dict[str, Any]:
    context = _base_context(surface, action, failure=failure, missing=missing)
    target = probe_target_for_context(context)
    target_values = target_map(target)
    record = {
        "schema_version": "pg322-cross-impl-decoy-record-v1",
        "record_id": record_id,
        "split": split,
        "training_eligible": bool(eligible),
        "source_meta": {
            "source_id": surface["source_id"],
            "implementation": surface["implementation"],
            "surface_id": surface["surface_id"],
            "expected_lane": surface["expected_lane"],
        },
        "context_tokens": context,
        "target_tokens": target,
        "expected_variant": str(target_values.get("probe_variant_ref", "none")),
        "expected_safe": str(target_values.get("safe_to_send", "0")) == "1",
        "expected_question": str(target_values.get("question", "none")),
        "hard_negative": bool(hard_negative),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_target_off_input": True,
    }
    record["record_sha256"] = _digest(record)
    return record


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index = 0
    # Only the first VulnerableApp surface is eligible for training.  The
    # second implementation surface, POST lane, and third-surface decoys stay
    # blind holdouts so route memorization cannot pass the audit.
    for surface in SURFACES:
        is_train_surface = surface["surface_id"] == "vapp_html_query"
        complete_split = "train" if is_train_surface else "implementation_holdout" if surface["surface_id"] == "vapp_img_query" else "third_surface_holdout"
        for action in ROLE_ACTIONS:
            records.append(_record(surface, action, split=complete_split, eligible=is_train_surface, record_id=f"pg322:complete:{index}"))
            index += 1
        for action in ("candidate_failed", "repair_requested"):
            records.append(_record(surface, action, split=complete_split, eligible=is_train_surface, record_id=f"pg322:failure:{index}", failure="effect_not_confirmed"))
            index += 1
        # All 15 double-missing combinations are preserved for each surface;
        # holdout surfaces are never used as training targets.
        for left_index, left in enumerate(OBSERVATION_KEYS):
            for right in OBSERVATION_KEYS[left_index + 1 :]:
                missing = {left, right}
                split = "train" if is_train_surface else "ask_holdout"
                eligible = bool(is_train_surface)
                records.append(_record(surface, "candidate_request", split=split, eligible=eligible, record_id=f"pg322:missing:{index}", missing=missing, hard_negative=not is_train_surface))
                index += 1
    # Explicit hard-negative complete rows prevent the third surface from
    # being accepted merely because the model predicts a valid-looking shape.
    for surface in SURFACES[2:]:
        for action in ("candidate_request", "reference_request", "negative_control"):
            records.append(_record(surface, action, split="hard_negative_eval", eligible=False, record_id=f"pg322:hard:{index}", hard_negative=True))
            index += 1

    dataset = {
        "schema_version": "pg322-cross-impl-decoy-dataset-v1",
        "status": "completed_pg322_dataset_build",
        "sources": {
            "pg321_role_dataset": "research/pg321_variant_role_lattice_dataset_v1.json",
            "vulnerableapp_image": "sasanlabs/owasp-vulnerableapp@sha256:7bc084dac341f089c6e788d2369a27f599c902d742c5e113d7bb50661cd92406",
            "authorization": "workspace_local_only",
            "raw_payloads_in_context": False,
            "raw_response_bodies_in_context": False,
        },
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row["split"] == "train" for row in records),
            "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records),
            "third_surface_holdout": sum(row["split"] == "third_surface_holdout" for row in records),
            "ask_holdout": sum(row["split"] == "ask_holdout" for row in records),
            "hard_negative_eval": sum(row["split"] == "hard_negative_eval" for row in records),
            "ask_rows": sum(row["expected_question"] != "none" for row in records),
            "failure_rows": sum(any(token.startswith("history_action=") and token.split("=", 1)[1] in FAILURE_ACTIONS for token in row["context_tokens"]) for row in records),
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    forbidden = {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss", "xxe"}
    bad_forbidden: list[int] = []
    bad_shape: list[int] = []
    for row_index, row in enumerate(records):
        keys = {str(token).split("=", 1)[0] for token in row["context_tokens"] + row["target_tokens"] if "=" in str(token)}
        if keys & forbidden:
            bad_forbidden.append(row_index)
        if not row["target_tokens"] or row["target_tokens"][0] != "[TARGET_BOS]" or row["target_tokens"][-1] != "[TARGET_EOS]":
            bad_shape.append(row_index)
    source_train = {row["source_meta"]["source_id"] for row in records if row["split"] == "train"}
    source_holdout = {row["source_meta"]["source_id"] for row in records if row["split"] in {"implementation_holdout", "third_surface_holdout", "ask_holdout"}}
    surface_train = {row["source_meta"]["surface_id"] for row in records if row["split"] == "train"}
    surface_holdout = {row["source_meta"]["surface_id"] for row in records if row["split"] in {"implementation_holdout", "third_surface_holdout", "ask_holdout"}}
    audit = {
        "schema_version": "pg322-cross-impl-decoy-dataset-audit-v1",
        "checks": {
            "records_present": bool(records),
            "train_present": any(row["split"] == "train" for row in records),
            "implementation_holdout_present": any(row["split"] == "implementation_holdout" for row in records),
            "third_surface_holdout_present": any(row["split"] == "third_surface_holdout" for row in records),
            "ask_holdout_present": any(row["split"] == "ask_holdout" for row in records),
            "hard_negative_present": any(row["split"] == "hard_negative_eval" for row in records),
            "source_split_isolated": bool(source_train) and bool(source_holdout) and surface_train.isdisjoint(surface_holdout),
            "forbidden_context_absent": not bad_forbidden,
            "target_shape": not bad_shape,
            "raw_excluded": all(not row["raw_payload_stored"] and not row["raw_response_body_stored"] for row in records),
            "ask_rows_safe": all(not row["expected_safe"] for row in records if row["expected_question"] != "none"),
        },
        "bad_forbidden_indices": bad_forbidden,
        "bad_shape_indices": bad_shape,
        "audit_sha256": "",
    }
    audit["status"] = "passed" if all(audit["checks"].values()) else "failed"
    audit["audit_sha256"] = _digest(audit)
    return dataset, audit


def main() -> int:
    dataset, audit = build()
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["status"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
