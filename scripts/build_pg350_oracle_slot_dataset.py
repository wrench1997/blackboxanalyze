"""Build a versioned PG-350 target-slot view from abstract PG-349 rows.

This is a pure JSON transformation.  It appends ``oracle_ref`` and
``negative_control_presence_ref`` to the Rule-IR target, never copies a raw
probe/response/evaluator body, and quarantines every derived row until a
fresh source/evaluator audit is run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from app.pg331_source_row import sha256_json, validate_pg331_source_row


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg349_dynamic_typed_source_rows_v5.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
RAW_KEYS = frozenset({"payload", "raw_payload", "response_body", "raw_response", "wire", "probe_value", "body"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_raw(value: Any, key: str = "") -> bool:
    key_text = str(key).casefold()
    if key_text in RAW_KEYS or key_text.startswith("raw_"):
        return True
    if isinstance(value, str):
        text = value.casefold()
        return text.startswith(("raw_payload=", "payload=", "response_body=", "raw_response=", "wire="))
    if isinstance(value, dict):
        return any(_contains_raw(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, list):
        return any(_contains_raw(child, key) for child in value)
    return False


def _slots(row: dict[str, Any]) -> dict[str, str]:
    target = dict(row.get("target_projection") or {})
    evaluator = dict(row.get("evaluator_sidecar") or {})
    variant = str(target.get("probe_variant_ref", "none"))
    failure = str(target.get("question", "none")) == "ask_failure"
    typed = evaluator.get("typed_available") is True and evaluator.get("fresh_reset") is True
    if failure or not typed:
        oracle = "unknown"
    elif variant == "negative_control":
        oracle = "negative_no_effect"
    else:
        oracle = "typed_effect"
    matched = (
        evaluator.get("negative_control") is True
        and evaluator.get("reference_present") is True
        and evaluator.get("candidate_present") is True
        and evaluator.get("fresh_reset") is True
    )
    return {
        "oracle_ref": oracle,
        "negative_control_presence_ref": "matched_triplet" if matched else "unknown",
    }


def build(dataset: dict[str, Any], *, input_sha256: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    slot_counts = {"oracle_ref": {}, "negative_control_presence_ref": {}}
    invalid: list[str] = []
    for source_row in list(dataset.get("records") or []):
        row = copy.deepcopy(dict(source_row))
        if _contains_raw(row.get("context_tokens"), "context_tokens") or _contains_raw(row.get("target_tokens"), "target_tokens"):
            invalid.append(str(row.get("record_id", "unknown")))
            continue
        target = dict(row.get("target_projection") or {})
        target.update(_slots(row))
        row["target_projection"] = target
        row["target_tokens"] = ["[TARGET_BOS]"] + [
            f"{key}={int(bool(target[key])) if key == 'safe_to_send' else target[key]}"
            for key in (
                "question",
                "next_action",
                "repair_action",
                "transport_ref",
                "field_role_ref",
                "encoding_ref",
                "probe_variant_ref",
                "safe_to_send",
                "payload_shape_ref",
                "oracle_ref",
                "negative_control_presence_ref",
            )
            if key in target
        ] + ["[TARGET_EOS]"]
        # This derived view is diagnostic until a fresh source-row audit and
        # operator review independently authorize it.
        row["operator_reviewed"] = False
        row["training_eligible"] = False
        row["promotion"] = {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "derived_target_slots": True,
        }
        row["record_sha256"] = ""
        body = dict(row)
        body.pop("record_sha256", None)
        row["record_sha256"] = sha256_json(body)
        check = validate_pg331_source_row(row)
        if not check["valid"]:
            invalid.append(str(row.get("record_id", "unknown")))
            continue
        records.append(row)
        for key in slot_counts:
            value = str(target[key])
            slot_counts[key][value] = int(slot_counts[key].get(value, 0)) + 1
    return {
        "schema_version": "pg350-oracle-slot-source-rows-v1",
        "status": "diagnostic_only" if not invalid else "blocked_incomplete",
        "source_dataset": "research/pg349_dynamic_typed_source_rows_v5.json",
        "source_dataset_sha256": input_sha256,
        "records": records,
        "counts": {
            "records": len(records),
            "input_records": len(list(dataset.get("records") or [])),
            "invalid_records": len(invalid),
            "train_rows": sum(row.get("split") == "train" for row in records),
            "implementation_holdout_rows": sum(row.get("split") == "implementation_holdout" for row in records),
            "training_eligible_rows": 0,
        },
        "target_slot_counts": slot_counts,
        "invalid_record_ids_sha256": hashlib.sha256(json.dumps(sorted(invalid), separators=(",", ":")).encode("utf-8")).hexdigest(),
        "context_policy": {"raw_payload": False, "raw_response": False, "evaluator_answer": False, "target_literals": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(dataset, input_sha256=_file_sha(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output_sha256": _file_sha(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
