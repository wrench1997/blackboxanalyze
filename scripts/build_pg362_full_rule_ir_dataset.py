"""Build a full-target PG-362 dataset from fresh PG-361 abstract source rows.

PG-361's slotwise view has one target token per row.  PG-362 deliberately
keeps the original complete Rule-IR target sequence so a causal decoder can
learn cross-slot dependencies.  Only abstract context/target tokens are
copied; evaluator/source sidecars stay out of the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SLOTS = (
    "question",
    "ask_reason",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "syntax_category_ref",
    "probe_variant_ref",
    "safe_to_send",
    "payload_shape_ref",
    "oracle_ref",
    "negative_control_presence_ref",
)
RAW_FRAGMENTS = (
    "raw_payload=",
    "payload=",
    "response_body=",
    "response_body_text=",
    "raw_response=",
    "wire=",
    "evaluator=",
    "oracle=",
    "route_literal=",
    "family=",
    "implementation=",
    "image=",
    "source=",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


DEFAULTS = {
    "question": "none",
    "ask_reason": "none",
    "next_action": "none",
    "repair_action": "none",
    "transport_ref": "unknown",
    "field_role_ref": "unknown",
    "encoding_ref": "unknown",
    "syntax_category_ref": "unknown",
    "probe_variant_ref": "none",
    "safe_to_send": "0",
    "payload_shape_ref": "unknown",
    "oracle_ref": "unknown",
    "negative_control_presence_ref": "unknown",
}


def _target_values(tokens: list[str]) -> dict[str, str]:
    # Normalize older PG-361 source rows into the append-only full Rule-IR
    # contract.  ``ask_reason`` was absent in the source collector; deriving
    # its bounded enum from the already abstract question is not evaluator
    # supervision and avoids silently training a 12-slot target.
    values: dict[str, str] = dict(DEFAULTS)
    for token in tokens:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in SLOTS:
                values[key] = value
    if values["ask_reason"] == "none" and values["question"].startswith("ask_"):
        values["ask_reason"] = "failure_feedback" if values["question"] == "ask_failure" else "typed_evidence"
    return values


def _abstract_tokens(tokens: Any) -> tuple[list[str] | None, str | None]:
    if not isinstance(tokens, list) or not tokens:
        return None, "not_list"
    values = [str(token) for token in tokens]
    if values[0] != "[TARGET_BOS]" or values[-1] != "[TARGET_EOS]":
        return None, "target_boundary"
    if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in values):
        return None, "raw_token"
    return values, None


def build(source: Mapping[str, Any], *, source_sha256: str, source_path: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    context_vocab: set[str] = set()
    target_vocab: set[str] = {"[TARGET_BOS]", "[TARGET_EOS]"}
    for index, raw in enumerate(source.get("records") or []):
        if not isinstance(raw, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        context = raw.get("context_tokens")
        if not isinstance(context, list) or not context:
            failures.append(f"row_{index}:context")
            continue
        context_tokens = [str(token) for token in context]
        target_tokens, target_failure = _abstract_tokens(raw.get("target_tokens"))
        if target_failure:
            failures.append(f"row_{index}:{target_failure}")
            continue
        assert target_tokens is not None
        values = _target_values(target_tokens)
        # Emit one deterministic token per slot, in ontology order.  This is
        # the whole-sequence target that PG-362 trains; the original target
        # order is not a stable contract across PG-361 collector versions.
        target_tokens = ["[TARGET_BOS]", *[f"{slot}={values[slot]}" for slot in SLOTS], "[TARGET_EOS]"]
        if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall")
            continue
        if any(raw.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}:raw_flag")
            continue
        split = str(raw.get("split", ""))
        if split not in {"train", "implementation_holdout"}:
            failures.append(f"row_{index}:split")
            continue
        source_digest = str(raw.get("record_id") or raw.get("record_sha256") or _sha({"index": index, "context": context_tokens, "target": target_tokens}))
        record = {
            "schema_version": "pg362-full-rule-ir-row-v1",
            "record_id": _sha({"source_record": source_digest, "context": context_tokens, "target": target_tokens, "split": split}),
            "source_record_digest": source_digest,
            "split": split,
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "full_target_contract": {"slot_order": list(SLOTS), "target_values_not_in_context": True, "source_sidecars_off_context": True},
            "operator_reviewed": False,
            "training_eligible": False,
            "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        }
        record["record_sha256"] = _sha(record)
        records.append(record)
        context_vocab.update(context_tokens)
        target_vocab.update(target_tokens)
    return {
        "schema_version": "pg362-full-rule-ir-dataset-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "source_dataset": source_path,
        "source_dataset_sha256": source_sha256,
        "records": records,
        "slot_order": list(SLOTS),
        "vocabulary": {"context_tokens": sorted(context_vocab), "target_tokens": sorted(target_vocab), "shared_tokens": sorted(context_vocab | target_vocab), "append_only": True},
        "counts": {
            "records": len(records),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
            "target_slots": len(SLOTS),
            "unique_target_sequences": len({tuple(row["target_tokens"]) for row in records}),
            "training_eligible_rows": 0,
        },
        "failures": sorted(failures),
        "full_target_contract": {"slotwise_source_excluded": True, "whole_sequence_target": True, "context_preserved": True, "raw_payload_in_context": False, "evaluator_sidecar_read": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-362 full Rule-IR target dataset")
    parser.add_argument("--input", type=Path, default=ROOT / "research" / "pg361_dynamic_syntax_typed_source_rows_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg362_full_rule_ir_dataset_v1.json")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(source, source_sha256=_file_sha(args.input), source_path=str(args.input.resolve().relative_to(ROOT.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"]}, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
