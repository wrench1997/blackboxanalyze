"""Expand abstract Rule-IR rows into independent token-slot queries.

Each training example keeps the complete original context and appends only a
schema query token (for example ``slot_query=next_action``).  The target is a
single abstract value token between BOS/EOS.  This removes autoregressive
error propagation between unrelated slots without putting any target value
back into the context.  It is still a causal next-token LM experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "research" / "pg359_context_index_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg360_slotwise_dataset_v1.json"
SLOTS = (
    "question",
    "ask_reason",
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
DEFAULTS = {
    "question": "none",
    "ask_reason": "none",
    "next_action": "none",
    "repair_action": "none",
    "transport_ref": "unknown",
    "field_role_ref": "unknown",
    "encoding_ref": "unknown",
    "probe_variant_ref": "none",
    "safe_to_send": "0",
    "payload_shape_ref": "unknown",
    "oracle_ref": "unknown",
    "negative_control_presence_ref": "unknown",
}
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_map(tokens: Sequence[str]) -> dict[str, str]:
    result = dict(DEFAULTS)
    for token in tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key in result:
            result[key] = value
    return result


def _raw_free(tokens: Sequence[str]) -> bool:
    return not any(any(fragment in str(token).casefold() for fragment in RAW_FRAGMENTS) for token in tokens)


def _query_tokens(slot: str) -> list[str]:
    return ["[SLOT_QUERY_BOS]", f"slot_query={slot}", "[SLOT_QUERY_EOS]"]


def build(dataset: Mapping[str, Any], *, input_sha256: str, input_path: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    source_records = list(dataset.get("records") or [])
    context_tokens_added: set[str] = {"[SLOT_QUERY_BOS]", "[SLOT_QUERY_EOS]"}
    target_tokens_added: set[str] = {"[TARGET_BOS]", "[TARGET_EOS]"}
    for source_index, source in enumerate(source_records):
        if not isinstance(source, Mapping):
            failures.append(f"source_{source_index}:not_mapping")
            continue
        original_context = [str(token) for token in source.get("context_tokens") or []]
        original_target = [str(token) for token in source.get("target_tokens") or []]
        if not original_context or not original_target or not _raw_free(original_context + original_target):
            failures.append(f"source_{source_index}:stream_or_raw")
            continue
        values = _target_map(original_target)
        source_digest = _sha({"record_id": source.get("record_id"), "context": original_context, "split": source.get("split")})
        for slot in SLOTS:
            query = _query_tokens(slot)
            context = [*original_context, *query]
            value = values[slot]
            row = {
                "schema_version": "pg360-slotwise-row-v1",
                "record_id": _sha({"source": source_digest, "slot": slot}),
                "source_record_digest": source_digest,
                "split": str(source.get("split", "")),
                "slot": slot,
                "context_tokens": context,
                "target_tokens": ["[TARGET_BOS]", f"{slot}={value}", "[TARGET_EOS]"],
                "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
                "oracle_answer_in_context": False,
                "slot_query_contract": {
                    "query_is_schema_only": True,
                    "target_value_in_context": False,
                    "source_context_sha256": _sha(original_context),
                    "source_target_tokens_read_for_value_only": True,
                },
                "operator_reviewed": False,
                "training_eligible": False,
                "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
            }
            row["record_sha256"] = _sha(row)
            records.append(row)
            context_tokens_added.update(query)
            target_tokens_added.add(f"{slot}={value}")

    base_vocab = dict(dataset.get("vocabulary") or {})
    context_vocab = list(dict.fromkeys(str(token) for token in base_vocab.get("context_tokens") or []))
    target_vocab = list(dict.fromkeys(str(token) for token in base_vocab.get("target_tokens") or []))
    for token in sorted(context_tokens_added):
        if token not in context_vocab:
            context_vocab.append(token)
    for token in sorted(target_tokens_added):
        if token not in target_vocab:
            target_vocab.append(token)
    shared = sorted(set(context_vocab) | set(target_vocab) | {str(token) for token in base_vocab.get("shared_tokens") or []})
    result = {
        "schema_version": "pg360-slotwise-dataset-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "source_dataset": input_path,
        "source_dataset_sha256": input_sha256,
        "records": records,
        "slot_order": list(SLOTS),
        "defaults": dict(DEFAULTS),
        "vocabulary": {**base_vocab, "context_tokens": sorted(context_vocab), "target_tokens": sorted(target_vocab), "shared_tokens": shared, "slot_query_tokens": sorted(context_tokens_added), "append_only": True},
        "slot_query_contract": {
            "full_original_context_preserved": True,
            "query_tokens_are_schema_only": True,
            "target_values_not_in_context": True,
            "raw_payload_in_context": False,
            "evaluator_sidecar_read": False,
            "slot_count": len(SLOTS),
            "source_rows": len(source_records),
        },
        "counts": {
            "source_rows": len(source_records),
            "records": len(records),
            "slots": len(SLOTS),
            "train_rows": sum(str(row.get("split")) == "train" for row in records),
            "implementation_holdout_rows": sum(str(row.get("split")) == "implementation_holdout" for row in records),
            "raw_payload_in_context": 0,
            "target_information_added_to_context": 0,
        },
        "failures": sorted(failures),
        "provenance": {"builder": "scripts/build_pg360_slotwise_dataset.py", "builder_sha256": _file_sha(Path(__file__))},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(json.loads(args.input.read_text(encoding="utf-8-sig")), input_sha256=_file_sha(args.input), input_path=str(args.input.resolve().relative_to(ROOT.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
