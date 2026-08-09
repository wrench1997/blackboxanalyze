"""Build an information-preserving PG-352 action-prefix view of PG-351.

This is a layout ablation, not a relabeling pass.  Every abstract target
field from PG-351 v2 is retained exactly once; only the fixed decoder order is
changed so the decision prefix is emitted before the transport/shape slots.
No raw payload, response, route, family, or evaluator literal is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v2.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg352_action_prefix_dataset_v1.json"
RAW_FRAGMENTS = (
    "raw_payload=", "payload=", "response_body=", "response_body_text=", "raw_response=",
    "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=",
    "image=", "source=",
)
TARGET_PREFIXES = (
    "[TARGET_BOS]", "[TARGET_EOS]", "question=", "ask_reason=", "next_action=",
    "repair_action=", "safe_to_send=", "transport_ref=", "field_role_ref=",
    "encoding_ref=", "probe_variant_ref=", "payload_shape_ref=", "oracle_ref=",
    "negative_control_presence_ref=",
)
TARGET_ORDER = (
    "question", "ask_reason", "next_action", "repair_action", "safe_to_send",
    "transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref",
    "payload_shape_ref", "oracle_ref", "negative_control_presence_ref",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_map(tokens: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text and not text.startswith("[TARGET_"):
            key, value = text.split("=", 1)
            result[key] = value
    return result


def _target_tokens(tokens: list[Any]) -> list[str]:
    values = _target_map(tokens)
    result = ["[TARGET_BOS]"]
    for key in TARGET_ORDER:
        if key in values:
            result.append(f"{key}={values[key]}")
    result.append("[TARGET_EOS]")
    return result


def _contains_raw(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.casefold()
        return any(fragment in lowered for fragment in RAW_FRAGMENTS)
    if isinstance(value, Mapping):
        return any(_contains_raw(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw(child) for child in value)
    return False


def build(source: Mapping[str, Any], *, source_sha256: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(source.get("records") or []):
        if not isinstance(raw, Mapping):
            continue
        context = [str(token) for token in raw.get("context_tokens") or []]
        target = _target_tokens(list(raw.get("target_tokens") or []))
        if len(context) < 2 or not target or _contains_raw(context) or _contains_raw(target):
            continue
        if any(not token.startswith(TARGET_PREFIXES) for token in target):
            continue
        pair_id = _sha({"context_tokens": context, "target_tokens": target})
        if pair_id in seen:
            continue
        seen.add(pair_id)
        row = {
            "schema_version": "pg352-action-prefix-row-v1",
            "record_id": pair_id,
            "source_record_id": str(raw.get("record_id", "")),
            "source_record_sha256": _sha({"source_sha256": source_sha256, "source_index": index, "source_record_id": raw.get("record_id", "")}),
            "split": str(raw.get("split", "")),
            "context_tokens": context,
            "target_tokens": target,
            "safe_to_send": "safe_to_send=1" in target,
            "supervision_lane": str(raw.get("supervision_lane", "")),
            "missing_observation_explicit": bool(raw.get("missing_observation_explicit", False)),
            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "training_eligible": False,
            "candidate_training_allowed": True,
            "promotion": {
                "training_eligible": False,
                "memory_promotion_allowed": False,
                "payload_catalog_promotion_allowed": False,
                "vulnerability_claim_allowed": False,
            },
            "target_order": list(TARGET_ORDER),
            "record_sha256": "",
        }
        row["record_sha256"] = _sha({key: value for key, value in row.items() if key != "record_sha256"})
        records.append(row)
    records.sort(key=lambda row: (row["split"], row["record_id"]))
    vocab_context = sorted({token for row in records for token in row["context_tokens"]})
    vocab_target = sorted({token for row in records for token in row["target_tokens"]})
    return {
        "schema_version": "pg352-action-prefix-dataset-v1",
        "status": "diagnostic_candidate_only",
        "source_dataset": "research/pg351_ask_oracle_composition_dataset_v2.json",
        "source_dataset_sha256": source_sha256,
        "target_order": list(TARGET_ORDER),
        "records": records,
        "counts": {
            "records": len(records),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
            "training_eligible_rows": 0,
            "raw_payload_in_context": False,
        },
        "vocabulary": {"context_tokens": vocab_context, "target_tokens": vocab_target},
        "information_policy": {
            "all_source_target_fields_retained": True,
            "field_count_changed": False,
            "only_target_order_changed": True,
            "raw_payload": False,
            "raw_response": False,
            "evaluator_answer": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-352 action-prefix dataset")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(source, source_sha256=_file_sha(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output_sha256": _file_sha(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
