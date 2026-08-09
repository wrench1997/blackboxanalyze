"""Build the PG-351 abstract ASK + Rule-IR composition candidate dataset.

Two already-produced abstract streams are joined:

* PG-350 typed replay rows (full Rule-IR target slots); and
* PG-348 dynamic context rows whose evaluator is intentionally unavailable
  and whose safe target is ``ask_typed``.

The second stream is not relabelled as a vulnerability result.  It is an
explicit missing-observation supervision lane.  Exact context/target pairs
are deduplicated, all raw/evaluator material is rejected, and every output
row remains candidate-only with promotion disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TYPED = ROOT / "research" / "pg350_oracle_slot_source_rows_v1.json"
DEFAULT_ASK = ROOT / "research" / "pg348_dynamic_context_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg351_ask_oracle_composition_dataset_v1.json"

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
TARGET_PREFIXES = (
    "[TARGET_BOS]",
    "[TARGET_EOS]",
    "question=",
    "ask_reason=",
    "next_action=",
    "repair_action=",
    "transport_ref=",
    "field_role_ref=",
    "encoding_ref=",
    "probe_variant_ref=",
    "safe_to_send=",
    "payload_shape_ref=",
    "oracle_ref=",
    "negative_control_presence_ref=",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_raw(value: Any) -> bool:
    if isinstance(value, str):
        text = value.casefold()
        return any(fragment in text for fragment in RAW_FRAGMENTS)
    if isinstance(value, Mapping):
        return any(_contains_raw(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_raw(child) for child in value)
    return False


def _target_map(tokens: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in tokens:
        text = str(token)
        if "=" in text and not text.startswith("[TARGET_"):
            key, value = text.split("=", 1)
            result[key] = value
    return result


def _safe_value(value: Any) -> bool:
    """Parse serialized boolean slots without treating ``"0"`` as true."""
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def _target_tokens(target: Mapping[str, Any], *, ask: bool) -> list[str]:
    order = (
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
    values = dict(target)
    if ask:
        # These are safe abstract consequences of an unavailable typed
        # evaluator, not a guessed vulnerability answer.
        values.update(
            {
                "question": "ask_typed",
                "ask_reason": "typed_evidence",
                "next_action": "ask_typed",
                "repair_action": "observe",
                "transport_ref": "unknown",
                "field_role_ref": "unknown",
                "encoding_ref": "unknown",
                "probe_variant_ref": "none",
                "safe_to_send": False,
                "payload_shape_ref": "unknown",
                "oracle_ref": "unknown",
                "negative_control_presence_ref": "not_observed",
            }
        )
    result = ["[TARGET_BOS]"]
    for key in order:
        if key not in values:
            continue
        value = int(_safe_value(values[key])) if key == "safe_to_send" else values[key]
        result.append(f"{key}={value}")
    result.append("[TARGET_EOS]")
    return result


def _valid_context_row(row: Mapping[str, Any]) -> bool:
    context = row.get("context_tokens")
    firewall = row.get("context_firewall")
    if not isinstance(context, list) or len(context) < 2:
        return False
    if not isinstance(firewall, Mapping) or firewall.get("forbidden_token_count") != 0 or firewall.get("sidecars_off_context") is not True:
        return False
    if any(row.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
        return False
    if _contains_raw(context):
        return False
    return True


def _make_row(source: Mapping[str, Any], *, lane: str, ask: bool, source_file: str) -> dict[str, Any] | None:
    if not _valid_context_row(source):
        return None
    context = [str(token) for token in source.get("context_tokens") or []]
    source_target = _target_map(list(source.get("target_tokens") or []))
    target = _target_tokens(source_target, ask=ask)
    if any(not token.startswith(TARGET_PREFIXES) for token in target) or _contains_raw(target):
        return None
    pair_digest = _sha({"context_tokens": context, "target_tokens": target})
    row: dict[str, Any] = {
        "schema_version": "pg351-abstract-ask-oracle-row-v1",
        "record_id": pair_digest,
        "split": str(source.get("split", "")),
        "context_tokens": context,
        "target_tokens": target,
        "safe_to_send": "safe_to_send=1" in target,
        "supervision_lane": lane,
        "missing_observation_explicit": bool(ask),
        "source_artifact_digest": _sha({"source_file": source_file, "source_record_id": source.get("record_id", "")}),
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
        "failures": [],
        "record_sha256": "",
    }
    row["record_sha256"] = _sha({key: value for key, value in row.items() if key != "record_sha256"})
    return row


def build(typed: Mapping[str, Any], ask: Mapping[str, Any], *, typed_sha256: str, ask_sha256: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid = Counter()
    seen: set[str] = set()
    input_counts = Counter()
    for source, lane, ask_flag, source_file in (
        (typed, "typed_replay", False, "pg350_oracle_slot_source_rows_v1.json"),
        (ask, "ask_missing_observation", True, "pg348_dynamic_context_dataset_v1.json"),
    ):
        for source_row in list(source.get("records") or []):
            input_counts[lane] += 1
            row = _make_row(source_row, lane=lane, ask=ask_flag, source_file=source_file)
            if row is None:
                invalid[lane] += 1
                continue
            if row["record_id"] in seen:
                invalid["duplicate_exact_context_target"] += 1
                continue
            seen.add(row["record_id"])
            rows.append(row)
    rows.sort(key=lambda item: (str(item.get("split")), str(item.get("record_id"))))
    vocab_context = sorted({token for row in rows for token in row["context_tokens"]})
    vocab_target = sorted({token for row in rows for token in row["target_tokens"]})
    split_counts = Counter(str(row.get("split")) for row in rows)
    lane_counts = Counter(str(row.get("supervision_lane")) for row in rows)
    question_counts = Counter(next((token.split("=", 1)[1] for token in row["target_tokens"] if token.startswith("question=")), "none") for row in rows)
    action_counts = Counter(next((token.split("=", 1)[1] for token in row["target_tokens"] if token.startswith("next_action=")), "none") for row in rows)
    return {
        "schema_version": "pg351-abstract-ask-oracle-composition-dataset-v1",
        "status": "diagnostic_candidate_only" if rows and not invalid.get("typed_replay") and not invalid.get("ask_missing_observation") else "blocked_incomplete",
        "inputs": {
            "typed_dataset": "research/pg350_oracle_slot_source_rows_v1.json",
            "typed_dataset_sha256": typed_sha256,
            "ask_dataset": "research/pg348_dynamic_context_dataset_v1.json",
            "ask_dataset_sha256": ask_sha256,
        },
        "records": rows,
        "counts": {
            "records": len(rows),
            "input_records": sum(input_counts.values()),
            "input_typed_replay": input_counts["typed_replay"],
            "input_ask_missing_observation": input_counts["ask_missing_observation"],
            "duplicate_exact_context_target": invalid["duplicate_exact_context_target"],
            "invalid_records": invalid["typed_replay"] + invalid["ask_missing_observation"],
            "train_rows": split_counts["train"],
            "implementation_holdout_rows": split_counts["implementation_holdout"],
            "training_eligible_rows": 0,
        },
        "supervision_lanes": dict(lane_counts),
        "target_coverage": {"questions": dict(question_counts), "next_actions": dict(action_counts)},
        "vocabulary": {"context_tokens": vocab_context, "target_tokens": vocab_target},
        "context_policy": {"raw_payload": False, "raw_response": False, "evaluator_answer": False, "target_literals": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-351 abstract ASK/oracle candidate dataset")
    parser.add_argument("--typed", type=Path, default=DEFAULT_TYPED)
    parser.add_argument("--ask", type=Path, default=DEFAULT_ASK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    typed = json.loads(args.typed.read_text(encoding="utf-8-sig"))
    ask = json.loads(args.ask.read_text(encoding="utf-8-sig"))
    result = build(typed, ask, typed_sha256=_file_sha(args.typed), ask_sha256=_file_sha(args.ask))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output_sha256": _file_sha(args.output)}, ensure_ascii=False))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
