"""Build a balanced multi-axis representation diagnostic corpus.

PG-339 retained the historical split and consequently had only nine training
rows from one implementation.  PG-340 uses an explicit *new* representation
split: Pikachu and WebGoat are training implementations and DVWA is a held-out
implementation.  The original split is retained as metadata; it is never
silently rewritten.  Only abstract context/target projections are written.

This script does not contact Docker, send requests, or train a model.  It is a
diagnostic source-row transformation and keeps every promotion flag false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DEFAULT_SOURCE = RESEARCH / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"
DEFAULT_OUTPUT = RESEARCH / "pg340_balanced_axis_representation_dataset_v1.json"
SCHEMA = "pg340-balanced-axis-representation-dataset-v1"
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
PRESENCE_KEYS = {
    "document_structure": "document_presence",
    "navigation": "navigation_presence",
    "request_transport": "request_transport_presence",
    "response_transport": "response_transport_presence",
    "javascript_surface": "javascript_presence",
    "failure_feedback": "failure_feedback_presence",
    "belief_and_replay": "belief_replay_presence",
}
TRAIN_IMPLEMENTATIONS = ("pikachu-fixed", "webgoat")
HOLDOUT_IMPLEMENTATIONS = ("vulnerables-web-dvwa",)
FORBIDDEN = (
    "family=", "implementation=", "route=", "route_literal=", "source=",
    "image=", "path=", "url=", "payload=", "payload_", "raw_",
    "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"source dataset must be an object: {path}")
    return value


def _tokens(value: Any) -> list[str]:
    return [str(item) for item in list(value or [])]


def _parsed(tokens: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for token in tokens:
        if "=" in token and not token.startswith("["):
            key, value = token.split("=", 1)
            parsed.setdefault(key, []).append(value)
    return parsed


def _implementation_hash(implementation: str) -> str:
    # The label is retained only as a one-way source split attestation.
    return _sha({"implementation": implementation})


def _project(raw: Mapping[str, Any], source_hash: str) -> tuple[dict[str, Any] | None, list[str]]:
    source_meta = raw.get("source_meta")
    implementation = str(dict(source_meta or {}).get("implementation", "")) if isinstance(source_meta, Mapping) else ""
    failures: list[str] = []
    if implementation not in set(TRAIN_IMPLEMENTATIONS) | set(HOLDOUT_IMPLEMENTATIONS):
        failures.append("implementation_not_allowlisted")
    context = _tokens(raw.get("context_tokens"))
    target = _tokens(raw.get("target_tokens"))
    if not context or not target:
        failures.append("context_or_target_missing")
    if any(any(marker in token.casefold() for marker in FORBIDDEN) for token in context):
        failures.append("context_firewall")
    if raw.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        failures.append("firewall_metadata")
    if any(raw.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
        failures.append("raw_oracle_flag")
    parsed = _parsed(context)
    manifest = raw.get("field_capture_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != set(AXES):
        failures.append("field_manifest")
    if any(PRESENCE_KEYS[axis] not in parsed for axis in AXES):
        failures.append("seven_axis_presence")
    source_record_hash = str(raw.get("record_sha256", raw.get("source_record_sha256", "")))
    if len(source_record_hash) != 64:
        failures.append("source_record_hash")
    if failures:
        return None, sorted(set(failures))
    source_split = str(raw.get("split", "unknown"))
    split = "train" if implementation in TRAIN_IMPLEMENTATIONS else "shape_holdout"
    record = {
        "schema_version": SCHEMA,
        "record_id": f"pg340-{_sha({'source': source_record_hash, 'implementation': implementation})[:24]}",
        "split": split,
        "source_split": source_split,
        "source_dataset_sha256": source_hash,
        "source_record_sha256": source_record_hash,
        "source_implementation_hash": _implementation_hash(implementation),
        "context_target_sha256": _sha({"context": context, "target": target}),
        "context_tokens": context,
        "target_tokens": target,
        "field_capture_manifest": manifest,
        "axis_presence": {axis: parsed[PRESENCE_KEYS[axis]][0] for axis in AXES},
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "promotion": {
            "training_eligible": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    record["record_sha256"] = _sha(record)
    return record, []


def build(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source = _load(source_path)
    source_hash = _sha(source)
    accepted: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    failures: Counter[str] = Counter()
    for raw in list(source.get("records") or []):
        if not isinstance(raw, Mapping):
            failures["row_not_object"] += 1
            continue
        projected, reasons = _project(raw, source_hash)
        if projected is None:
            for reason in reasons:
                failures[reason] += 1
            continue
        key = str(projected["context_target_sha256"])
        if key in accepted:
            duplicates.append({
                "context_target_sha256": key,
                "kept_record_sha256": str(accepted[key]["record_sha256"]),
                "discarded_record_sha256": str(projected["record_sha256"]),
            })
            continue
        accepted[key] = projected
    records = sorted(accepted.values(), key=lambda row: str(row["record_sha256"]))
    split_counts = Counter(str(row["split"]) for row in records)
    impl_hashes = {
        "train": sorted({str(row["source_implementation_hash"]) for row in records if row["split"] == "train"}),
        "shape_holdout": sorted({str(row["source_implementation_hash"]) for row in records if row["split"] == "shape_holdout"}),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "diagnostic_only_pending_information_gate",
        "purpose": "balanced multi-axis representation diagnostic; implementation holdout is never training input",
        "source": {
            "dataset_sha256": source_hash,
            "original_split_preserved_as": "source_split",
            "train_implementations": list(TRAIN_IMPLEMENTATIONS),
            "holdout_implementations": list(HOLDOUT_IMPLEMENTATIONS),
            "implementation_labels_not_in_context": True,
        },
        "records": records,
        "counts": {
            "input_rows": len(list(source.get("records") or [])),
            "accepted_rows": len(records),
            "train_rows": int(split_counts["train"]),
            "shape_holdout_rows": int(split_counts["shape_holdout"]),
            "duplicate_rows": len(duplicates),
            "rejected_rows": sum(failures.values()),
            "accepted_training_rows": 0,
            "train_implementation_count": len(impl_hashes["train"]),
            "holdout_implementation_count": len(impl_hashes["shape_holdout"]),
        },
        "duplicate_manifest": duplicates,
        "rejection_reason_counts": dict(sorted(failures.items())),
        "implementation_hashes_by_split": impl_hashes,
        "isolation": {
            "source_split_preserved": True,
            "implementation_split_explicit": True,
            "implementation_split_disjoint": not bool(set(impl_hashes["train"]) & set(impl_hashes["shape_holdout"])),
            "shape_holdout_excluded_from_training": True,
        },
        "information_gate": {
            "status": "pending_audit",
            "field_entropy_required": True,
            "axis_sequence_entropy_required": True,
            "field_ablation_required": True,
            "predictive_entropy_required": True,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-340 balanced-axis diagnostic dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(args.source)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
