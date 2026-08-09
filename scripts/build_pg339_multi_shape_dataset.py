"""Build a PG-339 multi-page-shape diagnostic corpus without promoting rows.

The written corpus keeps only abstract source-row context/targets plus hashed
provenance.  Its CLI prints a bounded summary, never a token stream.  This is
read-only dataset engineering: it neither contacts targets nor trains a model.
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
PG333 = RESEARCH / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"
PG338 = RESEARCH / "pg338_information_preserving_process_token_v1.json"
OUTPUT = RESEARCH / "pg339_multi_shape_diagnostic_dataset_v1.json"
SCHEMA = "pg339-multi-shape-diagnostic-dataset-v1"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
PRESENCE = {"document_structure": "document_presence", "navigation": "navigation_presence", "request_transport": "request_transport_presence", "response_transport": "response_transport_presence", "javascript_surface": "javascript_presence", "failure_feedback": "failure_feedback_presence", "belief_and_replay": "belief_replay_presence"}
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping): raise ValueError(f"PG-339 dataset must be an object: {path}")
    return value


def _tokens(value: Any) -> list[str]: return [str(item) for item in list(value or [])]


def _parsed(tokens: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for token in tokens:
        if "=" in token and not token.startswith("["):
            key, value = token.split("=", 1); result.setdefault(key, []).append(value)
    return result


def _implementation_hash(row: Mapping[str, Any], source: str) -> str:
    if source == "pg338": return str(row.get("source_implementation_hash", ""))
    return _sha({"implementation": str(dict(row.get("source_meta") or {}).get("implementation", "unknown"))})


def _row(source: str, source_hash: str, row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    context, target = _tokens(row.get("context_tokens")), _tokens(row.get("target_tokens"))
    parsed, failures = _parsed(context), []
    if not context or not target: failures.append("context_or_target_missing")
    if any(any(marker in token.casefold() for marker in FORBIDDEN) for token in context): failures.append("context_firewall")
    if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}: failures.append("firewall_metadata")
    if any(row.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")): failures.append("raw_oracle_flag")
    manifest = row.get("field_capture_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != set(AXES): failures.append("field_manifest")
    if any(PRESENCE[axis] not in parsed for axis in AXES): failures.append("seven_axis_presence")
    if failures: return None, sorted(set(failures))
    source_split = str(row.get("source_split", row.get("split", "unknown")))
    split = "shape_holdout" if source_split == "implementation_holdout" else "train" if source_split == "train" else "unassigned"
    source_record_hash = str(row.get("source_record_sha256", row.get("record_sha256", "")))
    if len(source_record_hash) != 64: return None, ["source_record_hash"]
    key = _sha({"context": context, "target": target})
    result = {"schema_version": SCHEMA, "record_id": f"pg339-{_sha({'source': source, 'row': source_record_hash})[:24]}", "split": split, "source_split": source_split, "source_dataset_sha256": source_hash, "source_record_sha256": source_record_hash, "source_implementation_hash": _implementation_hash(row, source), "context_target_sha256": key, "context_tokens": context, "target_tokens": target, "field_capture_manifest": manifest, "axis_presence": {axis: parsed[PRESENCE[axis]][0] for axis in AXES}, "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True}, "raw_payload_stored": False, "raw_response_body_stored": False, "oracle_answer_in_context": False, "training_eligible": False, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    result["record_sha256"] = _sha(result)
    return result, []


def build(*, pg333_path: Path = PG333, pg338_path: Path = PG338) -> dict[str, Any]:
    accepted: dict[str, dict[str, Any]] = {}; duplicates: list[dict[str, str]] = []; failures: Counter[str] = Counter()
    for source, path in (("pg333", pg333_path), ("pg338", pg338_path)):
        document = _load(path); source_hash = _sha(document)
        for raw in list(document.get("records") or []):
            if not isinstance(raw, Mapping): failures["row_not_object"] += 1; continue
            record, reasons = _row(source, source_hash, raw)
            if record is None:
                for reason in reasons: failures[reason] += 1
                continue
            key = record["context_target_sha256"]
            if key in accepted:
                previous = accepted[key]
                # A holdout always wins the dedupe tie; it may never leak into train.
                if record["split"] == "shape_holdout" and previous["split"] != "shape_holdout": accepted[key] = record
                duplicates.append({"context_target_sha256": key, "kept_record_sha256": str(accepted[key]["record_sha256"]), "discarded_record_sha256": str(record["record_sha256"])})
            else: accepted[key] = record
    records = sorted(accepted.values(), key=lambda item: str(item["record_sha256"]))
    counts = Counter(str(row["split"]) for row in records)
    result: dict[str, Any] = {"schema_version": SCHEMA, "status": "diagnostic_only_pending_information_gate", "purpose": "multi-page-shape diagnostic; source/implementation holdout is never training input", "sources": {"pg333_sha256": _sha(_load(pg333_path)), "pg338_sha256": _sha(_load(pg338_path)), "split_policy": "preserve source_split; source implementation_holdout becomes shape_holdout"}, "records": records, "counts": {"input_rows": sum(counts.values()) + len(duplicates) + sum(failures.values()), "accepted_rows": len(records), "train_rows": int(counts["train"]), "shape_holdout_rows": int(counts["shape_holdout"]), "duplicate_rows": len(duplicates), "rejected_rows": sum(failures.values()), "accepted_training_rows": 0}, "duplicate_manifest": duplicates, "rejection_reason_counts": dict(sorted(failures.items())), "isolation": {"source_split_preserved": True, "implementation_hash_preserved": True, "shape_holdout_excluded_from_training": True}, "information_gate": {"status": "pending_audit", "field_entropy_required": True, "field_ablation_required": True, "split_implementation_isolation_required": True}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False}}
    result["dataset_sha256"] = _sha(result); return result


def _summary(data: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": data.get("status"), "counts": data.get("counts"), "dataset_sha256": data.get("dataset_sha256"), "promotion": data.get("promotion")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-339 multi-shape diagnostic dataset")
    parser.add_argument("--pg333", type=Path, default=PG333); parser.add_argument("--pg338", type=Path, default=PG338); parser.add_argument("--output", type=Path, default=OUTPUT); parser.add_argument("--json", action="store_true")
    args = parser.parse_args(); data = build(pg333_path=args.pg333, pg338_path=args.pg338)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_summary(data), ensure_ascii=False, indent=2 if args.json else None)); return 0


if __name__ == "__main__": raise SystemExit(main())
