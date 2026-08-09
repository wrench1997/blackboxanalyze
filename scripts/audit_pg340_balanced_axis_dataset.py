"""Read-only information and implementation-isolation audit for PG-340."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg340_balanced_axis_representation_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg340_balanced_axis_representation_audit_v1.json"
SCHEMA = "pg340-balanced-axis-representation-dataset-v1"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
FORBIDDEN = ("payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "url=", "path=", "route=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _entropy(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "unique": 0, "bits": None}
    counts = Counter(values)
    total = len(values)
    return {"count": total, "unique": len(counts), "bits": round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)}


def _axis_segment(tokens: list[str], axis: str) -> list[str]:
    try:
        start = tokens.index(f"axis_begin={axis}")
        stop = tokens.index(f"axis_end={axis}", start + 1)
    except ValueError:
        return []
    return tokens[start : stop + 1]


def _field_status(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def audit(data: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in list(data.get("records") or []) if isinstance(row, Mapping)]
    failures: list[str] = []
    if data.get("schema_version") != SCHEMA:
        failures.append("schema")
    split = Counter(str(row.get("split")) for row in rows)
    train_keys: set[str] = set()
    holdout_keys: set[str] = set()
    impl_by_split = {"train": set(), "shape_holdout": set()}
    forbidden_count = 0
    for row in rows:
        tokens = [str(token) for token in list(row.get("context_tokens") or [])]
        forbidden_count += sum(any(marker in token.casefold() for marker in FORBIDDEN) for token in tokens)
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append("firewall_metadata")
        if any(row.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append("raw_oracle_flag")
        if row.get("training_eligible") is not False:
            failures.append("training_flag")
        key = str(row.get("context_target_sha256", ""))
        split_name = str(row.get("split", ""))
        impl = str(row.get("source_implementation_hash", ""))
        if split_name == "train":
            train_keys.add(key)
            impl_by_split["train"].add(impl)
        elif split_name == "shape_holdout":
            holdout_keys.add(key)
            impl_by_split["shape_holdout"].add(impl)
        if any(not bool((row.get("axis_presence") or {}).get(axis)) for axis in AXES):
            failures.append("axis_presence")
    if not rows:
        failures.append("records")
    if not split.get("train"):
        failures.append("train_missing")
    if not split.get("shape_holdout"):
        failures.append("shape_holdout_missing")
    if train_keys & holdout_keys:
        failures.append("context_target_split_overlap")
    if impl_by_split["train"] & impl_by_split["shape_holdout"]:
        failures.append("implementation_split_overlap")
    if forbidden_count:
        failures.append("context_firewall")
    axis_report: dict[str, Any] = {}
    for axis in AXES:
        train_rows = [row for row in rows if str(row.get("split")) == "train"]
        holdout_rows = [row for row in rows if str(row.get("split")) == "shape_holdout"]
        train_segments = [json.dumps(_axis_segment([str(token) for token in row.get("context_tokens") or []], axis), ensure_ascii=False, separators=(",", ":")) for row in train_rows]
        holdout_segments = [json.dumps(_axis_segment([str(token) for token in row.get("context_tokens") or []], axis), ensure_ascii=False, separators=(",", ":")) for row in holdout_rows]
        field_values: list[str] = []
        for row in rows:
            manifest = row.get("field_capture_manifest") or {}
            axis_fields = manifest.get(axis) if isinstance(manifest, Mapping) else {}
            if isinstance(axis_fields, Mapping):
                field_values.extend(_field_status(value) for value in axis_fields.values())
        ablated = []
        original = []
        for row in holdout_rows:
            tokens = [str(token) for token in row.get("context_tokens") or []]
            original.append(_sha(tokens))
            segment = _axis_segment(tokens, axis)
            ablated.append(_sha([token for token in tokens if token not in set(segment)]))
        axis_report[axis] = {
            "train_sequence_entropy": _entropy(train_segments),
            "shape_holdout_sequence_entropy": _entropy(holdout_segments),
            "field_status_entropy": _entropy(field_values),
            "field_ablation": {"eligible_rows": len(holdout_rows), "unique_before": len(set(original)), "unique_after": len(set(ablated)), "changed_rate": round(sum(a != b for a, b in zip(original, ablated)) / len(original), 6) if original else None},
        }
    result: dict[str, Any] = {
        "schema_version": "pg340-balanced-axis-representation-audit-v1",
        "status": "blocked_information_gate" if failures else "diagnostic_only_information_gate_pending",
        "dataset_sha256": str(data.get("dataset_sha256", "")),
        "counts": {"records": len(rows), "train": int(split["train"]), "shape_holdout": int(split["shape_holdout"]), "forbidden_token_count": forbidden_count, "train_implementation_count": len(impl_by_split["train"]), "holdout_implementation_count": len(impl_by_split["shape_holdout"])},
        "axis_entropy": axis_report,
        "split_implementation_isolation": {"context_target_overlap_count": len(train_keys & holdout_keys), "implementation_overlap_count": len(impl_by_split["train"] & impl_by_split["shape_holdout"]), "passed": not bool(train_keys & holdout_keys or impl_by_split["train"] & impl_by_split["shape_holdout"])},
        "information_gate": {"field_entropy_measured": True, "axis_sequence_entropy_measured": True, "field_ablation_measured": True, "predictive_entropy_holdout": "not_run", "passed": False},
        "scientific_gate": {"accepted_training_rows": 0, "reason": "predictive-entropy holdout and capability labels are not yet evaluated"},
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-340 balanced-axis diagnostic dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
