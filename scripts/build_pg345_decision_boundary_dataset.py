"""Add an explicit abstract context/target boundary to PG-344 rows.

This is a representation ablation, not new security data: all original
seven-axis context tokens, targets, provenance hashes and split assignments
are retained.  The single boundary token only tests whether the causal model
can identify where Rule-IR decoding begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research/pg344_cross_impl_role_bound_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research/pg345_decision_boundary_role_bound_dataset_v1.json"
SCHEMA_VERSION = "pg345-decision-boundary-role-bound-dataset-v1"
BOUNDARY = "decision_boundary=target"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(dataset: Mapping[str, Any], *, dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for raw in dataset.get("records") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("row_not_mapping")
        context = [str(token) for token in raw.get("context_tokens") or []]
        target = [str(token) for token in raw.get("target_tokens") or []]
        if BOUNDARY in context:
            raise ValueError("boundary_already_present")
        if not context or target[:1] != ["[TARGET_BOS]"]:
            raise ValueError("row_boundary_missing")
        row = dict(raw)
        row["context_tokens"] = [*context, BOUNDARY]
        row["record_sha256"] = _sha(row)
        records.append(row)

    if len({(_sha(row["context_tokens"]), _sha(row["target_tokens"])) for row in records}) != len(records):
        raise ValueError("boundary_created_duplicate")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only_pending_target_audit",
        "purpose": "decision-boundary ablation without deleting full-axis information",
        "base_dataset_sha256": str(dataset.get("dataset_sha256", "")),
        "base_dataset_file_sha256": _file_sha(dataset_path),
        "boundary_token": BOUNDARY,
        "records": records,
        "counts": {
            "input_rows": len(dataset.get("records") or []),
            "accepted_rows": len(records),
            "train_rows": sum(row.get("split") == "train" for row in records),
            "implementation_holdout_rows": sum(row.get("split") == "implementation_holdout" for row in records),
            "boundary_tokens_added": len(records),
            "accepted_training_rows": 0,
        },
        "information_preservation": {
            "original_context_tokens_retained": True,
            "original_target_tokens_retained": True,
            "seven_axis_tokens_deleted": 0,
            "boundary_is_abstract": True,
            "raw_payload_response_oracle_added": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "dataset_sha256": "",
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-345 decision-boundary diagnostic dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    result = build(dataset, dataset_path=args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
