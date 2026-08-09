"""Plan a leakage-safe PG-331 train/holdout split without training.

The planner is deliberately stricter than a normal dataset splitter: it never
relabels rows, never copies context tokens, and only considers rows already
marked ``training_eligible=true`` by the source-row contract.  It emits row
identifiers and hashes for an operator/audit record; callers must not interpret
``passed`` as model capability or promotion permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import validate_pg331_source_row  # noqa: E402


SCHEMA_VERSION = "pg331-train-holdout-plan-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    return json.loads(resolved.read_text(encoding="utf-8-sig")), _file_sha256(resolved)


def _rows(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = document.get("records")
    if not isinstance(values, list):
        raise ValueError("dataset records must be a list")
    return [item for item in values if isinstance(item, Mapping)]


def _implementation(row: Mapping[str, Any]) -> str:
    source_meta = row.get("source_meta")
    return str(source_meta.get("implementation", "unknown")) if isinstance(source_meta, Mapping) else "unknown"


def _family(row: Mapping[str, Any]) -> str:
    source_meta = row.get("source_meta")
    return str(source_meta.get("family_id", "unknown")) if isinstance(source_meta, Mapping) else "unknown"


def _row_digest(row: Mapping[str, Any]) -> str:
    # The digest is only an audit join key; context tokens are never emitted.
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def plan(path: Path, *, train_implementation: str, holdout_implementation: str) -> dict[str, Any]:
    document, file_sha256 = _load(path)
    rows = _rows(document)
    if train_implementation == holdout_implementation:
        raise ValueError("train and holdout implementations must differ")

    validation: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id", ""))
        if record_id:
            validation[record_id] = validate_pg331_source_row(row)

    train_rows = [
        row
        for row in rows
        if _implementation(row) == train_implementation
        and str(row.get("split", "")) in {"train", "dev"}
        and bool(row.get("training_eligible"))
        and bool(validation.get(str(row.get("record_id", "")), {}).get("valid"))
    ]
    holdout_rows = [
        row
        for row in rows
        if _implementation(row) == holdout_implementation
        and str(row.get("split", "")) in {"implementation_holdout", "family_holdout", "route_holdout", "dev"}
    ]
    failures: list[str] = []
    if not train_rows:
        failures.append("empty:eligible_train_rows")
    if not holdout_rows:
        failures.append("empty:holdout_rows")
    if any(_implementation(row) == holdout_implementation and str(row.get("split", "")) in {"train", "dev"} for row in rows):
        failures.append("holdout_implementation_in_train_split")
    if any(_implementation(row) == train_implementation and str(row.get("split", "")) in {"implementation_holdout", "family_holdout"} for row in rows):
        failures.append("train_implementation_in_holdout_split")
    if any(not bool(validation.get(str(row.get("record_id", "")), {}).get("valid")) for row in train_rows):
        failures.append("invalid_train_row")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "blocked",
        "dataset": str(path.resolve()),
        "dataset_sha256": file_sha256,
        "train_implementation": train_implementation,
        "holdout_implementation": holdout_implementation,
        "counts": {
            "input_rows": len(rows),
            "eligible_train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "validated_rows": sum(int(bool(item.get("valid"))) for item in validation.values()),
            "training_eligible_input_rows": sum(int(bool(row.get("training_eligible"))) for row in rows),
        },
        "train_row_ids": [str(row.get("record_id")) for row in train_rows],
        "holdout_row_ids": [str(row.get("record_id")) for row in holdout_rows],
        "train_row_digests": [_row_digest(row) for row in train_rows],
        "holdout_row_digests": [_row_digest(row) for row in holdout_rows],
        "failures": sorted(set(failures)),
        "promotion": PROMOTION,
        "interpretation": "A passed plan only authorizes a later, separately gated training command; this artifact never changes row splits or grants capability.",
    }
    report["plan_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="plan PG-331 train/holdout split without training")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--train-implementation", required=True)
    parser.add_argument("--holdout-implementation", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = plan(args.dataset, train_implementation=args.train_implementation, holdout_implementation=args.holdout_implementation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"plan_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report if args.json else {"status": report["status"], "counts": report["counts"], "failures": report["failures"]}, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["plan"]
