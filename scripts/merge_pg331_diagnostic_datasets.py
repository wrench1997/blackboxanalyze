"""Merge sanitized PG-331 source-row datasets for cross-implementation audit.

This is an append-only, diagnostic-only operation.  It preserves each source
row exactly as emitted by its authorized adapter, records the input file
hashes, rejects duplicate record identifiers, and never changes split or
training eligibility.  The output is intentionally not a training manifest.
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

from app.pg331_source_row import sha256_json  # noqa: E402


SCHEMA_VERSION = "pg331-cross-implementation-diagnostic-merge-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(document: Any) -> list[Mapping[str, Any]]:
    values = document.get("records") if isinstance(document, Mapping) else document
    if not isinstance(values, list):
        raise ValueError("dataset records must be a list")
    return [item for item in values if isinstance(item, Mapping)]


def merge(paths: Sequence[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("cross-implementation merge requires at least two datasets")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifests: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        source_rows = _records(document)
        input_ids: list[str] = []
        for row in source_rows:
            record_id = str(row.get("record_id", ""))
            if not record_id:
                raise ValueError(f"missing record_id in {_label(path)}")
            if record_id in seen:
                raise ValueError(f"duplicate record_id: {record_id}")
            seen.add(record_id)
            input_ids.append(record_id)
            rows.append(dict(row))
        manifests.append(
            {
                "path": _label(path),
                "file_sha256": _file_sha256(path),
                "reported_dataset_sha256": str(document.get("dataset_sha256", "")),
                "record_count": len(source_rows),
                "record_id_sha256": sha256_json(sorted(input_ids)),
            }
        )

    implementation_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    training_eligible = 0
    for row in rows:
        source_meta = row.get("source_meta") if isinstance(row.get("source_meta"), Mapping) else {}
        implementation = str(source_meta.get("implementation", "unknown"))
        family = str(source_meta.get("family_id", "unknown"))
        implementation_counts[implementation] = implementation_counts.get(implementation, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        training_eligible += int(bool(row.get("training_eligible")))

    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "collector": "scripts/merge_pg331_diagnostic_datasets.py",
        "diagnostic_only": True,
        "input_manifests": manifests,
        "records": rows,
        "counts": {
            "input_datasets": len(manifests),
            "records": len(rows),
            "implementations": len(implementation_counts),
            "families": len(family_counts),
            "training_eligible": training_eligible,
            "duplicate_record_ids": 0,
        },
        "implementation_counts": implementation_counts,
        "family_counts": family_counts,
        "promotion": PROMOTION,
        "interpretation": "Merged rows remain diagnostic until typed/fresh/negative/replay, field entropy, source/implementation/family holdouts, operator review, and capacity gates pass.",
    }
    output["dataset_sha256"] = sha256_json(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="merge PG-331 datasets for diagnostic cross-implementation audit")
    parser.add_argument("--input", action="append", required=True, type=Path, help="sanitized PG-331 dataset; repeat at least twice")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        output = merge(args.input)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"merge_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output if args.json else output["counts"], ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["merge"]
