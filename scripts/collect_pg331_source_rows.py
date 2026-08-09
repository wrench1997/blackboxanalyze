"""Ingest sanitized local-adapter observations into PG-331A source rows.

This is an offline boundary, not a crawler or an HTTP client.  The input is
expected to be produced by an authorised loopback/browser adapter.  Each item
is passed through :func:`app.pg331_source_row.collect_pg331_source_row`; the
output never copies the input observation, raw payload, response body, or
evaluator answer.  Rejected items are represented by an error class and an
input digest only, so a bad capture cannot silently become training data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import SCHEMA_VERSION, collect_pg331_source_row, sha256_json


COLLECTION_SCHEMA = "pg331-source-row-collection-v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _items(document: Any) -> list[Mapping[str, Any]]:
    values = document.get("records") if isinstance(document, Mapping) else document
    return [value for value in values or [] if isinstance(value, Mapping)]


def _rejected(index: int, item: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    record_id = str(item.get("record_id") or f"input:{index}")[:256]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "training_eligible": False,
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "collector_status": "rejected",
        "collector_error_class": type(error).__name__,
        "collector_error_code": "strict_schema_rejection",
        "input_digest": sha256_json(item),
        "failures": ["collector_rejected"],
    }


def collect_rows(document: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts = {"input": 0, "accepted": 0, "incomplete": 0, "rejected": 0, "training_eligible": 0}
    for index, item in enumerate(_items(document)):
        counts["input"] += 1
        try:
            row = collect_pg331_source_row(
                record_id=str(item.get("record_id") or f"input:{index}"),
                observation=item.get("observation") or {},
                source_meta=item.get("source_meta") or {},
                reset=item.get("reset") or {},
                evaluator=item.get("evaluator") or {},
                field_capture_manifest=item.get("field_capture_manifest") or {},
                target_projection=item.get("target_projection") or {},
                split=str(item.get("split", "unassigned")),
                operator_reviewed=bool(item.get("operator_reviewed", False)),
                hard_negative=bool(item.get("hard_negative", False)),
            )
        except (TypeError, ValueError, KeyError) as error:
            rows.append(_rejected(index, item, error))
            counts["rejected"] += 1
            continue
        rows.append(row)
        counts["accepted"] += 1
        counts["incomplete"] += int(not bool(row.get("training_eligible")))
        counts["training_eligible"] += int(bool(row.get("training_eligible")))
    dataset: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA,
        "collector": "scripts/collect_pg331_source_rows.py",
        "local_adapter_only": True,
        "records": rows,
        "counts": counts,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="sanitized adapter JSON/JSONL-like array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    document = _load(args.input)
    dataset = collect_rows(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset["counts"], ensure_ascii=False) if not args.json else json.dumps(dataset, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
