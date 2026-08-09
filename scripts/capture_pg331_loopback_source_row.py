"""Capture one local page and pass it through the PG-331A source-row gate.

This is an operator-facing bridge, not a scanner.  It accepts only an
explicit loopback origin and GET or a neutral POST whose field values are
empty.  The response body is parsed by ``capture_loopback`` in memory and is
never written to the output.  Provenance/reset/evaluator/target projections
are supplied as separate JSON sidecars and remain outside model context.

The command is intentionally useful even when the sidecars are incomplete:
the strict collector returns an auditable ASK/incomplete row rather than
silently promoting a baseline observation to training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_loopback_adapter import capture_loopback
from app.pg331_source_row import SCHEMA_VERSION, collect_pg331_source_row, sha256_json


COLLECTION_SCHEMA = "pg331-loopback-source-row-capture-v1"
FIELD_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def _load_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _safe_post_fields(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if not FIELD_NAME.fullmatch(str(value)):
            raise ValueError("--post-field accepts only an ASCII field name")
        result[str(value)] = ""
    return result


def capture_source_row(
    *,
    origin: str,
    method: str,
    post_fields: list[str],
    record_id: str,
    source_meta: Mapping[str, Any],
    reset: Mapping[str, Any],
    evaluator: Mapping[str, Any],
    target_projection: Mapping[str, Any],
    split: str = "unassigned",
    operator_reviewed: bool = False,
    hard_negative: bool = False,
) -> dict[str, Any]:
    """Capture in memory, then apply the strict source-row contract."""

    normalized_method = str(method).upper()
    form_data = _safe_post_fields(post_fields) if normalized_method == "POST" else None
    capture = capture_loopback(origin, method=normalized_method, form_data=form_data)
    return collect_pg331_source_row(
        record_id=record_id,
        observation=capture["observation"],
        source_meta=source_meta,
        reset=reset,
        evaluator=evaluator,
        field_capture_manifest=capture["field_capture_manifest"],
        target_projection=target_projection,
        split=split,
        operator_reviewed=operator_reviewed,
        hard_negative=hard_negative,
    )


def _dataset(row: Mapping[str, Any]) -> dict[str, Any]:
    records = [dict(row)]
    eligible = int(bool(row.get("training_eligible")))
    dataset: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA,
        "collector": "scripts/capture_pg331_loopback_source_row.py",
        "source_row_schema": SCHEMA_VERSION,
        "local_adapter_only": True,
        "records": records,
        "counts": {"input": 1, "accepted": 1, "incomplete": 1 - eligible, "rejected": 0, "training_eligible": eligible},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one authorized loopback page into a strict PG-331A source row")
    parser.add_argument("--origin", required=True, help="explicit http(s)://127.0.0.1|localhost|[::1]:port origin")
    parser.add_argument("--method", choices=("GET", "POST"), default="GET")
    parser.add_argument("--post-field", action="append", default=[], help="neutral POST field name; value is always empty")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--source-meta-json", type=Path, required=True)
    parser.add_argument("--reset-json", type=Path, required=True)
    parser.add_argument("--evaluator-json", type=Path, required=True)
    parser.add_argument("--target-projection-json", type=Path, required=True)
    parser.add_argument("--split", default="unassigned")
    parser.add_argument("--operator-reviewed", action="store_true")
    parser.add_argument("--hard-negative", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    row = capture_source_row(
        origin=args.origin,
        method=args.method,
        post_fields=list(args.post_field),
        record_id=args.record_id,
        source_meta=_load_object(args.source_meta_json, "source_meta"),
        reset=_load_object(args.reset_json, "reset"),
        evaluator=_load_object(args.evaluator_json, "evaluator"),
        target_projection=_load_object(args.target_projection_json, "target_projection"),
        split=args.split,
        operator_reviewed=bool(args.operator_reviewed),
        hard_negative=bool(args.hard_negative),
    )
    dataset = _dataset(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dataset if args.json else dataset["counts"], ensure_ascii=False, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

