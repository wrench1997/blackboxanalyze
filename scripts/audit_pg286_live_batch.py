"""Audit persisted PG-286 live records without promoting them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DEFAULT_INPUT = RESEARCH / "pg286_live_records_v1.json"
DEFAULT_OUTPUT = RESEARCH / "pg286_live_batch_audit_v1.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg286_live_batch import audit_pg286_live_batch


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)], []
    if isinstance(data, dict):
        return [dict(row) for row in list(data.get("records") or []) if isinstance(row, dict)], [dict(row) for row in list(data.get("hard_negative_records") or []) if isinstance(row, dict)]
    raise ValueError("PG-286 live records must be a list or object")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--remote-docker-status", default="unavailable")
    parser.add_argument("--independent-audit-pass", action="store_true")
    args = parser.parse_args()
    records, hard = _load_rows(args.input)
    result = audit_pg286_live_batch(records, hard_negative_records=hard, independent_audit_pass=args.independent_audit_pass, remote_docker_status=args.remote_docker_status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "record_count": result["record_count"], "eligible_record_count": result["eligible_record_count"], "blocking_reasons": result["blocking_reasons"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
