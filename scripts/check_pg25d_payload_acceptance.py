"""Run the evaluator-only PG-25D acceptance gate on a safe Catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg25d_acceptance_gate import evaluate_catalog  # noqa: E402


CATALOG_PATH = ROOT / "research" / "pg_pk_25d_vulnerableapp_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_25d_payload_acceptance_v1.json"


def main() -> int:
    catalog_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CATALOG_PATH
    report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPORT_PATH
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    report = evaluate_catalog(catalog)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
