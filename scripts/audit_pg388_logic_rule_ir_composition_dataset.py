"""Read-only audit wrapper for the PG-388 structured Rule-IR dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg388_logic_rule_ir_composition_dataset import audit_dataset


DEFAULT_DATASET = ROOT / "research" / "pg388_logic_rule_ir_composition_dataset_v1.json"


def audit_file(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8-sig"))
    report = audit_dataset(artifact)
    report["dataset"] = str(path)
    report["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["report_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in report.items() if key != "report_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="research/pg388_logic_rule_ir_composition_audit_v1.json")
    args = parser.parse_args()
    report = audit_file(Path(args.dataset))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
