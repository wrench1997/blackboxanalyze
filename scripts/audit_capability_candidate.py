#!/usr/bin/env python3
"""Audit existing experiment reports against the PG-30 capability gate.

This is an offline, read-only checker.  It does not train, load a checkpoint,
generate a dataset, or make network requests.  The only output mutation is
the explicitly requested JSON audit report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.capability_candidate_audit import audit_capability_reports  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline audit of explicit PG-30 model-capability evidence."
    )
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Existing JSON experiment report; may be repeated for batch auditing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/capability_candidate_audit_v1.json"),
        help="Audit JSON output path (the only file this command writes).",
    )
    args = parser.parse_args()
    reports = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.report]
    result = audit_capability_reports(reports)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # A blocked/no-proven-gain result is a valid audit outcome, not a CLI
    # crash.  Exit non-zero only for an authorised capability claim so a CI
    # caller must opt in to promotion explicitly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

