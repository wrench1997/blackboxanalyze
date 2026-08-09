#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.synthetic_curriculum import generate_curriculum  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate automatically verified SIFT synthetic training data.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/sift-synthetic"))
    parser.add_argument("--programs", type=int, default=5000)
    parser.add_argument("--traces-per-program", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--manifest-only", action="store_true", help="Hash the deterministic corpus without retaining duplicate JSONL files.")
    args = parser.parse_args()

    records = generate_curriculum(args.programs, args.traces_per_program, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "validation": [], "test": []}
    for record in records:
        by_split[record["split"]].append(record)

    manifest = {
        "schema_version": "sift-synthetic-v1",
        "seed": args.seed,
        "programs": len(records),
        "verified_traces": sum(row["verification"]["verified_trace_count"] for row in records),
        "verified_counterexamples": sum(row["verification"]["verified_counterexample_count"] for row in records),
        "family_counts": dict(Counter(row["family"] for row in records)),
        "splits": {},
    }
    for split, split_records in by_split.items():
        output = args.output_dir / f"{split}.jsonl"
        payload = "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for row in split_records) + ("\n" if split_records else "")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if not args.manifest_only:
            output.write_text(payload, encoding="utf-8")
        manifest["splits"][split] = {
            "file": None if args.manifest_only else output.name,
            "records": len(split_records),
            "sha256": digest,
            "bytes": len(payload.encode("utf-8")),
        }
    manifest["dataset_sha256"] = hashlib.sha256(
        json.dumps({name: details["sha256"] for name, details in sorted(manifest["splits"].items())}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest["retention"] = "manifest_only" if args.manifest_only else "jsonl_and_manifest"
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
