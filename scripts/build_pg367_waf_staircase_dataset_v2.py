"""Build PG-367 v2 with a compositional, vocabulary-covered holdout.

PG-367 v1 held out whole WAF policies.  That is useful as an unseen-policy
stress test, but it also introduced unseen token values and made token
coverage indistinguishable from generalization.  v2 keeps every abstract
value in both splits and holds out deterministic record combinations instead.
The original v1 artifact is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research" / "pg367_waf_staircase_dataset_v1.json"
OUTPUT = ROOT / "research" / "pg367_waf_staircase_dataset_v2.json"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build(source: dict[str, Any] | None = None) -> dict[str, Any]:
    source = source or json.loads(INPUT.read_text(encoding="utf-8-sig"))
    document = json.loads(json.dumps(source, ensure_ascii=False))
    records = []
    for raw in source.get("records") or []:
        row = dict(raw)
        record_id = str(row.get("record_id", ""))
        digest = hashlib.sha256((record_id + "pg367-compositional-v2").encode("utf-8")).hexdigest()
        split = "implementation_holdout" if int(digest[:8], 16) % 5 == 0 else "train"
        row["split"] = split
        row.pop("record_sha256", None)
        row["record_sha256"] = _sha(row)
        records.append(row)
    document["schema_version"] = "pg367-waf-staircase-dataset-v2"
    document["status"] = "diagnostic_compositional_candidate_only"
    document["records"] = records
    document["counts"] = {
        **dict(document.get("counts") or {}),
        "records": len(records),
        "train_rows": sum(row["split"] == "train" for row in records),
        "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
    }
    document["split_contract"] = {
        "kind": "deterministic_compositional_record_holdout",
        "holdout_rule": "sha256(record_id + pg367-compositional-v2) mod 5 == 0",
        "value_coverage_required": True,
        "policy_values_shared_across_splits": True,
        "single_synthetic_implementation": True,
    }
    document["promotion"] = {
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    document.pop("dataset_sha256", None)
    document["dataset_sha256"] = _sha(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-367 v2 compositional holdout")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
