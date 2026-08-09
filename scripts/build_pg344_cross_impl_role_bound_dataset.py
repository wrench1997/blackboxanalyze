"""Build an implementation-disjoint PG-344 diagnostic dataset.

PG-343 preserved each source artifact's historical split.  PG-344 defines a
new, explicit experiment split by one-way implementation attestation so that
the training and holdout implementations cannot overlap.  The original
``source_split`` remains in every row; this script never changes a target,
context token, or evaluator fact and never enables promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg343_role_bound_dataset import DEFAULT_SOURCES, _sha, build


DEFAULT_OUTPUT = ROOT / "research/pg344_cross_impl_role_bound_dataset_v1.json"
SCHEMA_VERSION = "pg344-cross-implementation-role-bound-dataset-v1"

# One-way implementation attestations produced by the existing source-row
# collector.  They are deliberately the only identity used by the model split.
TRAIN_IMPLEMENTATION_HASHES = frozenset(
    {
        "8fd17d70f910c4e0f96f2ebd26fa2892ae5f7c751400b5ac141f8a16dbd444c6",
        "a60f75b1234fdb2bf422725a4968e919d0abceca421bf19cb83caf29a5447232",
    }
)
HOLDOUT_IMPLEMENTATION_HASHES = frozenset(
    {"9ae6a3c02cf053298249f15c8c6ba023f926b254f1d8fae9cfe5710e949b8c36"}
)


def build_cross_impl(sources: tuple[Path, ...] = DEFAULT_SOURCES) -> dict[str, Any]:
    base = build(sources)
    records: list[dict[str, Any]] = []
    unknown: Counter[str] = Counter()
    for original in base["records"]:
        implementation = str(original.get("source_implementation_hash", ""))
        if implementation in TRAIN_IMPLEMENTATION_HASHES:
            split = "train"
        elif implementation in HOLDOUT_IMPLEMENTATION_HASHES:
            split = "implementation_holdout"
        else:
            unknown[implementation] += 1
            continue
        row = dict(original)
        row["split"] = split
        row["record_sha256"] = _sha(row)
        records.append(row)

    if unknown:
        raise ValueError("unassigned_implementation_attestation")
    if not records:
        raise ValueError("empty_cross_impl_dataset")

    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        groups[str(row["source_implementation_hash"])][str(row["split"])] += 1
    overlap = {
        implementation: dict(counts)
        for implementation, counts in groups.items()
        if counts.get("train", 0) and counts.get("implementation_holdout", 0)
    }
    if overlap:
        raise ValueError("implementation_split_overlap")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only_pending_target_audit",
        "purpose": "implementation-disjoint target-conditioned Rule-IR diagnostic",
        "source_dataset_sha256": str(base.get("dataset_sha256", "")),
        "source_split_preserved": True,
        "split_policy": {
            "type": "one_way_implementation_attestation",
            "train_implementation_hashes": sorted(TRAIN_IMPLEMENTATION_HASHES),
            "holdout_implementation_hashes": sorted(HOLDOUT_IMPLEMENTATION_HASHES),
            "historical_source_split_retained_in_each_row": True,
            "no_context_or_target_relabeling": True,
        },
        "records": records,
        "counts": {
            "input_rows": int(base["counts"]["accepted_rows"]),
            "accepted_rows": len(records),
            "train_rows": sum(row["split"] == "train" for row in records),
            "implementation_holdout_rows": sum(row["split"] == "implementation_holdout" for row in records),
            "accepted_training_rows": 0,
        },
        "implementation_hash_groups": {key: dict(value) for key, value in sorted(groups.items())},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "dataset_sha256": "",
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-344 cross-implementation diagnostic rows")
    parser.add_argument("--source", action="append", type=Path, dest="sources")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = tuple(args.sources) if args.sources else DEFAULT_SOURCES
    result = build_cross_impl(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
