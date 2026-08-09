"""Read-only audit for the PG-389 abstract JS chain dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg389_js_chain_projection import CHAIN_CASES, SCHEMA_VERSION  # noqa: E402


DEFAULT_DATASET = ROOT / "research" / "pg389_js_decode_filter_chain_dataset_v1.json"
FORBIDDEN = ("http://", "https://", "wire=", "payload=", "response_body=", "<script", "javascript:", "source_text=")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = max(len(values), 1)
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def audit_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = dataset.get("rows")
    if dataset.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_mismatch")
    if dataset.get("status") != "abstract_js_chain_candidate_only":
        failures.append("status_mismatch")
    if not isinstance(rows, list) or not rows:
        failures.append("rows_missing")
        rows = []
    raw_markers = 0
    row_hash_failures = 0
    train_context: set[tuple[str, ...]] = set()
    holdout_context: set[tuple[str, ...]] = set()
    train_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    holdout_pairs: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    axes: dict[str, list[str]] = {"decoder_chain": [], "filter_stage": [], "guard_precedence": [], "sink_context": [], "observation_sequence": []}
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append("row_not_object")
            continue
        text = json.dumps(row, ensure_ascii=False, sort_keys=True)
        raw_markers += sum(text.casefold().count(marker) for marker in FORBIDDEN)
        core = {key: value for key, value in row.items() if key != "row_sha256"}
        if row.get("row_sha256") != _sha(core):
            row_hash_failures += 1
        context = tuple(str(item) for item in row.get("context_tokens", []))
        target = tuple(str(item) for item in row.get("target_tokens", []))
        pair = (context, target)
        split = row.get("split")
        if split == "train":
            train_context.add(context)
            train_pairs.add(pair)
        elif split == "implementation_holdout":
            holdout_context.add(context)
            holdout_pairs.add(pair)
        chain = row.get("decode_filter_context", {})
        if isinstance(chain, Mapping):
            axes["decoder_chain"].append("→".join(str(item) for item in chain.get("decoder_chain", [])))
            axes["filter_stage"].append(str(chain.get("filter_stage", "")))
            axes["guard_precedence"].append(str(chain.get("guard_precedence", "")))
            axes["sink_context"].append(str(chain.get("sink_context", "")))
            axes["observation_sequence"].append("→".join(str(item) for item in chain.get("observation_sequence", [])))
        if row.get("source_text_stored") is not False or row.get("raw_value_stored") is not False or row.get("typed_evaluator_observed") is not False:
            failures.append("context_firewall_open")
    overlap_context = len(train_context & holdout_context)
    overlap_pairs = len(train_pairs & holdout_pairs)
    if overlap_context:
        failures.append("cross_split_context_overlap")
    if overlap_pairs:
        failures.append("cross_split_context_target_overlap")
    if row_hash_failures:
        failures.append("row_hash_mismatch")
    if raw_markers:
        failures.append("raw_marker_detected")
    expected = len(CHAIN_CASES) * 2 * 3 * 4
    if len(rows) != expected:
        failures.append("row_count_mismatch")
    coverage = {
        key: {"unique": len(set(values)), "entropy_bits": _entropy(values)}
        for key, values in axes.items()
    }
    report = {
        "schema_version": "pg389-js-decode-filter-chain-audit-v1",
        "status": "passed_candidate_audit" if not failures else "blocked",
        "dataset_status": dataset.get("status"),
        "counts": {
            "records": len(rows),
            "train": sum(1 for row in rows if isinstance(row, Mapping) and row.get("split") == "train"),
            "implementation_holdout": sum(1 for row in rows if isinstance(row, Mapping) and row.get("split") == "implementation_holdout"),
            "expected_records": expected,
        },
        "failures": sorted(set(failures)),
        "context_firewall": {"raw_marker_count": raw_markers, "row_hash_failures": row_hash_failures, "source_text_stored": False, "raw_value_stored": False},
        "split_isolation": {"cross_split_context_overlap": overlap_context, "cross_split_context_target_overlap": overlap_pairs},
        "coverage": coverage,
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def audit_path(dataset_path: str | Path = DEFAULT_DATASET, output_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(dataset_path)
    report = audit_dataset(json.loads(path.read_text(encoding="utf-8-sig")))
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="research/pg389_js_decode_filter_chain_audit_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_path(args.dataset, args.output)
    print(json.dumps(report if args.json else {"status": report["status"], "failures": report["failures"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
