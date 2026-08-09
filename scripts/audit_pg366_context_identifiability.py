"""Audit context-to-Rule-IR identifiability without training or target contact.

PG-365 reached perfect synthetic holdout scores while predictive entropy
collapsed.  This read-only audit checks whether the abstract context already
uniquely identifies the whole target sequence.  It does not remove tokens or
rewrite data; it reports shortcut risk so a later candidate can be interpreted
correctly.  No raw payload, response, route, family, or evaluator literal is
printed or persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RAW_FRAGMENTS = ("raw_", "payload=", "raw_payload=", "response_body=", "response_body_text=", "route_literal=", "oracle_answer=", "evaluator_answer=")
PRESENCE_SUFFIXES = ("=observed", "=absent", "=not_observed", "=unknown")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entropy(counter: Counter[str] | Counter[tuple[str, ...]]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def _conditional_entropy(groups: Mapping[tuple[str, ...], Counter[tuple[str, ...]]], total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(sum(counter.values()) / total * _entropy(counter) for counter in groups.values())


def _target_sequence(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("target_tokens")
    if not isinstance(value, list) or not value:
        raise ValueError("target_tokens_missing")
    return tuple(str(token) for token in value)


def _context(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("context_tokens")
    if not isinstance(value, list) or not value:
        raise ValueError("context_tokens_missing")
    tokens = tuple(str(token) for token in value)
    for token in tokens:
        lowered = token.casefold()
        if any(fragment in lowered for fragment in RAW_FRAGMENTS):
            raise ValueError("context_firewall_violation")
    return tokens


def _project(tokens: Sequence[str], mode: str) -> tuple[str, ...]:
    if mode == "exact":
        return tuple(tokens)
    if mode == "presence_only":
        return tuple(token for token in tokens if token.endswith(PRESENCE_SUFFIXES))
    if mode == "without_chunk_digest":
        return tuple(token for token in tokens if not token.startswith("chunk_digest="))
    if mode == "without_chunk_metadata":
        prefixes = ("chunk_digest=", "chunk_boundary=", "chunk_index=", "chunk_count=", "chunk_shape=")
        return tuple(token for token in tokens if not token.startswith(prefixes))
    raise ValueError(f"unknown_projection:{mode}")


def _group_target_entropy(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    groups: defaultdict[tuple[str, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in rows:
        groups[_project(_context(row), mode)][_target_sequence(row)] += 1
    target_counts = Counter(_target_sequence(row) for row in rows)
    conditional = _conditional_entropy(groups, len(rows))
    return {
        "projection": mode,
        "group_count": len(groups),
        "unique_target_sequence_count": len(target_counts),
        "target_entropy_bits": round(_entropy(target_counts), 6),
        "conditional_target_entropy_bits": round(conditional, 6),
        "information_gain_bits": round(_entropy(target_counts) - conditional, 6),
        "max_group_rows": max((sum(counter.values()) for counter in groups.values()), default=0),
    }


def audit_document(document: Mapping[str, Any], *, source_path: str, source_sha256: str) -> dict[str, Any]:
    rows = document.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("records_missing")
    failures: list[str] = []
    valid_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        try:
            _context(row)
            _target_sequence(row)
        except ValueError as error:
            failures.append(f"row_{index}:{error}")
            continue
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall_contract")
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError("no_valid_rows")
    exact = _group_target_entropy(valid_rows, "exact")
    presence = _group_target_entropy(valid_rows, "presence_only")
    no_digest = _group_target_entropy(valid_rows, "without_chunk_digest")
    no_chunk_meta = _group_target_entropy(valid_rows, "without_chunk_metadata")
    exact_unique_ratio = exact["group_count"] / len(valid_rows)
    exact_zero_conditional = exact["conditional_target_entropy_bits"] == 0.0
    source_meta_present = sum(bool(row.get("source_meta")) for row in valid_rows)
    split_counts = Counter(str(row.get("split") or "unknown") for row in valid_rows)
    return {
        "schema_version": "pg366-context-identifiability-audit-v1",
        "status": "diagnostic_shortcut_risk" if exact_zero_conditional else "diagnostic",
        "source_dataset": source_path,
        "source_dataset_sha256": source_sha256,
        "counts": {
            "input_rows": len(rows),
            "valid_rows": len(valid_rows),
            "invalid_rows": len(rows) - len(valid_rows),
            "split": dict(sorted(split_counts.items())),
            "source_meta_present_rows": source_meta_present,
        },
        "entropy": {
            "exact_context": exact,
            "presence_only": presence,
            "without_chunk_digest": no_digest,
            "without_chunk_metadata": no_chunk_meta,
        },
        "shortcut_risk": {
            "exact_context_unique_ratio": round(exact_unique_ratio, 6),
            "exact_context_conditional_entropy_zero": exact_zero_conditional,
            "interpretation": "full abstract context uniquely determines the synthetic target sequence; this is a data identifiability finding, not proof of generalization",
            "implementation_provenance_in_rows": source_meta_present > 0,
        },
        "predictive_entropy": {"status": "not_run", "must_not_be_inferred_from_context_entropy": True},
        "failures": sorted(set(failures)),
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "report_sha256": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PG-366 read-only context identifiability audit")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg364_compositional_rule_ir_dataset_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg366_context_identifiability_audit_v1.json")
    args = parser.parse_args()
    document = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    report = audit_document(document, source_path=str(args.dataset.resolve().relative_to(ROOT.resolve())), source_sha256=_file_sha(args.dataset))
    report["report_sha256"] = _sha(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "shortcut_risk": report["shortcut_risk"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
