"""Plan-only gate for a PG-388 varied-surface Rule-IR candidate.

The planner reads the local abstract source-row artifact, computes bounded
split/vocabulary/capacity diagnostics, and refuses to start an optimizer.  It
never emits rows or token values and never starts Docker, CUDA, networking, or
payload generation.  A later runner must satisfy every gate and obtain review
before it can train anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "research" / "pg388_logic_surface_source_rows_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_surface_candidate_plan_v1.json"
SCHEMA_VERSION = "pg388-logic-surface-candidate-plan-v1"
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source artifact must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(values: Any) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sequence_summary(values: list[list[str]]) -> dict[str, Any]:
    signatures = {_digest(value) for value in values}
    lengths = [len(value) for value in values]
    return {
        "count": len(values),
        "unique": len(signatures),
        "unique_ratio": round(len(signatures) / len(values), 6) if values else 0.0,
        "length_max": max(lengths, default=0),
    }


def _plan(source_path: Path) -> dict[str, Any]:
    document = _load(source_path)
    wrappers = document.get("rows") if isinstance(document.get("rows"), list) else []
    rows = [item.get("source_row") for item in wrappers if isinstance(item, Mapping) and isinstance(item.get("source_row"), Mapping)]
    split_rows: dict[str, list[Mapping[str, Any]]] = {"train": [], "implementation_holdout": []}
    implementations: set[str] = set()
    for row in rows:
        split = str(row.get("split", "unknown"))
        split_rows.setdefault(split, []).append(row)
        meta = row.get("source_meta") if isinstance(row.get("source_meta"), Mapping) else {}
        if meta.get("implementation"):
            implementations.add(str(meta["implementation"]))
    contexts = {split: [list(row.get("context_tokens", [])) for row in values if isinstance(row.get("context_tokens"), list)] for split, values in split_rows.items()}
    targets = {split: [list(row.get("target_tokens", [])) for row in values if isinstance(row.get("target_tokens"), list)] for split, values in split_rows.items()}
    context_sets = {split: {_digest(value) for value in values} for split, values in contexts.items()}
    target_sets = {split: {_digest(value) for value in values} for split, values in targets.items()}
    train_vocab = {token for sequence in contexts.get("train", []) for token in sequence}
    holdout_tokens = {token for sequence in contexts.get("implementation_holdout", []) for token in sequence}
    unknown_holdout = sorted(holdout_tokens - train_vocab)
    incomplete = sum(1 for item in wrappers if isinstance(item, Mapping) and item.get("strict_valid") is not True)
    declared_contract = document.get("source_contract") if isinstance(document.get("source_contract"), Mapping) else {}
    failures: list[str] = []
    if not rows:
        failures.append("source_rows_missing")
    if not split_rows.get("train"):
        failures.append("train_split_missing")
    if not split_rows.get("implementation_holdout"):
        failures.append("implementation_holdout_missing")
    if len(implementations) < 2:
        failures.append("independent_implementation_missing")
    if incomplete:
        failures.append("incomplete_source_rows")
    if unknown_holdout:
        failures.append("holdout_vocabulary_gap")
    if context_sets.get("train", set()) & context_sets.get("implementation_holdout", set()):
        failures.append("cross_split_context_overlap")
    if target_sets.get("train", set()) & target_sets.get("implementation_holdout", set()):
        failures.append("cross_split_target_overlap")
    if declared_contract.get("operator_reviewed") is not True:
        failures.append("operator_review_missing")
    if int(declared_contract.get("training_eligible", 0) or 0) <= 0:
        failures.append("training_eligible_zero")
    contract_passed = not failures
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_surface_candidate_plan" if contract_passed else "blocked_surface_source_contract",
        "source": {
            "file": source_path.name,
            "sha256": _sha256(source_path),
            "declared_status": str(document.get("status", "unknown")),
            "row_count": len(rows),
            "implementation_count": len(implementations),
            "split_counts": {split: len(values) for split, values in sorted(split_rows.items()) if values},
        },
        "sequence_diversity": {
            "by_split": {split: {"context": _sequence_summary(contexts.get(split, [])), "target": _sequence_summary(targets.get(split, []))} for split in sorted(contexts) if contexts.get(split)},
            "cross_split_context_overlap": len(context_sets.get("train", set()) & context_sets.get("implementation_holdout", set())),
            "cross_split_target_overlap": len(target_sets.get("train", set()) & target_sets.get("implementation_holdout", set())),
        },
        "train_only_vocabulary": {
            "scope": "train_context_only",
            "size": len(train_vocab),
            "holdout_unknown_token_count": len(unknown_holdout),
            "unknown_token_digest": _digest(unknown_holdout),
        },
        "capacity": {
            "context_length_max": max((len(value) for value in contexts.get("train", []) + contexts.get("implementation_holdout", [])), default=0),
            "target_length_max": max((len(value) for value in targets.get("train", []) + targets.get("implementation_holdout", [])), default=0),
            "required_context_window": max((len(value) for value in contexts.get("train", []) + contexts.get("implementation_holdout", [])), default=0),
            "max_length": 768,
        },
        "source_contract": {
            "in_process_fixture_only": declared_contract.get("in_process_fixture_only") is True,
            "row_bound_typed_evidence": declared_contract.get("row_bound_typed_evidence") is True,
            "fresh_role_reset_attested": declared_contract.get("fresh_role_reset_attested") is True,
            "operator_reviewed": declared_contract.get("operator_reviewed") is True,
            "training_eligible": int(declared_contract.get("training_eligible", 0) or 0),
        },
        "candidate_config": {
            "architecture": "decoder_only_rule_ir_surface_candidate",
            "d_model": 384,
            "n_layers": 6,
            "microbatch": 4,
            "epochs": 1,
            "optimizer_started": False,
            "device": "none",
        },
        "gate": {
            "passed": contract_passed,
            "failures": sorted(set(failures)),
            "gpu_touched": False,
            "docker_started": False,
            "network_contacted": False,
            "checkpoint_written": False,
        },
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
    }


def plan(source_path: Path = DEFAULT_SOURCE, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    report = _plan(source_path)
    report["report_sha256"] = _digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = plan(args.source, args.output)
    print(json.dumps({"status": report["status"], "failures": report["gate"]["failures"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "ready_surface_candidate_plan" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["plan"]
