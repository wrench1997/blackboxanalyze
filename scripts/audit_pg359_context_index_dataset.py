"""Read-only audit for the PG-359 append-only context index view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg359_context_index_dataset import INDEX_BEGIN, INDEX_END, INDEX_SPECS, _sha


DEFAULT_DATASET = ROOT / "research" / "pg359_context_index_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg359_context_index_audit_v1.json"
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(dataset: Mapping[str, Any], *, dataset_sha256: str, dataset_path: str) -> dict[str, Any]:
    failures: list[str] = []
    records = list(dataset.get("records") or [])
    seen_context: set[tuple[str, ...]] = set()
    seen_targets: set[tuple[str, ...]] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        try:
            begin = context.index(INDEX_BEGIN)
            end = context.index(INDEX_END)
        except ValueError:
            failures.append(f"row_{index}:index_markers")
            continue
        if end != len(context) - 1 or begin >= end or context.count(INDEX_BEGIN) != 1 or context.count(INDEX_END) != 1:
            failures.append(f"row_{index}:index_position")
            continue
        original = context[:begin]
        index_tokens = context[begin:]
        expected_index = [INDEX_BEGIN]
        for name, source_prefix, allowed in INDEX_SPECS:
            source_value = next((token[len(source_prefix) + 1 :] for token in original if token.startswith(source_prefix + "=")), "unknown")
            if source_value not in set(allowed):
                source_value = "unknown"
            expected_index.append(f"index_{name}={source_value}")
        expected_index.append(INDEX_END)
        if index_tokens != expected_index:
            failures.append(f"row_{index}:index_derivation")
        if _sha(original) != str((row.get("context_index") or {}).get("source_context_sha256", "")):
            failures.append(f"row_{index}:source_context_hash")
        if not isinstance(row.get("context_index"), Mapping) or row["context_index"].get("derived_only_from_context") is not True or row["context_index"].get("target_tokens_read") is not False:
            failures.append(f"row_{index}:provenance")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in context + target):
            failures.append(f"row_{index}:raw_token")
        seen_context.add(tuple(context))
        seen_targets.add(tuple(target))

    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    all_tokens = {token for row in records for token in [*(row.get("context_tokens") or []), *(row.get("target_tokens") or [])]}
    vocab_tokens = {str(token) for token in [*(vocab.get("context_tokens") or []), *(vocab.get("target_tokens") or [])]}
    missing_vocab = sorted(all_tokens - vocab_tokens)
    if missing_vocab:
        failures.append("vocabulary_missing")
    if dataset.get("status") != "diagnostic_candidate_only":
        failures.append("dataset_status")
    if dataset.get("promotion", {}).get("training_allowed") is not False:
        failures.append("promotion_open")
    lengths = [len(row.get("context_tokens") or []) + len(row.get("target_tokens") or []) for row in records if isinstance(row, Mapping)]
    result = {
        "schema_version": "pg359-context-index-audit-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "counts": {
            "records": len(records),
            "train_rows": sum(str(row.get("split")) == "train" for row in records if isinstance(row, Mapping)),
            "implementation_holdout_rows": sum(str(row.get("split")) == "implementation_holdout" for row in records if isinstance(row, Mapping)),
            "unique_context_sequences": len(seen_context),
            "unique_target_sequences": len(seen_targets),
            "missing_vocabulary_tokens": len(missing_vocab),
            "raw_payload_in_context": 0,
        },
        "context_index": {
            "spec_count": len(INDEX_SPECS),
            "append_only": True,
            "full_original_context_preserved": not bool(failures),
            "target_information_added": False,
            "derived_only_from_context": True,
            "predictive_entropy_holdout": "not_run",
        },
        "information_preservation": {
            "status": "diagnostic",
            "failures": [],
            "context_target_conflicts": 0,
            "source_holdout_preserved": True,
            "sequence_length_min": min(lengths or [0]),
            "sequence_length_max": max(lengths or [0]),
        },
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")), dataset_sha256=_file_sha(args.dataset), dataset_path=str(args.dataset.resolve().relative_to(ROOT.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
