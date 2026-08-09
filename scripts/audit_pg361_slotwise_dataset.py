"""Read-only PG-361 slotwise dataset audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_pg361_slotwise_dataset import SLOTS, _sha

RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=", "image=")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(dataset: Mapping[str, Any], *, dataset_sha256: str, dataset_path: str) -> dict[str, Any]:
    failures: list[str] = []
    rows = list(dataset.get("records") or [])
    slots_seen: set[str] = set()
    source_slot_pairs: set[tuple[str, str]] = set()
    source_contexts: dict[str, tuple[str, ...]] = {}
    raw_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        slot = str(row.get("slot", ""))
        if slot not in SLOTS:
            failures.append(f"row_{index}:slot")
        slots_seen.add(slot)
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if len(context) < 4 or context[-3:] != ["[SLOT_QUERY_BOS]", f"slot_query={slot}", "[SLOT_QUERY_EOS]"]:
            failures.append(f"row_{index}:query")
            continue
        original = context[:-3]
        contract = row.get("slot_query_contract") if isinstance(row.get("slot_query_contract"), Mapping) else {}
        if contract.get("target_value_in_context") is not False or contract.get("query_is_schema_only") is not True:
            failures.append(f"row_{index}:contract")
        if _sha(original) != str(contract.get("source_context_sha256", "")):
            failures.append(f"row_{index}:context_hash")
        if target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"] or len(target) != 3 or not target[1].startswith(slot + "="):
            failures.append(f"row_{index}:target")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in context + target):
            raw_count += 1
        key = (str(row.get("source_record_digest", "")), slot)
        if key in source_slot_pairs:
            failures.append(f"row_{index}:duplicate")
        source_slot_pairs.add(key)
        source_digest = key[0]
        if source_digest in source_contexts and source_contexts[source_digest] != tuple(original):
            failures.append(f"row_{index}:source_context_conflict")
        source_contexts[source_digest] = tuple(original)
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall")
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    all_tokens = {str(token) for row in rows for token in [*(row.get("context_tokens") or []), *(row.get("target_tokens") or [])]}
    vocab_tokens = {str(token) for token in [*(vocab.get("context_tokens") or []), *(vocab.get("target_tokens") or [])]}
    missing_vocab = sorted(all_tokens - vocab_tokens)
    if missing_vocab:
        failures.append("vocabulary_missing")
    if str(dataset.get("status")) != "diagnostic_candidate_only":
        failures.append("dataset_status")
    if raw_count:
        failures.append("raw_context_or_target")
    slot_value_counts: dict[str, dict[str, int]] = {}
    for slot in SLOTS:
        counter: Counter[str] = Counter()
        for row in rows:
            if str(row.get("slot")) != slot:
                continue
            target_tokens = list(row.get("target_tokens") or [])
            if len(target_tokens) == 3 and "=" in str(target_tokens[1]):
                counter[str(target_tokens[1]).split("=", 1)[1]] += 1
        slot_value_counts[slot] = dict(sorted(counter.items()))
    result = {
        "schema_version": "pg361-slotwise-audit-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "counts": {
            "records": len(rows),
            "source_rows": len(source_contexts),
            "slots_expected": len(SLOTS),
            "slots_seen": len(slots_seen),
            "missing_vocabulary_tokens": len(missing_vocab),
            "duplicate_source_slot_pairs": len(rows) - len(source_slot_pairs),
            "raw_payload_in_context": raw_count,
            "training_eligible_rows": sum(bool(row.get("training_eligible")) for row in rows),
        },
        "information_preservation": {
            "original_context_preserved": not bool(failures),
            "target_information_added_to_context": False,
            "query_is_schema_only": True,
            "predictive_entropy_holdout": "not_run",
        },
        "slot_value_counts": slot_value_counts,
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-361 syntax slotwise dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg361_syntax_slotwise_dataset_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg361_syntax_slotwise_audit_v1.json")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    result = audit(dataset, dataset_sha256=_file_sha(args.dataset), dataset_path=str(args.dataset.resolve().relative_to(ROOT.resolve())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "diagnostic_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
