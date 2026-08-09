"""Read-only structural audit for the PG-362 full-target dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("question", "ask_reason", "next_action", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "syntax_category_ref", "probe_variant_ref", "safe_to_send", "payload_shape_ref", "oracle_ref", "negative_control_presence_ref")
RAW_FRAGMENTS = ("raw_payload=", "payload=", "response_body=", "response_body_text=", "raw_response=", "wire=", "evaluator=", "oracle=", "route_literal=", "family=", "implementation=", "image=", "source=")


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def audit(dataset: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    records = dataset.get("records") or []
    seen: set[tuple[str, str]] = set()
    split_counts = Counter()
    target_lengths: list[int] = []
    raw_hits = 0
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            failures.append(f"row_{index}:not_mapping")
            continue
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}:firewall")
        if any(row.get(flag) is not False for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
            failures.append(f"row_{index}:raw_flag")
        if target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.append(f"row_{index}:target_boundary")
        values = {token.split("=", 1)[0] for token in target if "=" in token}
        if values != set(SLOTS):
            failures.append(f"row_{index}:slot_coverage")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
            raw_hits += 1
            failures.append(f"row_{index}:raw_token")
        split = str(row.get("split", ""))
        split_counts[split] += 1
        key = (str(row.get("source_record_digest", "")), split)
        if key in seen:
            failures.append(f"row_{index}:duplicate_source")
        seen.add(key)
        target_lengths.append(len(context) + len(target))
    vocab = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
    vocab_tokens = {str(token) for token in [*(vocab.get("context_tokens") or []), *(vocab.get("target_tokens") or [])]}
    missing_vocab = sorted({str(token) for row in records if isinstance(row, Mapping) for token in [*(row.get("context_tokens") or []), *(row.get("target_tokens") or [])]} - vocab_tokens)
    failures.extend(f"missing_vocab:{token}" for token in missing_vocab)
    failures = sorted(set(failures))
    return {
        "schema_version": "pg362-full-rule-ir-audit-v1",
        "status": "diagnostic_candidate_only" if not failures else "blocked_incomplete",
        "counts": {"records": len(records), "train_rows": split_counts.get("train", 0), "implementation_holdout_rows": split_counts.get("implementation_holdout", 0), "raw_hits": raw_hits, "missing_vocabulary_tokens": len(missing_vocab), "training_eligible_rows": 0, "unique_target_sequences": len({tuple(row.get("target_tokens") or []) for row in records if isinstance(row, Mapping)})},
        "capacity": {"min_context_target_length": min(target_lengths or [0]), "max_context_target_length": max(target_lengths or [0]), "required_context_window": max(target_lengths or [0]), "truncation_risk": False},
        "slot_order": list(SLOTS),
        "failures": failures,
        "predictive_entropy": "not_run",
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-362 full Rule-IR dataset")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg362_full_rule_ir_dataset_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg362_full_rule_ir_audit_v1.json")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
