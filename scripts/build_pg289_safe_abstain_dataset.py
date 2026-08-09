"""Build PG-289 training-only unresolved/evaluator-gap counterfactuals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg285_payload_grounding_dataset_v1.json"
OUTPUT = RESEARCH / "pg289_safe_abstain_dataset_v1.json"
AUDIT = RESEARCH / "pg289_safe_abstain_dataset_audit_v1.json"
SIGNATURES = ("unsupported_binding", "unresolved_surface", "oracle_unavailable", "unknown_effect")
TARGET_ACTIONS = frozenset({"abstain"})
FORBIDDEN_CONTEXT_PREFIXES = ("family=", "source=", "probe=", "oracle_label=")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def replace_token(tokens: list[str], prefix: str, replacement: str) -> list[str]:
    result = [token for token in tokens if not token.startswith(prefix)]
    result.append(replacement)
    return result


def abstain_target(base: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    target = dict(base.get("target") or {})
    method = str(target.get("method", base.get("method", "GET")))
    target = {
        "next_action": "abstain",
        "method": method if method in {"GET", "POST"} else "GET",
        "probe_class": "other",
        "channel": "unknown",
        "encoding": "unknown",
        "wire_kind": "none",
        "repair_delta": "none",
        "safe_to_send": False,
        "oracle_required": True,
    }
    tokens = [
        "[TARGET_BOS]",
        "plan=abstain",
        f"method={target['method']}",
        "probe_class=other",
        "channel=unknown",
        "encoding=unknown",
        "wire=none",
        "field_slot=none",
        "repair_delta=none",
        "family_agnostic=1",
        "final_action=abstain",
        "safe_to_send=0",
        "[TARGET_EOS]",
    ]
    return tokens, target


def make_record(base: dict[str, Any], index: int, signature: str) -> dict[str, Any]:
    context = [str(token) for token in list(base.get("context_tokens") or [])]
    context = replace_token(context, "failure_signature=", f"failure_signature={signature}")
    context = replace_token(context, "feedback=", "feedback=unresolved")
    context = replace_token(context, "typed_available=", "typed_available=0")
    context = replace_token(context, "candidate_sent=", "candidate_sent=0")
    context = replace_token(context, "replay_consistent=", "replay_consistent=0")
    context = replace_token(context, "oracle_label_in_context=", "oracle_label_in_context=0")
    context = replace_token(context, "phase=", "phase=guard")
    # The decoy must be family/source agnostic.  Preserve only surface and
    # transport observations, never a family or a target label.
    context = [token for token in context if not any(token.startswith(prefix) for prefix in FORBIDDEN_CONTEXT_PREFIXES)]
    target_tokens, target = abstain_target(base)
    record = {
        "record_id": f"pg289:{index:04d}:{digest({'base': base.get('record_id'), 'signature': signature})[:16]}",
        "source_record_id": f"pg285-train-template:{base.get('record_id', index)}",
        "source_group_id": "pg289:unresolved-evaluator-counterfactual",
        "source": "pg289_counterfactual",
        "family": None,
        "method": target["method"],
        "split": "train",
        "surface_variant": str(base.get("surface_variant", "abstract")),
        "state": "unresolved_evaluator",
        "replica": int(index % 4),
        "hard_negative": False,
        "training_decoy": True,
        "context_tokens": context,
        "target_tokens": target_tokens,
        "target": target,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "source_evidence_hash": digest({"source_record_id": base.get("record_id"), "signature": signature}),
    }
    return record


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_rows = [row for row in list(source.get("records") or []) if row.get("split") == "train"]
    if not base_rows:
        raise RuntimeError("PG-289 requires PG-285 train rows")
    records = [make_record(base_rows[index % len(base_rows)], index, SIGNATURES[index % len(SIGNATURES)]) for index in range(1512)]
    dataset = {
        "schema_version": "pg289-safe-abstain-dataset-v1",
        "purpose": "training-only family/source-agnostic unresolved evaluator counterfactuals for safe abstain",
        "source": {"base_dataset": str(SOURCE.relative_to(ROOT).as_posix()), "base_dataset_sha256": source["dataset_sha256"], "generation": "train-template-only", "signatures": list(SIGNATURES)},
        "records": records,
        "counts": {"train": len(records), "total": len(records), "signature_count": len(SIGNATURES)},
        "training_contract": {"remote_a800_required": True, "literal_probe_values_out_of_context": True, "raw_response_bodies_out_of_context": True, "family_labels_out_of_context": True, "hard_negative_eval_only": True, "memory_promotion_allowed": False},
        "scientific_contract": {"not_real_application_gold": True, "not_a_vulnerability_claim": True, "must_evaluate_on_source_heldout_and_family_holdout": True},
    }
    dataset["dataset_sha256"] = digest(dataset)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = {
        "base_hash_present": dataset["source"]["base_dataset_sha256"] == source["dataset_sha256"],
        "row_quota": len(records) == 1512,
        "train_only": all(row.get("split") == "train" for row in records),
        "target_abstain": all(row.get("target", {}).get("next_action") == "abstain" and row.get("target", {}).get("safe_to_send") is False for row in records),
        "context_family_agnostic": all(row.get("family") is None and not any(any(str(token).startswith(prefix) for prefix in FORBIDDEN_CONTEXT_PREFIXES) for token in row.get("context_tokens", [])) for row in records),
        "no_literal_payload": all(not row.get("raw_payload_strings_stored") and not row.get("raw_response_bodies_stored") and not row.get("oracle_in_context") for row in records),
        "training_memory_split": all(row.get("training_eligible") is True and row.get("memory_promotion_allowed") is False for row in records),
        "signature_coverage": {signature: sum(1 for row in records if f"failure_signature={signature}" in row.get("context_tokens", [])) for signature in SIGNATURES},
    }
    checks["all_signature_rows_present"] = all(int(value) > 0 for value in checks["signature_coverage"].values())
    audit = {
        "audit_id": "pg289-safe-abstain-dataset-independent-audit-v1",
        "status": "passed" if all(bool(value) for key, value in checks.items() if key != "signature_coverage") else "failed",
        "dataset": str(OUTPUT.relative_to(ROOT).as_posix()),
        "dataset_sha256": dataset["dataset_sha256"],
        "checks": checks,
        "interpretation": "PG-289 只增加训练用的 evaluator-gap 反事实；族外 hard-negative 仍不进入训练，真实 gold/记忆晋级仍关闭。",
    }
    audit["audit_sha256"] = digest(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "records": len(records), "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
