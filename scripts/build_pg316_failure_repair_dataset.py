"""Build PG-316 paired failure -> repair next-token records.

The source data already contains missing-observation and variant rows.  This
builder adds explicit counterfactual pairs where the visible surface is held
constant but feedback changes to a failure.  The target must then be a
bounded repair plan (or a fail-closed abstain), never a fresh safe send.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg313_probe_variant_dataset_v1.json"
SOURCE_AUDIT = RESEARCH / "pg313_probe_variant_dataset_audit_v1.json"
OUTPUT = RESEARCH / "pg316_failure_repair_dataset_v1.json"
AUDIT = RESEARCH / "pg316_failure_repair_dataset_audit_v1.json"
FAILURE_CLASSES = ("effect_not_confirmed", "oracle_disagreement", "replay_mismatch", "surface_mismatch")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _replace_token(tokens: list[str], key: str, value: str) -> list[str]:
    output = [token for token in tokens if not str(token).startswith(f"{key}=")]
    marker = next((index for index, token in enumerate(tokens) if str(token) == "[EOS]"), len(output))
    output.insert(marker, f"{key}={value}")
    return output


def _repair_clone(row: dict[str, Any], failure_class: str, ordinal: int) -> dict[str, Any]:
    context = [str(token) for token in row.get("context_tokens") or []]
    context = _replace_token(context, "history_action", "candidate_failed" if ordinal % 2 == 0 else "repair_requested")
    context = _replace_token(context, "failure_class", failure_class)
    # Keep all observation slots visible so the only causal change is failure
    # feedback.  The target is recomputed from the visible context.
    target = probe_target_for_context(context)
    clone = copy.deepcopy(row)
    clone["record_id"] = f"pg316-repair-{str(row.get('record_id', 'row'))}-{failure_class}-{ordinal}"
    clone["context_tokens"] = context
    clone["target_tokens"] = target
    clone["split"] = str(row.get("split") or "train")
    clone["training_eligible"] = clone["split"] == "train"
    clone["counterfactual_group"] = f"pg316-repair-pair-{str(row.get('record_id', 'row'))}"
    clone["counterfactual_kind"] = "failure_repair_pair"
    clone["failure_class"] = failure_class
    clone["repair_expected"] = True
    clone["hard_negative"] = False
    clone["raw_payload_stored"] = False
    clone["raw_response_body_stored"] = False
    clone["record_sha256"] = _digest({key: value for key, value in clone.items() if key != "record_sha256"})
    return clone


def main() -> int:
    source = _load(SOURCE)
    source_audit = _load(SOURCE_AUDIT)
    if source_audit.get("status") != "passed":
        raise RuntimeError("PG-316 requires the PG-313 dataset audit")
    records = [copy.deepcopy(row) for row in source.get("records", [])]
    def _context_value(row: dict[str, Any], key: str) -> str:
        return next((str(token).split("=", 1)[1] for token in row.get("context_tokens", []) if str(token).startswith(f"{key}=")), "unknown")

    repair_sources = [
        row for row in records
        if row.get("split") == "train"
        and row.get("training_eligible")
        and all(_context_value(row, key) == "1" for key in ("typed_available", "replay_ready", "evidence_present", "negative_control", "fresh_reset"))
        and _context_value(row, "feedback_state") != "unknown"
    ]
    # Use a bounded but diverse source sample; the split stays attached to the
    # source so generated repair rows cannot leak into the held-out families.
    repair_sources = repair_sources[:32]
    generated: list[dict[str, Any]] = []
    for ordinal, row in enumerate(repair_sources):
        for failure_class in FAILURE_CLASSES:
            generated.append(_repair_clone(row, failure_class, ordinal))
    records.extend(generated)
    counts = Counter(str(row.get("split")) for row in records)
    repair_rows = [row for row in records if row.get("counterfactual_kind") == "failure_repair_pair"]
    repair_targets = Counter(next((str(token).split("=", 1)[1] for token in row.get("target_tokens", []) if str(token).startswith("next_action=")), "none") for row in repair_rows)
    dataset = {
        "schema_version": "pg316-failure-repair-dataset-v1",
        "source": {"dataset": str(SOURCE.relative_to(ROOT)), "dataset_sha256": source.get("dataset_sha256"), "audit": str(SOURCE_AUDIT.relative_to(ROOT)), "audit_sha256": source_audit.get("audit_sha256")},
        "records": records,
        "counts": {"total": len(records), "train": counts.get("train", 0), "implementation_holdout": counts.get("implementation_holdout", 0), "real_live_holdout": counts.get("real_live_holdout", 0), "hard_negative_eval": counts.get("hard_negative_eval", 0), "repair_rows": len(repair_rows), "repair_train_rows": sum(int(row.get("split") == "train") for row in repair_rows), "repair_holdout_rows": sum(int(row.get("split") != "train") for row in repair_rows), "repair_target_next_action": dict(repair_targets), "source_repair_pairs": len(repair_sources)},
        "contract": {"paired_failure_repair": True, "missing_observation_rows_preserved": True, "probe_variant_abstract": True, "raw_payloads_excluded": True, "raw_response_bodies_excluded": True, "oracle_target_off_input": True, "training_promotion_allowed": False, "memory_promotion_allowed": False},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    checks = {
        "source_audit_pass": source_audit.get("status") == "passed",
        "records_present": bool(records),
        "repair_rows_present": bool(repair_rows),
        "repair_targets_are_repair": all(any(str(token) == "next_action=repair_abstract_plan" for token in row.get("target_tokens", [])) for row in repair_rows),
        "repair_targets_are_not_safe": all(any(str(token) == "safe_to_send=0" for token in row.get("target_tokens", [])) for row in repair_rows),
        "repair_variant_none": all(any(str(token) == "probe_variant_ref=none" for token in row.get("target_tokens", [])) for row in repair_rows),
        "paired_groups_present": len({str(row.get("counterfactual_group")) for row in repair_rows}) >= 8,
        "raw_payload_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
        "forbidden_context_fields_absent": not any(any(str(token).split("=", 1)[0] in {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss"} for token in row.get("context_tokens", [])) for row in records),
    }
    audit = {"schema_version": "pg316-failure-repair-dataset-audit-v1", "checks": checks, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["schema_version"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
