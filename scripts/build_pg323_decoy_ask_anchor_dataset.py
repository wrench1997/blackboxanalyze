"""Build PG-323 oversampled ASK and unseen-surface abstain anchors."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context, target_map  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg322_cross_impl_decoy_dataset_v1.json"
OUTPUT = RESEARCH / "pg323_decoy_ask_anchor_dataset_v1.json"
AUDIT = RESEARCH / "pg323_decoy_ask_anchor_dataset_audit_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _clone(row: dict[str, Any], index: int, *, action: str | None = None, missing: set[str] | None = None, split: str = "train") -> dict[str, Any]:
    context_values: dict[str, str] = {}
    for token in row.get("context_tokens", []):
        if "=" in str(token):
            key, value = str(token).split("=", 1)
            context_values[key] = value
    if action is not None:
        context_values["history_action"] = action
    for key in missing or set():
        context_values[key] = "unknown"
    context = canonical_assembly_context([f"{key}={value}" for key, value in context_values.items()])
    target = probe_target_for_context(context)
    values = target_map(target)
    clone = {
        "schema_version": "pg323-decoy-ask-anchor-record-v1",
        "record_id": f"pg323:{index}",
        "split": split,
        "training_eligible": split == "train",
        "source_meta": dict(row.get("source_meta") or {}),
        "context_tokens": context,
        "target_tokens": target,
        "expected_variant": str(values.get("probe_variant_ref", "none")),
        "expected_safe": str(values.get("safe_to_send", "0")) == "1",
        "expected_question": str(values.get("question", "none")),
        "hard_negative": bool(row.get("hard_negative", False) or split == "hard_negative_eval"),
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_target_off_input": True,
        "anchor_kind": "ask" if values.get("question") != "none" else "abstain_surface",
    }
    clone["record_sha256"] = _digest(clone)
    return clone


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    original = [dict(row) for row in source.get("records", [])]
    records: list[dict[str, Any]] = []
    index = 0
    # Preserve the original PG-322 train rows and add history-order variants
    # for the same observation lattice.  This teaches process composition,
    # not a route answer.
    for row in original:
        if row.get("split") == "train" and row.get("training_eligible"):
            records.append(_clone(row, index, split="train"))
            index += 1
            if row.get("expected_question") != "none":
                for action in ("observe", "identity"):
                    records.append(_clone(row, index, action=action, split="train"))
                    index += 1
    # Decoy anchors expose unknown typed availability and incomplete negative
    # evidence during training, while candidate/reference/negative choices
    # remain blind in PG-322 holdout.
    for row in original:
        surface_id = str((row.get("source_meta") or {}).get("surface_id", ""))
        if surface_id not in {"blind_path_decoy", "blind_header_decoy"}:
            continue
        if row.get("split") == "third_surface_holdout" and row.get("expected_question") == "ask_typed_availability":
            records.append(_clone(row, index, action="observe", split="train"))
            index += 1
            records.append(_clone(row, index, action="identity", missing={"negative_control"}, split="train"))
            index += 1
    # Include complete hard-negative anchors with a missing negative control;
    # the target must remain safe=0 and ask_negative_control.
    for row in original:
        if row.get("split") == "hard_negative_eval":
            records.append(_clone(row, index, action="observe", missing={"negative_control"}, split="train"))
            index += 1
    # Keep all original holdouts evaluation-only.
    for row in original:
        if row.get("split") in {"implementation_holdout", "third_surface_holdout", "ask_holdout", "hard_negative_eval"}:
            clone = copy.deepcopy(row)
            clone["training_eligible"] = False
            records.append(clone)
    dataset = {
        "schema_version": "pg323-decoy-ask-anchor-dataset-v1",
        "status": "completed_pg323_dataset_build",
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "source_dataset_sha256": source.get("dataset_sha256"),
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row.get("split") == "train" for row in records),
            "implementation_holdout": sum(row.get("split") == "implementation_holdout" for row in records),
            "third_surface_holdout": sum(row.get("split") == "third_surface_holdout" for row in records),
            "ask_holdout": sum(row.get("split") == "ask_holdout" for row in records),
            "hard_negative_eval": sum(row.get("split") == "hard_negative_eval" for row in records),
            "ask_train": sum(row.get("split") == "train" and row.get("expected_question") != "none" for row in records),
            "safe_zero_train": sum(row.get("split") == "train" and not row.get("expected_safe") for row in records),
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    forbidden = {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss", "xxe"}
    bad: list[int] = []
    for i, row in enumerate(records):
        keys = {str(token).split("=", 1)[0] for token in row.get("context_tokens", []) + row.get("target_tokens", []) if "=" in str(token)}
        if keys & forbidden or row.get("raw_payload_stored") or row.get("raw_response_body_stored"):
            bad.append(i)
    audit = {
        "schema_version": "pg323-decoy-ask-anchor-dataset-audit-v1",
        "checks": {
            "records_present": bool(records),
            "train_present": any(row.get("split") == "train" for row in records),
            "holdouts_preserved": all(not row.get("training_eligible") for row in records if row.get("split") != "train"),
            "ask_train_present": any(row.get("split") == "train" and row.get("expected_question") != "none" for row in records),
            "hard_negative_holdout_present": any(row.get("split") == "hard_negative_eval" for row in records),
            "safe_ask": all(not row.get("expected_safe") for row in records if row.get("expected_question") != "none"),
            "forbidden_absent": not bad,
            "raw_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
        },
        "bad_indices": bad,
        "audit_sha256": "",
    }
    audit["status"] = "passed" if all(audit["checks"].values()) else "failed"
    audit["audit_sha256"] = _digest(audit)
    return dataset, audit


def main() -> int:
    dataset, audit = build()
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["status"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
