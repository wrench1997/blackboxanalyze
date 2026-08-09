"""Build PG-287 evidence-conditioned plan data from PG-285.

PG-285 exposed a real identifiability collision: identical contexts map to
``encoding=url_percent`` and ``encoding=unknown``.  PG-287 teaches the model
to ask for typed encoding evidence when the slot is not observable, and only
emit the original bounded wire plan when an explicit ``encoding_observed``
token is present.  It never adds a literal payload or an oracle label to the
context.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg285_payload_grounding_dataset_v1.json"
SOURCE_HARD = RESEARCH / "pg285_payload_grounding_hard_negative_v1.json"
DATASET = RESEARCH / "pg287_identifiability_dataset_v1.json"
AUDIT = RESEARCH / "pg287_identifiability_dataset_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _ask_target(row: dict[str, Any]) -> list[str]:
    target = dict(row.get("target") or {})
    method = str(target.get("method", row.get("method", "GET")))
    return [
        "[TARGET_BOS]",
        "plan=ask_typed",
        f"method={method}",
        "probe_class=other",
        "channel=unknown",
        "encoding=unknown",
        "wire=none",
        "field_slot=unknown",
        "repair_delta=none",
        "family_agnostic=1",
        "final_action=ask_typed",
        "safe_to_send=0",
        "[TARGET_EOS]",
    ]


def _resolved_target(row: dict[str, Any]) -> list[str]:
    return [str(token) for token in list(row.get("target_tokens") or [])]


def _context(row: dict[str, Any], *, observed: str) -> list[str]:
    source = [str(token) for token in list(row.get("context_tokens") or [])]
    if "[CTX_END]" in source:
        source = [token for token in source if token != "[CTX_END]"]
    source.extend([f"encoding_observed={observed}", f"observation_sufficiency={'resolved' if observed != 'unknown' else 'ambiguous'}", "[CTX_END]"])
    return source


def _make_row(row: dict[str, Any], *, variant: str, observed: str, target_tokens: list[str], suffix: str) -> dict[str, Any]:
    out = {
        "record_id": f"pg287:{row.get('record_id', 'unknown')}:{suffix}",
        "source_record_id": str(row.get("record_id", "")),
        "source_group_id": str(row.get("source_group_id", row.get("record_id", ""))),
        "source": "pg285_identifiability_counterfactual",
        "family": str(row.get("family", "hidden")),
        "method": str(row.get("method", "GET")),
        "split": str(row.get("split", "hard_negative")),
        "variant": variant,
        "context_tokens": _context(row, observed=observed),
        "target_tokens": target_tokens,
        "target": {
            "next_action": "ask_typed" if variant == "ambiguous" else str(row.get("target", {}).get("next_action", "candidate_probe")),
            "method": str(row.get("target", {}).get("method", row.get("method", "GET"))),
            "probe_class": "other" if variant == "ambiguous" else str(row.get("target", {}).get("probe_class", "other")),
            "channel": "unknown" if variant == "ambiguous" else str(row.get("target", {}).get("channel", "unknown")),
            "encoding": "unknown" if variant == "ambiguous" else str(row.get("target", {}).get("encoding", "unknown")),
            "wire_kind": "none" if variant == "ambiguous" else str(row.get("target", {}).get("wire_kind", "none")),
            "safe_to_send": False if variant == "ambiguous" else bool(row.get("target", {}).get("safe_to_send", False)),
            "oracle_required": True,
        },
        "hard_negative": bool(row.get("hard_negative", False)),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_label_in_context": False,
        "literal_probe_in_context": False,
        "training_eligible": not bool(row.get("hard_negative", False)),
        "memory_promotion_allowed": False,
        "source_evidence_hash": str(row.get("source_evidence_hash", "")),
    }
    out["record_sha256"] = _digest(out)
    return out


def main() -> None:
    source = _load(SOURCE)
    hard_source = _load(SOURCE_HARD)
    records: list[dict[str, Any]] = []
    for row in list(source.get("records") or []):
        encoding = str(row.get("target", {}).get("encoding", "unknown"))
        # Unknown remains deliberately ambiguous; concrete encoding variants
        # get a resolved counterfactual to test evidence-conditioned decoding.
        records.append(_make_row(row, variant="ambiguous", observed="unknown", target_tokens=_ask_target(row), suffix="ambiguous"))
        if encoding in {"plain", "url_percent"}:
            records.append(_make_row(row, variant="resolved", observed=encoding, target_tokens=_resolved_target(row), suffix="resolved"))
    hard_rows: list[dict[str, Any]] = []
    for row in list(hard_source.get("records") or []):
        hard_rows.append(_make_row(row, variant="ambiguous", observed="unknown", target_tokens=_ask_target(row), suffix="hard"))
    counts = {
        "train": sum(row["split"] == "train" for row in records),
        "route_dev": sum(row["split"] == "route_dev" for row in records),
        "family_holdout": sum(row["split"] == "family_holdout" for row in records),
        "hard_negative": len(hard_rows),
        "total": len(records) + len(hard_rows),
        "ambiguous": sum(row["variant"] == "ambiguous" for row in records),
        "resolved": sum(row["variant"] == "resolved" for row in records),
    }
    split_variant_counts = {
        split: {
            "ambiguous": sum(row["split"] == split and row["variant"] == "ambiguous" for row in records),
            "resolved": sum(row["split"] == split and row["variant"] == "resolved" for row in records),
        }
        for split in ("train", "route_dev", "family_holdout")
    }
    dataset = {
        "schema_version": "pg287-identifiability-dataset-v1",
        "purpose": "evidence-conditioned ask_typed versus resolved abstract wire-plan decoding",
        "source": {"dataset": str(SOURCE.relative_to(ROOT).as_posix()), "dataset_sha256": str(source.get("dataset_sha256", "")), "hard_negative": str(SOURCE_HARD.relative_to(ROOT).as_posix()), "hard_negative_sha256": str(hard_source.get("dataset_sha256", ""))},
        "records": records,
        "hard_negative_records": hard_rows,
        "counts": counts,
        "coverage": {"split_variant_counts": split_variant_counts, "family_holdout_resolved_coverage": split_variant_counts["family_holdout"]["resolved"] > 0},
        "training_contract": {
            "family_hidden_in_context": True,
            "oracle_label_in_context": False,
            "literal_probe_values_out_of_context": True,
            "raw_response_bodies_out_of_context": True,
            "ambiguous_rows_train_ask_typed": True,
            "hard_negative_training_eligible": False,
            "remote_a800_required": True,
            "live_replay_required_for_promotion": True,
            "memory_promotion_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    all_rows = [*records, *hard_rows]
    forbidden = ("family=", "oracle=", "typed_effect=", "positive=", "payload=", "literal=", "<script", "javascript:", "union select", "drop table")
    checks = {
        "source_rows_present": len(records) > 0,
        "counts_match": counts["total"] == len(all_rows),
        "context_family_hidden": all("family_hidden=1" in row["context_tokens"] and "family=" not in " ".join(row["context_tokens"]) for row in all_rows),
        "context_no_labels_or_literals": all(not row["oracle_label_in_context"] and not any(any(bad.casefold() in token.casefold() for bad in forbidden) for token in row["context_tokens"]) for row in all_rows),
        "ambiguous_rows_ask": all(row["target"]["next_action"] == "ask_typed" and row["target"]["encoding"] == "unknown" and row["target"]["safe_to_send"] is False for row in records if row["variant"] == "ambiguous"),
        "resolved_rows_have_observed_encoding": all(any(token == f"encoding_observed={row['target']['encoding']}" for token in row["context_tokens"]) and row["target"]["encoding"] != "unknown" for row in records if row["variant"] == "resolved"),
        "hard_negative_quarantined": all(row["hard_negative"] and not row["training_eligible"] and row["target"]["next_action"] == "ask_typed" for row in hard_rows),
        "raw_material_excluded": all(not row["raw_payload_strings_stored"] and not row["raw_response_bodies_stored"] for row in all_rows),
    }
    audit = {"schema_version": "pg287-identifiability-dataset-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], "counts": counts, "coverage": dataset["coverage"], "checks": checks, "status": "passed" if all(checks.values()) else "blocked", "training_eligible_rows": sum(row["training_eligible"] for row in records), "promotion_blocked": True, "coverage_gate_status": "passed" if dataset["coverage"]["family_holdout_resolved_coverage"] else "blocked", "coverage_gaps": [] if dataset["coverage"]["family_holdout_resolved_coverage"] else ["family_holdout_resolved_rows"], "interpretation": "PG-285 的编码碰撞被显式转成 ask_typed；resolved 变体只在 context 中提供 observable encoding token。family holdout 没有 resolved 行时只能报告 coverage gap，不能显示 0% 模型准确率；该数据仍源自模板，不能证明真实靶场泛化。"}
    audit["audit_sha256"] = _digest(audit)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
