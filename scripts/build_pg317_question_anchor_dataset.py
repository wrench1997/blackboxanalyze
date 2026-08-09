"""Build PG-317 multi-missing-observation question anchors.

PG-316 contains single-missing-slot examples.  That is useful, but it still
allows a decoder to memorize a one-to-one mapping rather than learn the
priority rule ``ask for the first missing observation``.  PG-317 holds the
surface constant and pairs a complete observation vector with contexts in
which two slots are missing at once.  The target is derived only from the
visible context; no route, family, payload, or response body is copied into
model-visible data.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context, target_map  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg316_failure_repair_dataset_v1.json"
SOURCE_AUDIT = RESEARCH / "pg316_failure_repair_dataset_audit_v1.json"
OUTPUT = RESEARCH / "pg317_question_anchor_dataset_v1.json"
AUDIT = RESEARCH / "pg317_question_anchor_dataset_audit_v1.json"

# Two missing observations are deliberately used instead of single-slot
# examples.  All combinations are generated so the model cannot infer the
# question from the surface or from a fixed missing position.
MISSING_COMBINATIONS = tuple(itertools.combinations(OBSERVATION_KEYS, 2))
TRAIN_SOURCE_LIMIT = 32
HOLDOUT_SOURCE_LIMIT = 12


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _context_value(row: dict[str, Any], key: str) -> str:
    prefix = f"{key}="
    return next((str(token).split("=", 1)[1] for token in row.get("context_tokens", []) if str(token).startswith(prefix)), "unknown")


def _replace_token(tokens: list[str], key: str, value: str) -> list[str]:
    output = [token for token in tokens if not str(token).startswith(f"{key}=")]
    marker = next((index for index, token in enumerate(output) if str(token) == "[EOS]"), len(output))
    output.insert(marker, f"{key}={value}")
    return output


def _complete_source(row: dict[str, Any]) -> bool:
    values = {key: _context_value(row, key) for key in OBSERVATION_KEYS}
    return (
        all(values[key] == "1" for key in OBSERVATION_KEYS if key != "feedback_state")
        and values["feedback_state"] != "unknown"
        and _context_value(row, "history_action") == "none"
        and _context_value(row, "failure_class") == "none"
        and _context_value(row, "surface_method") in {"GET", "POST"}
        and _context_value(row, "surface_field_role") in {"query_param", "form_field", "header_value", "path_segment"}
        and _context_value(row, "surface_encoding") in {"url_percent", "form_urlencoded", "json_string", "base64_marker", "identity"}
        and row.get("counterfactual_kind") != "failure_repair_pair"
    )


def _signature(row: dict[str, Any]) -> tuple[str, ...]:
    # This signature intentionally uses only the abstract surface and not a
    # route/family identifier, so repeated template rows do not over-weight a
    # single surface.
    return tuple(
        f"{key}={_context_value(row, key)}"
        for key in ("surface_method", "surface_field_role", "surface_encoding", "feedback_state")
    )


def _anchor_clone(row: dict[str, Any], *, role: str, missing_slots: tuple[str, ...], ordinal: int) -> dict[str, Any]:
    context = canonical_assembly_context(list(row.get("context_tokens") or []))
    for key in missing_slots:
        context = _replace_token(context, key, "unknown")
    target = probe_target_for_context(context)
    values = target_map(target)
    source_id = str(row.get("record_id") or "source")
    suffix = "complete" if role == "complete" else "missing-" + "-".join(missing_slots)
    clone = copy.deepcopy(row)
    clone.update(
        {
            "schema_version": "pg317-question-anchor-v1",
            "record_id": f"pg317-anchor-{source_id}-{suffix}-{ordinal}",
            "context_tokens": context,
            "target_tokens": target,
            # The complete row and all multi-missing variants from one source
            # share one pair group; this is the audit-visible counterfactual
            # relation, while record_id remains unique per variant.
            "counterfactual_group": f"pg317-anchor-pair-{source_id}",
            "counterfactual_kind": "ask_complete_pair",
            "anchor_role": role,
            "missing_slots": list(missing_slots),
            "ask_expected": values.get("question") if role == "ask" else "none",
            "training_eligible": str(row.get("split")) == "train",
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "source_record_id": source_id,
        }
    )
    clone["record_sha256"] = _digest({key: value for key, value in clone.items() if key != "record_sha256"})
    return clone


def main() -> int:
    source = _load(SOURCE)
    source_audit = _load(SOURCE_AUDIT)
    if source_audit.get("status") != "passed":
        raise RuntimeError("PG-317 requires the PG-316 dataset audit")
    records = [copy.deepcopy(row) for row in source.get("records", [])]
    candidates = [row for row in records if _complete_source(row)]

    # Keep only the first row for each abstract surface signature.  This
    # prevents the extra anchors from being a disguised duplicate of a single
    # implementation while retaining a bounded local experiment.
    selected: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": []}
    # The source corpus has a small number of abstract surface combinations.
    # Keep several independent trace rows per combination (rather than
    # pretending four surfaces are four implementations), while bounding each
    # combination so one template cannot dominate the anchors.
    signature_counts: dict[str, Counter[tuple[str, ...]]] = {"train": Counter(), "holdout": Counter()}
    for row in candidates:
        split = "train" if row.get("split") == "train" and row.get("training_eligible") else "holdout" if row.get("split") in {"implementation_holdout", "real_live_holdout"} else ""
        signature = _signature(row)
        if not split or signature_counts[split][signature] >= 8:
            continue
        limit = TRAIN_SOURCE_LIMIT if split == "train" else HOLDOUT_SOURCE_LIMIT
        if len(selected[split]) >= limit:
            continue
        signature_counts[split][signature] += 1
        selected[split].append(row)

    if len(selected["train"]) < 8 or len(selected["holdout"]) < 4:
        raise RuntimeError(f"PG-317 needs diverse complete sources, got train={len(selected['train'])} holdout={len(selected['holdout'])}")

    generated: list[dict[str, Any]] = []
    ordinal = 0
    for split in ("train", "holdout"):
        for row in selected[split]:
            generated.append(_anchor_clone(row, role="complete", missing_slots=(), ordinal=ordinal))
            ordinal += 1
            for missing_slots in MISSING_COMBINATIONS:
                generated.append(_anchor_clone(row, role="ask", missing_slots=missing_slots, ordinal=ordinal))
                ordinal += 1
    records.extend(generated)

    anchor_rows = [row for row in records if row.get("counterfactual_kind") == "ask_complete_pair"]
    ask_rows = [row for row in anchor_rows if row.get("anchor_role") == "ask"]
    complete_rows = [row for row in anchor_rows if row.get("anchor_role") == "complete"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in anchor_rows:
        groups.setdefault(str(row.get("counterfactual_group")), []).append(row)
    counts = Counter(str(row.get("split")) for row in records)
    question_counts = Counter(str(target_map(row.get("target_tokens") or []).get("question")) for row in ask_rows)
    dataset = {
        "schema_version": "pg317-question-anchor-dataset-v1",
        "source": {"dataset": str(SOURCE.relative_to(ROOT)), "dataset_sha256": source.get("dataset_sha256"), "audit": str(SOURCE_AUDIT.relative_to(ROOT)), "audit_sha256": source_audit.get("audit_sha256")},
        "records": records,
        "counts": {
            "total": len(records),
            "train": counts.get("train", 0),
            "implementation_holdout": counts.get("implementation_holdout", 0),
            "real_live_holdout": counts.get("real_live_holdout", 0),
            "hard_negative_eval": counts.get("hard_negative_eval", 0),
            "anchor_rows": len(anchor_rows),
            "ask_rows": len(ask_rows),
            "complete_rows": len(complete_rows),
            "ask_train_rows": sum(int(row.get("split") == "train") for row in ask_rows),
            "ask_holdout_rows": sum(int(row.get("split") != "train") for row in ask_rows),
            "question_distribution": dict(question_counts),
            "train_anchor_groups": sum(int(any(item.get("split") == "train" for item in rows)) for rows in groups.values()),
            "holdout_anchor_groups": sum(int(any(item.get("split") != "train" for item in rows)) for rows in groups.values()),
        },
        "contract": {
            "multi_missing_observation_pairs": True,
            "complete_surface_pair": True,
            "question_is_derived_from_visible_priority": True,
            "same_surface_held_constant": True,
            "raw_payloads_excluded": True,
            "raw_response_bodies_excluded": True,
            "oracle_target_off_input": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)

    forbidden_keys = {"payload", "url", "route", "family", "response", "response_body", "source_code", "sql", "xss", "xxe"}
    anchor_groups_paired = all(any(item.get("anchor_role") == "complete" for item in items) and any(item.get("anchor_role") == "ask" for item in items) for items in groups.values())
    checks = {
        "source_audit_pass": source_audit.get("status") == "passed",
        "records_present": bool(records),
        "train_present": counts.get("train", 0) > 0,
        "implementation_holdout_present": counts.get("implementation_holdout", 0) > 0,
        "hard_negative_present": counts.get("hard_negative_eval", 0) > 0,
        "multi_missing_ask_rows": bool(ask_rows) and all(len(row.get("missing_slots") or []) >= 2 for row in ask_rows),
        "ask_targets_request_observation": all(target_map(row.get("target_tokens") or {}).get("next_action") == "request_observation" for row in ask_rows),
        "ask_targets_safe_zero": all(target_map(row.get("target_tokens") or {}).get("safe_to_send") == "0" for row in ask_rows),
        "ask_targets_variant_none": all(target_map(row.get("target_tokens") or {}).get("probe_variant_ref") == "none" for row in ask_rows),
        "complete_targets_assemble": bool(complete_rows) and all(target_map(row.get("target_tokens") or {}).get("next_action") == "assemble_abstract_plan" and target_map(row.get("target_tokens") or {}).get("safe_to_send") == "1" for row in complete_rows),
        "paired_groups": anchor_groups_paired,
        "holdout_anchor_present": any(row.get("split") != "train" for row in ask_rows),
        "training_not_leaking_holdout": all(bool(row.get("training_eligible")) == (row.get("split") == "train") for row in anchor_rows),
        "raw_payload_excluded": all(not row.get("raw_payload_stored") and not row.get("raw_response_body_stored") for row in records),
        "forbidden_context_fields_absent": not any(any(str(token).split("=", 1)[0] in forbidden_keys for token in row.get("context_tokens", [])) for row in records),
    }
    audit = {"schema_version": "pg317-question-anchor-dataset-audit-v1", "checks": checks, "status": "passed" if all(checks.values()) else "failed", "audit_sha256": ""}
    audit["audit_sha256"] = _digest(audit)
    OUTPUT.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": dataset["schema_version"], "counts": dataset["counts"], "audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
