"""Build PG-335: source-grounded process-token diagnostics.

The input is the real, typed PG-333 source-row collection.  It already has
complete seven-axis observations but does not contain enough missing/failure
transitions to teach a question-driven policy.  We therefore create explicit,
source-grounded counterfactual masks (one axis at a time), plus evaluator
negative-review and failure-repair diagnostics.  These rows remain
diagnostic-only; masking is not a new oracle label and never becomes gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg333_three_impl_get_post_diagnostic_source_rows_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg335_real_process_token_diagnostic_v1.json"

AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_replay",
)
TOKEN_AXIS = {"belief_replay": "belief_and_replay"}
FORBIDDEN_PREFIXES = (
    "family=", "implementation=", "route=", "route_literal=", "source=",
    "image=", "record=", "path=", "url=", "payload=", "payload_",
    "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=",
)
SAFE_VALUES = {
    "GET", "POST", "query", "form", "json", "multipart", "plain", "url_percent",
    "one", "two", "few", "many", "zero", "short", "medium", "long", "empty",
    "present", "absent", "none", "true", "false", "observed", "not_observed", "unknown",
    "html", "json_value", "object", "array", "text", "header", "attribute", "html_text",
    "parameter", "query_parameter", "form_parameter", "body", "cookie", "csrf", "same_origin",
    "source_attested_candidate", "reference", "negative_control", "result_shape", "none",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _normalize_token(token: Any) -> str | None:
    value = str(token)
    folded = value.casefold()
    if any(folded.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return None
    if value.startswith("[") and value.endswith("]"):
        return value
    if value.startswith("data_"):
        return "data_marker=abstract"
    if value.startswith("chunk_digest="):
        return "chunk_digest=bucket"
    if "=" not in value:
        # Field names such as data_pg332_* are handled above; opaque literals
        # are dropped rather than smuggled into the model vocabulary.
        return value if re.fullmatch(r"(?:axis_(?:begin|end)|chunk_boundary|chunk_index|chunk_count|chunk_shape)=[A-Za-z0-9_>-]+", value) else None
    key, raw = value.split("=", 1)
    if key.casefold() in {"family", "implementation", "route", "source", "image", "path", "url", "payload", "oracle", "evaluator"}:
        return None
    raw = raw.strip('"')
    if raw in SAFE_VALUES or raw.startswith("bucket") or raw.startswith(">") or raw.isdigit():
        return f"{key}={raw}"
    # Keep the ontology field key while hiding implementation-specific value
    # names (alpha/beta, digest fragments, route labels, etc.).
    return f"{key}=abstract"


def abstract_context(tokens: list[Any], *, missing_axis: str | None = None) -> list[str]:
    target_axis = TOKEN_AXIS.get(missing_axis or "", missing_axis)
    result: list[str] = []
    current: str | None = None
    for token in tokens:
        text = str(token)
        if text.startswith("axis_begin="):
            current = text.split("=", 1)[1]
            if current == target_axis:
                result.append(f"axis_begin={current}")
                result.append(f"axis_presence={missing_axis}=not_observed")
                continue
        if text.startswith("axis_end="):
            axis = text.split("=", 1)[1]
            if axis == target_axis:
                result.append(f"axis_end={axis}")
                current = None
                continue
            current = None
        if target_axis and current == target_axis:
            continue
        normalized = _normalize_token(token)
        if normalized is not None:
            result.append(normalized)
    if missing_axis and f"axis_missing={missing_axis}" not in result:
        result.insert(1 if result and result[0] == "[BOS]" else 0, f"axis_missing={missing_axis}")
    return list(dict.fromkeys(result))


def _manifest(row: Mapping[str, Any], *, missing_axis: str | None = None) -> dict[str, dict[str, str]]:
    original = row.get("field_capture_manifest") if isinstance(row.get("field_capture_manifest"), Mapping) else {}
    result: dict[str, dict[str, str]] = {}
    for axis in AXES:
        values = dict(original.get(axis) or {})
        if not values:
            values = {"presence": "observed"}
        if axis == missing_axis:
            values = {str(field): "not_observed" for field in values}
        result[axis] = values
    return result


def _target(kind: str, original: list[Any] | None = None) -> list[str]:
    if kind == "observed":
        safe = []
        for token in list(original or []):
            normalized = _normalize_token(token)
            if normalized is not None and (normalized.startswith("[TARGET_") or normalized.startswith(("question=", "next_action=", "repair_action=", "transport_ref=", "field_role_ref=", "encoding_ref=", "probe_variant_ref=", "safe_to_send="))):
                safe.append(normalized)
        return list(dict.fromkeys(safe)) or ["[TARGET_BOS]", "question=none", "next_action=abstain", "safe_to_send=0", "[TARGET_EOS]"]
    if kind == "ask":
        return ["[TARGET_BOS]", "question=ask_missing_observation", "next_action=ask_typed", "repair_action=observe", "failure_class=missing_observation", "action_changed=1", "safe_to_send=0", "[TARGET_EOS]"]
    if kind == "failure":
        return ["[TARGET_BOS]", "question=ask_failure", "next_action=repair", "repair_action=observe", "failure_class=environment_response_mismatch", "action_changed=1", "safe_to_send=0", "[TARGET_EOS]"]
    return ["[TARGET_BOS]", "question=review_negative", "next_action=abstain", "repair_action=none", "failure_class=none", "action_changed=1", "safe_to_send=0", "[TARGET_EOS]"]


def _row_base(row: Mapping[str, Any], *, source_index: int, kind: str, missing_axis: str | None) -> dict[str, Any]:
    source_hash = digest({"record_id": row.get("record_id"), "record_sha256": row.get("record_sha256"), "source_meta": row.get("source_meta")})
    split = "train" if str(row.get("split")) == "train" else "implementation_holdout"
    identity = digest({"source": source_hash, "kind": kind, "axis": missing_axis})
    return {
        "schema_version": "pg335-real-process-token-row-v1",
        "record_id": "pg335:" + identity,
        "paired_id": "pg335:" + digest({"source": source_hash, "axis": missing_axis or "none"}),
        "split": split,
        "source_row_index": source_index,
        "diagnostic_kind": kind,
        "axis_mask": missing_axis,
        "source_grounded": True,
        "synthetic_counterfactual": kind in {"ask", "failure"},
        "source_row_digest": source_hash,
    }


def _make_row(row: Mapping[str, Any], *, source_index: int, kind: str, missing_axis: str | None = None) -> dict[str, Any]:
    base = _row_base(row, source_index=source_index, kind=kind, missing_axis=missing_axis)
    context = abstract_context(list(row.get("context_tokens") or []), missing_axis=missing_axis)
    negative = str(row.get("record_id", "")).endswith(":negative") or "negative_control" in list(row.get("target_tokens") or [])
    target = _target(kind, list(row.get("target_tokens") or []))
    result = {
        **base,
        "context_tokens": context,
        "target_tokens": target,
        "target_projection": {
            "question": "none" if kind == "observed" else ("ask_missing_observation" if kind == "ask" else "ask_failure" if kind == "failure" else "review_negative"),
            "next_action": "send_probe" if kind == "observed" else ("ask_typed" if kind == "ask" else "repair" if kind == "failure" else "abstain"),
            "repair_action": "none" if kind in {"observed", "negative_review"} else "observe",
            "safe_to_send": kind == "observed",
            "action_changed": kind != "observed",
        },
        "process_metadata": {
            "negative_control": negative,
            "failure_class": "none" if kind in {"observed", "negative_review"} else "missing_observation" if kind == "ask" else "environment_response_mismatch",
            "source_evaluator_typed": bool(dict(row.get("evaluator_sidecar") or {}).get("typed_available")),
            "source_fresh_reset": bool(dict(row.get("evaluator_sidecar") or {}).get("fresh_reset")),
        },
        "field_capture_manifest": _manifest(row, missing_axis=missing_axis),
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "source": {"kind": "pg333_real_typed_source_row", "source_row_hash": base["source_row_digest"], "evidence_present": bool(dict(row.get("evaluator_sidecar") or {}).get("evidence_hash"))},
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    return result


def build_dataset(source: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = [row for row in list(source.get("records") or []) if isinstance(row, Mapping)]
    records: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        records.append(_make_row(row, source_index=index, kind="observed"))
        for axis in AXES:
            records.append(_make_row(row, source_index=index, kind="ask", missing_axis=axis))
        target_text = " ".join(str(item) for item in list(row.get("target_tokens") or []))
        if "negative_control" in target_text:
            records.append(_make_row(row, source_index=index, kind="negative_review"))
        if str(row.get("record_id", "")).endswith((":candidate", "-candidate")):
            records.append(_make_row(row, source_index=index, kind="failure"))
    context_tokens = sorted({token for row in records for token in row["context_tokens"]})
    target_tokens = sorted({token for row in records for token in row["target_tokens"]})
    counts = Counter(str(row["diagnostic_kind"]) for row in records)
    axis_masks = Counter(str(row["axis_mask"]) for row in records if row.get("axis_mask"))
    payload = {
        "schema_version": "pg335-real-process-token-diagnostic-v1",
        "purpose": "source-grounded ASK/failure/negative process-token diagnostics without raw wire or evaluator answers",
        "source": {"dataset": "research/pg333_three_impl_get_post_diagnostic_source_rows_v1.json", "source_rows": len(source_rows), "real_source_rows": len(source_rows), "synthetic_counterfactual_rows": sum(bool(row["synthetic_counterfactual"]) for row in records), "real_gold_rows": 0},
        "records": records,
        "counts": {"total": len(records), "source_rows": len(source_rows), "observed": counts.get("observed", 0), "ask": counts.get("ask", 0), "failure": counts.get("failure", 0), "negative_review": counts.get("negative_review", 0), "train": sum(row["split"] == "train" for row in records), "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in records), "axis_masks": dict(axis_masks)},
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "process_policy": {"mask_rows_are_diagnostic": True, "failure_rows_are_synthetic_diagnostic": True, "real_negative_review_rows_are_evaluator_grounded": True, "preserve_axis_identity": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    payload["dataset_sha256"] = digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-335 source-grounded process-token diagnostics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_dataset(json.loads(args.input.read_text(encoding="utf-8-sig")))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "diagnostic_only", "records": len(result["records"]), "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
