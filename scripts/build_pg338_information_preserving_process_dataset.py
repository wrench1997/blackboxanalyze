"""Build the PG-338 full-axis process-token diagnostic dataset.

PG-337's old cross-implementation builder intentionally made a compact
15-token view.  That view is useful as a baseline, but it is not allowed to
stand in for the whole-page ontology.  This builder keeps the complete
abstract context from the DVWA and WebGoat source rows and only adds bounded
process metadata outside model context.  Raw wire values, payloads,
responses, route literals and evaluator answers never enter context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DVWA = RESEARCH / "pg337_dvwa_failure_repair_source_rows_v1.json"
WEBGOAT = RESEARCH / "pg333_webgoat_typed_method_shape_source_rows_v1.json"
OUTPUT = RESEARCH / "pg338_information_preserving_process_token_v1.json"

SCHEMA_VERSION = "pg338-information-preserving-process-token-v1"
EXPECTED_AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)
FORBIDDEN = (
    "family=", "implementation=", "route=", "route_literal=", "source=", "image=",
    "path=", "url=", "payload=", "payload_", "raw_", "response_body=",
    "response_body_text=", "oracle=", "evaluator=", "canary=",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _tokens(value: Any) -> list[str]:
    return [str(token) for token in list(value or [])]


def _contains_forbidden(tokens: list[str]) -> list[str]:
    return sorted({token for token in tokens if any(marker in token.casefold() for marker in FORBIDDEN)})


def _parse(tokens: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for token in tokens:
        if "=" not in token or token.startswith("["):
            continue
        key, value = token.split("=", 1)
        parsed.setdefault(key, []).append(value)
    return parsed


def _axis_presence(tokens: list[str]) -> dict[str, bool]:
    parsed = _parse(tokens)
    result: dict[str, bool] = {}
    for axis in EXPECTED_AXES:
        key = {
            "document_structure": "document_presence",
            "javascript_surface": "javascript_presence",
            "belief_and_replay": "belief_replay_presence",
        }.get(axis, f"{axis}_presence")
        result[axis] = key in parsed and bool(parsed[key])
    return result


def _next_action(target_tokens: list[str]) -> str:
    for token in target_tokens:
        if token.startswith("next_action="):
            return token.split("=", 1)[1]
    return "unknown"


def _question(target_tokens: list[str]) -> str:
    for token in target_tokens:
        if token.startswith("question="):
            return token.split("=", 1)[1]
    return "unknown"


def _kind(target_tokens: list[str]) -> str:
    action = _next_action(target_tokens)
    if action == "repair":
        return "failure_repair"
    if action == "abstain":
        return "negative_review"
    if action in {"ask_typed", "ask_missing", "ask_failure"}:
        return "ask_preflight"
    return "probe_observed"


def _sidecar_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    sidecar = dict(row.get("evaluator_sidecar") or {})
    return {
        "evidence_sha256": str(sidecar.get("evidence_hash", "")),
        "typed_available": bool(sidecar.get("typed_available", False)),
        "fresh_reset": bool(sidecar.get("fresh_reset", False)),
        "negative_control": bool(sidecar.get("negative_control", False)),
        "confirmed_positive": bool(sidecar.get("confirmed_positive", False)),
    }


def _record(row: Mapping[str, Any], *, source_name: str, new_split: str, source_split: str) -> tuple[dict[str, Any] | None, list[str]]:
    context = _tokens(row.get("context_tokens"))
    target = _tokens(row.get("target_tokens"))
    failures: list[str] = []
    if len(context) < 32:
        failures.append("context_too_short")
    if not target:
        failures.append("target_missing")
    forbidden = _contains_forbidden(context)
    if forbidden:
        failures.append("context_firewall")
    presence = _axis_presence(context)
    failures.extend(f"axis_missing:{axis}" for axis, present in presence.items() if not present)
    if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        failures.append("firewall_metadata")
    if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False or row.get("oracle_answer_in_context") is not False:
        failures.append("raw_oracle_flag")
    parsed = _parse(context)
    refs = {token.split("=", 1)[1] for token in target if token.startswith(("transport_ref=", "field_role_ref=", "encoding_ref="))}
    aliases = {
        "request_method": {"request_method", "request_transport_field_method"},
        "parameter_role": {"parameter_role", "request_transport_field_parameter_role"},
        "encoding_chain": {"encoding_chain", "request_encoding_chain", "request_transport_field_encoding_chain"},
    }
    if any(not (aliases.get(ref, {ref}) & set(parsed)) for ref in refs):
        failures.append("context_target_alignment")
    if failures:
        return None, sorted(set(failures))
    digest = str(row.get("record_sha256", ""))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg338-{_sha({'source': source_name, 'record': digest})[:24]}",
        "split": new_split,
        "source_split": source_split,
        "source_record_sha256": digest,
        "source_implementation_hash": _sha(source_name),
        "diagnostic_kind": _kind(target),
        "source_grounded": True,
        "synthetic_counterfactual": False,
        "context_tokens": context,
        "target_tokens": target,
        "target_projection": {
            "question": _question(target),
            "next_action": _next_action(target),
            "safe_to_send": any(token == "safe_to_send=1" for token in target),
        },
        "field_capture_manifest": row.get("field_capture_manifest") or {},
        "axis_presence": presence,
        "process_metadata": {
            "source_track": source_name,
            "source_split": source_split,
            "full_axis_context": True,
            "process_kind": _kind(target),
            "question_token": _question(target),
            "next_action_token": _next_action(target),
        },
        "evaluator_sidecar_ref": _sidecar_ref(row),
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    record["record_sha256"] = _sha(record)
    return record, []


def build(*, dvwa_path: Path = DVWA, webgoat_path: Path = WEBGOAT) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    sources = (("webgoat", webgoat_path, "train"), ("dvwa", dvwa_path, "implementation_holdout"))
    for source_name, path, new_split in sources:
        document = _load(path)
        for index, row in enumerate(list(document.get("records") or [])):
            if not isinstance(row, Mapping):
                failures.append({"source": source_name, "index": index, "reason": "row_not_object"})
                continue
            record, row_failures = _record(row, source_name=source_name, new_split=new_split, source_split=str(row.get("split", "unknown")))
            if record is None:
                failures.append({"source": source_name, "index": index, "reason": row_failures})
            else:
                rows.append(record)
    counts = Counter(str(row.get("diagnostic_kind")) for row in rows)
    split_counts = Counter(str(row.get("split")) for row in rows)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only_full_axis_cross_implementation" if not failures else "blocked_incomplete",
        "purpose": "Preserve whole-page seven-axis context while testing process-token generalization; no capability promotion.",
        "source": {
            "webgoat": str(webgoat_path.relative_to(ROOT)).replace("\\", "/"),
            "dvwa": str(dvwa_path.relative_to(ROOT)).replace("\\", "/"),
            "split_policy": "explicit_webgoat_train_dvwa_implementation_holdout; source_split preserved per row",
            "independent_implementation_holdout": True,
            "full_axis_required": True,
            "raw_values_in_context": False,
        },
        "records": rows,
        "counts": {
            "total": len(rows),
            "train": int(split_counts.get("train", 0)),
            "implementation_holdout": int(split_counts.get("implementation_holdout", 0)),
            "probe_observed": int(counts.get("probe_observed", 0)),
            "failure_repair": int(counts.get("failure_repair", 0)),
            "negative_review": int(counts.get("negative_review", 0)),
            "ask_preflight": int(counts.get("ask_preflight", 0)),
            "full_axis_rows": len(rows),
        },
        "failures": failures,
        "process_policy": {
            "accepted_training_rows": 0,
            "training_eligible_rows": 0,
            "operator_review_required": True,
            "context_policy": "full abstract context; evaluator sidecars and raw values off-context",
            "promotion_blocked_until": ["field_entropy_audit", "cross_implementation_holdout", "operator_review"],
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["dataset_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-338 information-preserving process tokens")
    parser.add_argument("--dvwa", type=Path, default=DVWA)
    parser.add_argument("--webgoat", type=Path, default=WEBGOAT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build(dvwa_path=args.dvwa, webgoat_path=args.webgoat)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] != "blocked_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
