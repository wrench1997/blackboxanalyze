"""Audit PG-338 axis coverage, entropy and context-target alignment.

This is a read-only information audit.  It does not start a target, send a
probe or authorize training/promotion.  Missing axes are reported explicitly;
they are never filled with defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "research" / "pg338_information_preserving_process_token_v1.json"
OUTPUT = ROOT / "research" / "pg338_information_preserving_process_audit_v1.json"
SCHEMA = "pg338-information-preserving-process-token-v1"
AXIS_KEYS = {
    "document_structure": ("document_presence", "doc_doctype", "dom_element_count"),
    "navigation": ("navigation_presence", "nav_link_count", "link_method", "nav_query_key_count"),
    "request_transport": ("request_transport_presence", "request_method", "request_placement", "request_encoding_chain", "request_transport_field_parameter_role"),
    "response_transport": ("response_transport_presence", "response_status_class", "response_content_type_class", "response_body_length", "response_redirect_hop_count"),
    "javascript_surface": ("javascript_presence", "js_script_count", "js_event_handler_count", "js_sink_category", "js_ast_node_count"),
    "failure_feedback": ("failure_feedback_presence", "failure_failure_class", "failure_previous_action", "failure_next_action", "failure_repair_delta_axis"),
    "belief_and_replay": ("belief_replay_presence", "belief_belief_delta_axis", "belief_history_action", "belief_step_budget", "belief_probe_count"),
}
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _entropy(values: list[str]) -> dict[str, Any]:
    if not values:
        return {"status": "missing", "bits": None, "count": 0, "unique": 0, "unique_ratio": 0.0}
    counts = Counter(values)
    total = sum(counts.values())
    bits = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return {"status": "measured", "bits": round(bits, 6), "count": total, "unique": len(counts), "unique_ratio": round(len(counts) / total, 6)}


def _parse(tokens: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for token in tokens:
        if "=" not in token or token.startswith("["):
            continue
        key, value = token.split("=", 1)
        parsed.setdefault(key, []).append(value)
    return parsed


def _axis_values(rows: list[Mapping[str, Any]], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for row in rows:
        parsed = _parse([str(t) for t in row.get("context_tokens") or []])
        values.append("|".join(f"{key}={parsed[key][0] if parsed.get(key) else 'missing'}" for key in keys))
    return values


def _ablation(rows: list[Mapping[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    eligible = 0
    changed = 0
    unique_before: set[str] = set()
    unique_after: set[str] = set()
    for row in rows:
        tokens = [str(t) for t in row.get("context_tokens") or []]
        if not any(token.split("=", 1)[0] in keys for token in tokens if "=" in token):
            continue
        eligible += 1
        before = " ".join(tokens)
        after = " ".join(token for token in tokens if token.split("=", 1)[0] not in keys)
        changed += int(before != after)
        unique_before.add(before)
        unique_after.add(after)
    return {"eligible_rows": eligible, "changed_rows": changed, "changed_rate": round(changed / eligible, 6) if eligible else None, "unique_before": len(unique_before), "unique_after": len(unique_after), "status": "measured" if eligible else "missing"}


def audit(data: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    rows = [row for row in list(data.get("records") or []) if isinstance(row, Mapping)]
    if data.get("schema_version") != SCHEMA or data.get("status") != "diagnostic_only_full_axis_cross_implementation":
        failures.append("schema")
    if not rows:
        failures.append("records")
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    if not split_counts.get("train") or not split_counts.get("implementation_holdout"):
        failures.append("split_missing")
    if data.get("source", {}).get("independent_implementation_holdout") is not True:
        failures.append("implementation_holdout")
    axis_report: dict[str, Any] = {}
    for axis, keys in AXIS_KEYS.items():
        missing = 0
        for row in rows:
            parsed = _parse([str(t) for t in row.get("context_tokens") or []])
            if not all(key in parsed for key in keys[:1]):
                missing += 1
        if missing:
            failures.append(f"axis_missing:{axis}")
        axis_report[axis] = {
            "required_signal_keys": list(keys),
            "missing_presence_rows": missing,
            "coverage": round((len(rows) - missing) / max(len(rows), 1), 6),
            "entropy": _entropy(_axis_values(rows, keys)),
            "field_ablation": _ablation(rows, keys),
        }
    forbidden: list[dict[str, str]] = []
    alignment_failures = 0
    for index, row in enumerate(rows):
        context = [str(t) for t in row.get("context_tokens") or []]
        for token in context:
            if any(marker in token.casefold() for marker in FORBIDDEN):
                forbidden.append({"index": str(index), "token": token})
        if row.get("context_firewall") != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.append(f"row_{index}_firewall_metadata")
        if row.get("raw_payload_stored") is not False or row.get("raw_response_body_stored") is not False or row.get("oracle_answer_in_context") is not False:
            failures.append(f"row_{index}_raw_flag")
        parsed = _parse(context)
        target = [str(t) for t in row.get("target_tokens") or []]
        refs = {token.split("=", 1)[1] for token in target if token.startswith(("transport_ref=", "field_role_ref=", "encoding_ref="))}
        aliases = {"request_method": {"request_method", "request_transport_field_method"}, "parameter_role": {"parameter_role", "request_transport_field_parameter_role"}, "encoding_chain": {"encoding_chain", "request_encoding_chain", "request_transport_field_encoding_chain"}}
        if any(not (aliases.get(ref, {ref}) & set(parsed)) for ref in refs):
            alignment_failures += 1
    if forbidden:
        failures.append("context_firewall")
    if alignment_failures:
        failures.append("context_target_alignment")
    source_impl = Counter(str(row.get("process_metadata", {}).get("source_track", "missing")) for row in rows)
    context_lengths = [len(list(row.get("context_tokens") or [])) for row in rows]
    context_sequences = {" ".join(str(t) for t in row.get("context_tokens") or []) for row in rows}
    target_sequences = {" ".join(str(t) for t in row.get("target_tokens") or []) for row in rows}
    result: dict[str, Any] = {
        "schema_version": "pg338-information-preserving-process-audit-v1",
        "status": "diagnostic_only" if not failures else "blocked",
        "dataset_sha256": str(data.get("dataset_sha256", "")),
        "record_count": len(rows),
        "split_counts": dict(split_counts),
        "implementation_counts": dict(source_impl),
        "context_length": {"min": min(context_lengths) if context_lengths else 0, "max": max(context_lengths) if context_lengths else 0, "mean": round(sum(context_lengths) / len(context_lengths), 3) if context_lengths else 0.0},
        "unique_context_sequences": len(context_sequences),
        "unique_target_sequences": len(target_sequences),
        "context_token_entropy_bits": _entropy([str(t) for row in rows for t in row.get("context_tokens") or []]),
        "target_token_entropy_bits": _entropy([str(t) for row in rows for t in row.get("target_tokens") or []]),
        "axis_entropy": axis_report,
        "context_target_alignment": {"failed_rows": alignment_failures, "total_rows": len(rows), "rate": round((len(rows) - alignment_failures) / max(len(rows), 1), 6)},
        "context_firewall": {"forbidden_token_count": len(forbidden), "examples": forbidden[:5]},
        "information_gate": {"all_axes_present": not any(item.startswith("axis_missing:") for item in failures), "full_axis_rows": int(data.get("counts", {}).get("full_axis_rows", 0) or 0), "relative_entropy_gate": "diagnostic_only_not_yet_paired", "promotion_threshold": "no axis/holdout predictive entropy collapse >25%"},
        "failures": sorted(set(failures)),
        "scientific_gate": {"status": "blocked", "accepted_training_rows": 0, "reason": "operator review and multi-seed field/entropy ablation still required"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-338 full-axis process tokens")
    parser.add_argument("--dataset", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    result = audit(data)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.json else None))
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
