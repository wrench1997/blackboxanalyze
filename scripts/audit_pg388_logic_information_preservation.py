"""Read-only information-preservation audit for PG-388 logic source rows.

The audit measures whether the two local implementations carry useful
abstract context diversity without copying rows, tokens, wire data, payloads,
response bodies or evaluator answers into its report.  It is diagnostic only:
the current PG-388 source collection has no approved train split, so this
script can never grant training or promotion permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_ROWS = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_rows_v1.json"
DEFAULT_B_ROWS = ROOT / "research" / "pg388_logic_holdout_b_source_rows_rows_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_information_preservation_audit_v1.json"

SCHEMA_VERSION = "pg388-logic-information-preservation-audit-v1"
FORBIDDEN_KEYS = {
    "rows",
    "context_tokens",
    "target_tokens",
    "logic_context_tokens",
    "logic_rule_ir_target_tokens",
    "payload",
    "wire",
    "response_body",
    "oracle_answer",
    "evaluator_answer",
}
PROMOTION = {
    "training_allowed": False,
    "memory_promotion_allowed": False,
    "payload_catalog_promotion_allowed": False,
    "vulnerability_claim_allowed": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[Mapping[str, Any]]:
    document = _load(path)
    values = document.get("rows")
    if not isinstance(values, list):
        raise ValueError(f"source-row wrapper list missing: {path}")
    result: list[Mapping[str, Any]] = []
    for wrapper in values:
        if not isinstance(wrapper, Mapping):
            raise ValueError("source-row wrapper must be an object")
        row = wrapper.get("source_row")
        if not isinstance(row, Mapping):
            raise ValueError("source-row wrapper missing source_row")
        result.append(row)
    return result


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _sequence_summary(sequences: list[list[str]]) -> dict[str, Any]:
    if not sequences:
        return {"count": 0, "unique": 0, "unique_ratio": 0.0, "length_min": 0, "length_max": 0, "length_mean": 0.0}
    signatures = {_digest(sequence) for sequence in sequences}
    lengths = [len(sequence) for sequence in sequences]
    return {
        "count": len(sequences),
        "unique": len(signatures),
        "unique_ratio": round(len(signatures) / len(sequences), 6),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": round(sum(lengths) / len(lengths), 3),
    }


def _axis_summary(rows: list[Mapping[str, Any]], axis: str) -> dict[str, Any]:
    presence: list[str] = []
    unique_rows: set[str] = set()
    for row in rows:
        value = row.get("axis_presence")
        if isinstance(value, Mapping):
            presence.append(str(value.get(axis, "unknown")))
        else:
            presence.append("unknown")
        unique_rows.add(_digest(value if isinstance(value, Mapping) else {}))
    counts = Counter(presence)
    return {
        "row_count": len(rows),
        "unique_presence_states": len(counts),
        "presence_entropy_bits": _entropy(presence),
        "observed_count": int(counts.get("observed", 0)),
        "absent_count": int(counts.get("absent", 0)),
        "not_observed_count": int(counts.get("not_observed", 0)),
        "unknown_count": int(counts.get("unknown", 0)),
        "row_shape_groups": len(unique_rows),
    }


def _prefix_counts(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize token-prefix diversity without returning token names/values."""
    values_by_prefix: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        tokens = row.get("context_tokens")
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            if not isinstance(token, str) or "=" not in token:
                continue
            prefix = token.split("=", 1)[0]
            # Prefixes are ontology coordinates, not values.  Keep only a
            # bounded allowlist so an accidental literal cannot enter output.
            if prefix.startswith(("chunk_", "document_", "doc_", "dom_", "element_", "text_", "attribute_", "navigation_", "request_", "response_", "redirect_", "javascript_", "script_", "failure_", "belief_", "history_", "logic_", "state_", "phase", "role", "fresh_reset", "oracle_mode", "safe_to_send")):
                values_by_prefix[prefix].append(token.split("=", 1)[1])
    result: dict[str, dict[str, Any]] = {}
    for prefix, values in sorted(values_by_prefix.items()):
        result[prefix] = {
            "observations": len(values),
            "unique_values": len(set(values)),
            "entropy_bits": _entropy(values),
        }
    return result


def _assert_safe(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden audit output key: {key}")
            _assert_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_safe(item)


def audit(*, a_rows_path: Path = DEFAULT_A_ROWS, b_rows_path: Path = DEFAULT_B_ROWS) -> dict[str, Any]:
    source_specs = (("backend_a", a_rows_path), ("backend_b", b_rows_path))
    source_rows: dict[str, list[Mapping[str, Any]]] = {name: _rows(path) for name, path in source_specs}
    all_rows = [row for rows in source_rows.values() for row in rows]
    split_counts = Counter(str(row.get("split", "unknown")) for row in all_rows)
    implementations = Counter(str((row.get("source_meta") or {}).get("implementation", "unknown")) for row in all_rows)
    roles = Counter(str(row.get("role", "unknown")) for row in all_rows)
    context_sequences = [list(row.get("context_tokens", [])) for row in all_rows if isinstance(row.get("context_tokens"), list)]
    target_sequences = [list(row.get("logic_rule_ir_target_tokens", [])) for row in all_rows if isinstance(row.get("logic_rule_ir_target_tokens"), list)]
    context_by_source = {name: [list(row.get("context_tokens", [])) for row in rows if isinstance(row.get("context_tokens"), list)] for name, rows in source_rows.items()}
    target_by_source = {name: [list(row.get("logic_rule_ir_target_tokens", [])) for row in rows if isinstance(row.get("logic_rule_ir_target_tokens"), list)] for name, rows in source_rows.items()}
    context_sets = {name: {_digest(sequence) for sequence in sequences} for name, sequences in context_by_source.items()}
    target_sets = {name: {_digest(sequence) for sequence in sequences} for name, sequences in target_by_source.items()}
    axes = ("document_presence", "navigation_presence", "request_transport_presence", "response_transport_presence", "javascript_presence", "failure_feedback_presence", "belief_replay_presence")
    raw_marker_hits = 0
    row_hash_failures = 0
    for row in all_rows:
        for token in row.get("context_tokens", []) if isinstance(row.get("context_tokens"), list) else []:
            if isinstance(token, str) and any(marker in token.casefold() for marker in ("http://", "https://", "payload=", "wire=", "response_body=", "raw_")):
                raw_marker_hits += 1
        if row.get("record_sha256") and str(row.get("record_sha256")) != _digest({key: value for key, value in row.items() if key != "record_sha256"}):
            # Source rows use their own canonical hash contract; mismatch is
            # reported as a bounded count, never by emitting the row.
            row_hash_failures += 1
    failures: list[str] = []
    if split_counts.get("train", 0) == 0:
        failures.append("train_split_missing")
    if split_counts.get("implementation_holdout", 0) == 0:
        failures.append("implementation_holdout_missing")
    if len(implementations) < 2:
        failures.append("independent_implementation_missing")
    if raw_marker_hits:
        failures.append("raw_context_marker")
    if row_hash_failures:
        failures.append("row_hash_mismatch")
    status = "blocked_information_gate" if failures or split_counts.get("train", 0) == 0 else "diagnostic_information_candidate_only"
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "purpose": "bounded PG-388 abstract context information audit; no training authorization",
        "sources": {
            "backend_a_file": a_rows_path.name,
            "backend_a_file_sha256": _file_sha256(a_rows_path),
            "backend_b_file": b_rows_path.name,
            "backend_b_file_sha256": _file_sha256(b_rows_path),
            "implementation_count": len(implementations),
            "implementation_row_counts": dict(sorted(implementations.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "role_unique_count": len(roles),
            "role_observation_count": sum(roles.values()),
        },
        "sequence_diversity": {
            "context": _sequence_summary(context_sequences),
            "target": _sequence_summary(target_sequences),
            "cross_implementation_context_overlap": len(context_sets.get("backend_a", set()) & context_sets.get("backend_b", set())),
            "cross_implementation_target_overlap": len(target_sets.get("backend_a", set()) & target_sets.get("backend_b", set())),
        },
        "axis_presence": {axis: _axis_summary(all_rows, axis) for axis in axes},
        "abstract_prefix_diversity": _prefix_counts(all_rows),
        "capacity": {
            "context_length_max": max((len(sequence) for sequence in context_sequences), default=0),
            "target_length_max": max((len(sequence) for sequence in target_sequences), default=0),
            "required_window_estimate": max((len(sequence) for sequence in context_sequences + target_sequences), default=0),
            "truncation_observed": False,
        },
        "integrity": {
            "row_count": len(all_rows),
            "raw_context_marker_hits": raw_marker_hits,
            "row_hash_failures": row_hash_failures,
            "context_sequence_emitted": False,
            "target_sequence_emitted": False,
            "row_payload_emitted": False,
        },
        "information_gate": {
            "predictive_entropy_holdout": "not_run",
            "field_ablation": "not_run",
            "train_only_vocabulary": "not_available_without_train_split",
            "passed": False,
            "failures": sorted(set(failures)),
        },
        "training_eligible": 0,
        "promotion": dict(PROMOTION),
    }
    _assert_safe(output)
    output["audit_sha256"] = _digest(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-rows", type=Path, default=DEFAULT_A_ROWS)
    parser.add_argument("--b-rows", type=Path, default=DEFAULT_B_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(a_rows_path=args.a_rows, b_rows_path=args.b_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "row_count": report["sources"]["role_observation_count"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] != "blocked_information_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit"]
