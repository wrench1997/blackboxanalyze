"""Read-only audit for PG-331A whole-web source rows.

The audit deliberately consumes serialized abstract rows only.  It never
contacts a target, evaluates a payload, starts Docker, or trains a model.  A
missing/invalid row is reported as incomplete instead of being repaired by
default values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg331_source_row import AXIS_PRESENCE_KEYS, validate_pg331_source_row


DEFAULT_DATASET = ROOT / "research" / "pg331_source_row_collection_v1.json"
DEFAULT_REPORT = ROOT / "research" / "pg331_source_row_audit_v1.json"
REQUIRED_AXES = tuple(sorted(AXIS_PRESENCE_KEYS))
AXIS_PREFIXES = {
    "document_presence": ("document_", "doc_", "dom_", "head_", "body_", "section_", "element_", "attribute_", "text_"),
    "navigation_presence": ("navigation_", "nav_", "link_", "route_shape_", "query_key_", "fragment_"),
    "request_transport_presence": ("request_", "transport_", "param_", "header_", "cookie_", "csrf_", "encoding_"),
    "response_transport_presence": ("response_", "status_", "content_", "body_", "header_", "redirect_", "cache_"),
    "javascript_presence": ("javascript_", "script_", "js_", "event_", "fetch_", "xhr_", "source_", "sink_", "ast_"),
    "failure_feedback_presence": ("failure_", "feedback_", "error_", "timeout_", "blocked_", "repair_"),
    "belief_replay_presence": ("belief_", "history_", "observation_", "oracle_state_", "replay_", "evidence_", "budget_"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, Mapping):
        values = document.get("records")
    else:
        values = document
    return [dict(item) for item in values or [] if isinstance(item, Mapping)]


def _context_keys(tokens: Sequence[Any]) -> set[str]:
    keys: set[str] = set()
    for token in tokens:
        text = str(token)
        if "=" in text:
            keys.add(text.split("=", 1)[0])
    return keys


# Rule-IR targets use stable, short slot names while the context tokenizer
# keeps the ontology axis/field prefix.  Alignment must therefore compare a
# slot to its declared context-field aliases, rather than requiring the
# short slot literal to appear in the context.  Probe-variant references are
# output choices bound by the evaluator sidecar and are intentionally not
# treated as context labels (literal evaluator answers stay off-context).
_TARGET_CONTEXT_ALIASES: dict[str, frozenset[str]] = {
    "request_method": frozenset({"request_method"}),
    "request_placement": frozenset({"request_placement"}),
    "surface_method": frozenset({"surface_method"}),
    "parameter_role": frozenset(
        {
            "parameter_role",
            "request_parameter_role",
            "request_transport_field_parameter_role",
            "surface_field_role",
        }
    ),
    "surface_field_role": frozenset({"surface_field_role", "request_transport_field_surface_field_role"}),
    "encoding_chain": frozenset(
        {
            "encoding_chain",
            "request_encoding_chain",
            "request_transport_field_encoding_chain",
            "surface_encoding",
        }
    ),
    "surface_encoding": frozenset({"surface_encoding", "request_transport_field_surface_encoding"}),
}


def _axis_values(tokens: Sequence[Any], axis: str) -> list[str]:
    prefixes = AXIS_PREFIXES[axis]
    values: list[str] = []
    for token in tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key.startswith(prefixes):
            values.append(value)
    return values


def _entropy(values: Sequence[str]) -> dict[str, Any]:
    if not values:
        return {"status": "missing", "bits": None, "unique": 0, "count": 0}
    counts = Counter(values)
    total = sum(counts.values())
    nats = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {"status": "measured", "bits": round(nats / math.log(2), 6), "unique": len(counts), "count": total}


def _field_ablation(rows: Sequence[Mapping[str, Any]], axis: str) -> dict[str, Any]:
    eligible = 0
    changed = 0
    removed = 0
    prefixes = AXIS_PREFIXES[axis]
    for row in rows:
        if not row.get("training_eligible"):
            continue
        tokens = [str(token) for token in row.get("context_tokens") or []]
        axis_tokens = [token for token in tokens if "=" in token and token.split("=", 1)[0].startswith(prefixes)]
        if not axis_tokens:
            continue
        eligible += 1
        reduced = [token for token in tokens if token not in axis_tokens]
        removed += len(axis_tokens)
        changed += int(reduced != tokens)
    return {"eligible_rows": eligible, "observable_delta_rows": changed, "observable_delta_rate": round(changed / max(eligible, 1), 6) if eligible else None, "removed_token_count": removed, "status": "measured" if eligible else "missing"}


def _alignment(row: Mapping[str, Any]) -> bool:
    keys = _context_keys(row.get("context_tokens") or [])
    for token in row.get("target_tokens") or []:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if not key.endswith("_ref") or value in {"none", "unknown"}:
            continue
        # probe_variant_ref is an evaluator-bound output choice.  Requiring
        # its literal in model context would leak the evaluator answer.
        if key == "probe_variant_ref":
            continue
        aliases = _TARGET_CONTEXT_ALIASES.get(value, frozenset({value}))
        if not any(candidate in aliases or candidate == value or candidate.endswith(f"_{value}") for candidate in keys):
            return False
    return True


def _cross_split(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = str(dict(row.get("source_meta") or {}).get(key, "missing"))
        # Reports need grouping evidence, but need not expose the source name.
        group = sha256_json({"key": key, "value": value})[:16]
        groups[group][str(row.get("split", "missing"))] += 1
    return {group: dict(counts) for group, counts in groups.items() if len(counts) > 1}


def audit_source_rows(document: Any, *, dataset_path: str = "") -> dict[str, Any]:
    rows = _rows(document)
    failures: list[str] = []
    validation_counts = Counter()
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    sequences = set()
    axis_counts = Counter()
    training_rows = 0
    alignment_count = 0
    forbidden_count = 0
    axis_tokens_by_row: dict[str, list[list[str]]] = {axis: [] for axis in REQUIRED_AXES}
    for row in rows:
        result = validate_pg331_source_row(row)
        validation_counts["valid" if result.get("valid") else "invalid"] += 1
        for item in result.get("failures") or []:
            validation_counts[str(item)] += 1
        if not result.get("valid"):
            failures.append(f"invalid_row:{row.get('record_id', 'unknown')}")
        sequences.add(sha256_json({"context": row.get("context_tokens") or [], "target": row.get("target_tokens") or []}))
        presence = dict(result.get("axis_presence") or {})
        for axis in REQUIRED_AXES:
            axis_counts[f"{axis}:{presence.get(axis, 'missing')}"] += 1
        if bool(row.get("training_eligible")):
            training_rows += 1
            alignment_count += int(_alignment(row))
            for axis in REQUIRED_AXES:
                axis_tokens_by_row[axis].append(_axis_values(row.get("context_tokens") or [], axis))
        forbidden_count += int(dict(row.get("context_firewall") or {}).get("forbidden_token_count", 0) or 0)

    source_cross = _cross_split(rows, "source_id")
    implementation_cross = _cross_split(rows, "implementation")
    family_cross = _cross_split(rows, "family_id")
    if not rows:
        failures.append("empty:records")
    if not training_rows:
        failures.append("empty:training_eligible_rows")
    if any(axis_counts[f"{axis}:observed"] < training_rows for axis in REQUIRED_AXES):
        failures.append("training_axis_coverage")
    if forbidden_count:
        failures.append("context_firewall")
    if source_cross:
        failures.append("source_cross_split")
    if implementation_cross:
        failures.append("implementation_cross_split")
    if family_cross:
        failures.append("family_cross_split")
    if training_rows and alignment_count != training_rows:
        failures.append("context_target_alignment")

    axis_quality = {}
    for axis in REQUIRED_AXES:
        values = [value for row_values in axis_tokens_by_row[axis] for value in row_values]
        axis_quality[axis] = {"entropy": _entropy(values), "field_ablation": _field_ablation(rows, axis), "status": "measured" if values else "missing"}
        if training_rows and not values:
            failures.append(f"axis_token_values_missing:{axis}")

    report: dict[str, Any] = {
        "protocol_id": "pg-pk-331a-source-row-audit-v1",
        "schema_version": "pg331a-source-row-audit-v1",
        "status": "passed" if not failures else "blocked",
        "dataset": dataset_path,
        "record_count": len(rows),
        "training_eligible_count": training_rows,
        "split_counts": dict(split_counts),
        "unique_sequence_count": len(sequences),
        "unique_sequence_ratio": round(len(sequences) / max(len(rows), 1), 6),
        "axis_presence_counts": {key: int(value) for key, value in sorted(axis_counts.items())},
        "axis_quality": axis_quality,
        "validation_counts": dict(validation_counts),
        "context_target_alignment": {"aligned_training_rows": alignment_count, "total_training_rows": training_rows, "rate": round(alignment_count / max(training_rows, 1), 6)},
        "context_firewall": {"forbidden_token_count": forbidden_count},
        "split_isolation": {
            "source_cross_split_groups": source_cross,
            "implementation_cross_split_groups": implementation_cross,
            "family_cross_split_groups": family_cross,
            "status": "review" if source_cross or implementation_cross or family_cross else "clean",
        },
        "failures": sorted(set(failures)),
        "promotion": {
            "training_allowed": not bool(failures),
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "independent_implementation_gate_required": True,
        },
        "interpretation": "PG-331A 只审计整页 token/source-row 完整性；blocked 表示必须 ASK/补采，不表示漏洞阴性。",
    }
    report["audit_sha256"] = ""
    report["audit_sha256"] = sha256_json(report)
    return report


def _dataset_label(path: Path) -> str:
    """Return a stable report label for relative and external dataset paths.

    CLI callers commonly pass ``research/...`` while tests and operators may
    use an absolute staging path outside the workspace.  Normalizing before
    ``relative_to`` keeps the audit read-only and avoids turning a valid
    incomplete/ASK report into a path-handling exception.
    """

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def audit(path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    path = Path(path)
    label = _dataset_label(path)
    if not path.exists():
        report = audit_source_rows({}, dataset_path=label)
        report["failures"] = ["missing:dataset"]
        report["status"] = "blocked"
        report["promotion"]["training_allowed"] = False
        report["audit_sha256"] = ""
        report["audit_sha256"] = sha256_json(report)
        return report
    return audit_source_rows(_load(path), dataset_path=label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.dataset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else f"{report['status']}: {', '.join(report.get('failures') or []) or 'no failures'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
