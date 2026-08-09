"""PG-331 read-only audit for abstract-token information preservation.

This audit never starts Docker, sends requests, or trains a model.  It checks
whether the current abstract replay actually preserves the axes required by
the research objective instead of silently compressing them into a few coarse
labels.  Missing axes are reported as incomplete; they are not filled with
zeros or inferred from an evaluator answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg323_decoy_ask_anchor_dataset_v1.json"
ONTOLOGY = RESEARCH / "pg331_web_token_ontology_v1.json"
REPORT = RESEARCH / "pg331_information_preservation_audit_v1.json"

AXES: dict[str, tuple[str, ...]] = {
    "transport_method": ("surface_method",),
    "parameter_role": ("surface_field_role",),
    "encoding_chain": ("surface_encoding",),
    "request_shape": ("surface_method", "surface_field_role", "surface_encoding"),
    "response_shape": ("typed_available", "feedback_state", "evidence_present"),
    "redirect_shape": ("redirect_shape",),
    "script_surface": ("script_surface",),
    "failure_signature": ("failure_class",),
    "history_action": ("history_action",),
    "belief_delta": ("belief_delta",),
    "step_budget": ("step_budget",),
    "replay_state": ("replay_ready", "fresh_reset"),
}
FORBIDDEN_MARKERS = (
    "family=", "route=", "oracle=", "evaluator=", "payload", "response_body", "raw_body", "source_code", "sql", "xss",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(path.name)
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse(tokens: Iterable[Any]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = defaultdict(list)
    for token in tokens:
        text = str(token)
        if "=" not in text or text.startswith("["):
            continue
        key, value = text.split("=", 1)
        parsed[key].append(value)
    return dict(parsed)


def _entropy(values: Sequence[str]) -> dict[str, Any]:
    if not values:
        return {"status": "missing", "bits": None, "nats": None, "count": 0, "unique": 0}
    counts = Counter(values)
    total = sum(counts.values())
    nats = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {"status": "measured", "bits": round(nats / math.log(2), 6), "nats": round(nats, 6), "count": total, "unique": len(counts)}


def _axis_values(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[str]:
    values: list[str] = []
    for row in rows:
        parsed = _parse(row.get("context_tokens") or [])
        if all(key in parsed for key in keys):
            values.append("|".join(f"{key}={parsed[key][0]}" for key in keys))
    return values


def _ablation(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    affected = 0
    eligible = 0
    for row in rows:
        tokens = [str(token) for token in (row.get("context_tokens") or [])]
        if not any(token.split("=", 1)[0] in keys for token in tokens if "=" in token):
            continue
        eligible += 1
        reduced = [token for token in tokens if token.split("=", 1)[0] not in keys]
        affected += int(reduced != tokens)
    return {"eligible_rows": eligible, "observable_delta_rows": affected, "observable_delta_rate": round(affected / eligible, 6) if eligible else None, "status": "measured" if eligible else "missing"}


def audit() -> dict[str, Any]:
    failures: list[str] = []
    if not DATASET.exists():
        return {"schema_version": "pg331-information-preservation-audit-v1", "status": "blocked", "failures": ["missing:dataset"], "promotion_allowed": False}
    dataset = _load(DATASET)
    if not ONTOLOGY.exists():
        failures.append("missing:ontology")
        ontology: dict[str, Any] = {}
    else:
        ontology = _load(ONTOLOGY)
    ontology_axes = {
        str(axis): (str(spec.get("presence_token")),)
        for axis, spec in dict(ontology.get("axes") or {}).items()
        if isinstance(spec, Mapping) and spec.get("required") and spec.get("presence_token")
    }
    if not ontology_axes:
        failures.append("empty:ontology_axes")
        ontology_axes = dict(AXES)
    rows = [dict(row) for row in dataset.get("records", []) if isinstance(row, Mapping)]
    if not rows:
        failures.append("empty:records")
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    unique_sequences = len({json.dumps([row.get("context_tokens") or [], row.get("target_tokens") or []], ensure_ascii=False, sort_keys=True) for row in rows})
    token_axes: dict[str, Any] = {}
    for axis, keys in ontology_axes.items():
        values = _axis_values(rows, keys)
        missing_rows = sum(1 for row in rows if not all(key in _parse(row.get("context_tokens") or []) for key in keys))
        token_axes[axis] = {"required_context_keys": list(keys), "missing_rows": missing_rows, "coverage": round((len(rows) - missing_rows) / max(len(rows), 1), 6), "entropy": _entropy(values), "field_ablation": _ablation(rows, keys), "status": "complete" if missing_rows == 0 else "incomplete"}
        if missing_rows:
            failures.append(f"axis_missing:{axis}")

    raw_leaks: list[dict[str, Any]] = []
    for row in rows:
        for token in row.get("context_tokens") or []:
            text = str(token).lower()
            if any(marker in text for marker in FORBIDDEN_MARKERS):
                raw_leaks.append({"record_id": str(row.get("record_id", "")), "token": str(token)})
    if raw_leaks:
        failures.append("context_firewall")

    reference_alignment = 0
    for row in rows:
        context_keys = set(_parse(row.get("context_tokens") or []))
        target = {str(token) for token in (row.get("target_tokens") or [])}
        refs = {token.split("=", 1)[1] for token in target if token.startswith(("transport_ref=", "field_role_ref=", "encoding_ref="))}
        reference_alignment += int(refs <= context_keys)
    if reference_alignment != len(rows):
        failures.append("context_target_alignment")

    source_split: dict[str, Counter[str]] = defaultdict(Counter)
    implementation_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        source = str((row.get("source_meta") or {}).get("source_id", "missing"))
        implementation = str((row.get("source_meta") or {}).get("implementation", "missing"))
        split = str(row.get("split", "missing"))
        source_split[source][split] += 1
        implementation_split[implementation][split] += 1
    split_leaks = {"source": {key: dict(value) for key, value in source_split.items() if len(value) > 1}, "implementation": {key: dict(value) for key, value in implementation_split.items() if len(value) > 1}}

    report: dict[str, Any] = {
        "protocol_id": "pg-pk-331-information-preservation-audit-v1",
        "schema_version": "pg331-information-preservation-audit-v1",
        "status": "passed" if not failures else "blocked",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": _sha256(dataset),
        "ontology": str(ONTOLOGY.relative_to(ROOT)),
        "ontology_sha256": _sha256(ontology) if ontology else "",
        "ontology_schema_version": str(ontology.get("schema_version", "")),
        "record_count": len(rows),
        "split_counts": dict(split_counts),
        "unique_sequence_count": unique_sequences,
        "unique_sequence_ratio": round(unique_sequences / max(len(rows), 1), 6),
        "token_axes": token_axes,
        "context_target_alignment": {"aligned_rows": reference_alignment, "total_rows": len(rows), "rate": round(reference_alignment / max(len(rows), 1), 6)},
        "context_firewall": {"forbidden_token_count": len(raw_leaks), "examples": raw_leaks[:10]},
        "split_isolation": {"source_cross_split_groups": split_leaks["source"], "implementation_cross_split_groups": split_leaks["implementation"], "status": "review" if split_leaks["source"] or split_leaks["implementation"] else "clean"},
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "interpretation": "PG-331 是表示/数据完整性审计；缺轴说明不可辨识，不是用默认值补齐，也不是漏洞阴性。",
    }
    report["audit_sha256"] = ""
    report["audit_sha256"] = _sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit()
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else f"{report['status']}: {', '.join(report.get('failures') or []) or 'no failures'}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
