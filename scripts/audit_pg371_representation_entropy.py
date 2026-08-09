"""Read-only PG-371 audit for information loss before the next candidate.

PG-370's high task scores coexist with a large predictive-entropy collapse.
This audit checks the abstract PG-362/PG-367 inputs for train-only vocabulary,
slot diversity, failure-to-repair coverage and cross-split leakage.  It never
reads evaluator sidecars and never emits rows, payloads, wire values or URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "pg362": ROOT / "research" / "pg362_full_rule_ir_dataset_v1.json",
    "pg367": ROOT / "research" / "pg367_waf_staircase_dataset_v2.json",
}
TARGET_SLOTS = (
    "question", "ask_reason", "next_action", "repair_action", "transport_ref", "field_role_ref",
    "encoding_ref", "syntax_category_ref", "probe_variant_ref", "safe_to_send",
    "payload_shape_ref", "oracle_ref", "negative_control_presence_ref",
)
RAW_FRAGMENTS = (
    "raw_payload=", "payload=", "response_body=", "response_body_text=", "wire=",
    "raw_response=", "route_literal=", "family=", "implementation=", "evaluator=",
    "oracle=", "url=", "http://", "https://",
)
FAILURE_SIGNAL_KEYS = {
    "failure_signature", "failure_failure_class", "failure_failure_stage",
    "failure_filter_action", "failure_transform_class", "failure_blocked_reason_class",
    "failure_parse_error_class", "failure_encoding_error_class", "failure_redirect_error_class",
    "failure_repair_axis", "failure_repair_outcome",
}
NEUTRAL_FAILURE_VALUES = {"none", "absent", "zero", "empty", "identity", "allow", "not_applicable"}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def _slot_values(row: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in row.get("target_tokens") or []:
        text = str(token)
        if "=" in text:
            key, value = text.split("=", 1)
            if key in TARGET_SLOTS:
                result[key] = value
    return result


def _tokens(row: Mapping[str, Any], name: str) -> list[str]:
    values = row.get(name) or []
    return [str(item) for item in values] if isinstance(values, list) else []


def _context_failure(context: list[str]) -> bool:
    for token in context:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in FAILURE_SIGNAL_KEYS and value not in NEUTRAL_FAILURE_VALUES:
            return True
    return False


def _axis_summary(rows: list[Mapping[str, Any]], slot: str, split: str) -> dict[str, Any]:
    counts = Counter(_slot_values(row).get(slot, "<missing>") for row in rows if str(row.get("split")) == split)
    total = sum(counts.values())
    dominant = max(counts.values(), default=0)
    entropy = _entropy(counts)
    return {
        "rows": total,
        "value_count": len(counts),
        "entropy_bits": round(entropy, 6),
        "normalized_entropy": round(entropy / math.log2(len(counts)), 6) if len(counts) > 1 else 0.0,
        "dominant_share": round(dominant / total, 6) if total else 0.0,
        "missing_count": int(counts.get("<missing>", 0)),
    }


def audit(datasets: Mapping[str, Mapping[str, Any]], candidate_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    all_rows: list[Mapping[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}
    raw_hits = 0
    for source, dataset in datasets.items():
        rows = [row for row in list(dataset.get("records") or []) if isinstance(row, Mapping)]
        all_rows.extend(rows)
        train = sum(str(row.get("split")) == "train" for row in rows)
        holdout = sum(str(row.get("split")) != "train" for row in rows)
        source_counts[str(source)] = {"records": len(rows), "train": train, "holdout": holdout}
        if not rows:
            failures.append(f"{source}:no_records")
        for index, row in enumerate(rows):
            context = _tokens(row, "context_tokens")
            target = _tokens(row, "target_tokens")
            if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
                raw_hits += 1
                failures.append(f"{source}:raw_token")
            if target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
                failures.append(f"{source}:target_boundary")
            if any(slot not in _slot_values(row) for slot in TARGET_SLOTS):
                failures.append(f"{source}:slot_missing")
            firewall = row.get("context_firewall")
            if not isinstance(firewall, Mapping) or firewall.get("sidecars_off_context") is not True:
                failures.append(f"{source}:context_firewall")

    train_rows = [row for row in all_rows if str(row.get("split")) == "train"]
    holdout_rows = [row for row in all_rows if str(row.get("split")) != "train"]
    train_context = {token for row in train_rows for token in _tokens(row, "context_tokens")}
    train_target = {token for row in train_rows for token in _tokens(row, "target_tokens")}
    holdout_context_unknown = sorted({token for row in holdout_rows for token in _tokens(row, "context_tokens")} - train_context)
    holdout_target_unknown = sorted({token for row in holdout_rows for token in _tokens(row, "target_tokens")} - train_target)
    train_signatures = {_sha([_tokens(row, "context_tokens"), _tokens(row, "target_tokens")]) for row in train_rows}
    holdout_signatures = {_sha([_tokens(row, "context_tokens"), _tokens(row, "target_tokens")]) for row in holdout_rows}
    exact_split_overlap = len(train_signatures & holdout_signatures)
    if raw_hits:
        failures.append("raw_material_present")
    if holdout_context_unknown or holdout_target_unknown:
        failures.append("holdout_vocabulary_gap")
    if exact_split_overlap:
        failures.append("train_holdout_exact_overlap")
    declared_context: set[str] = set()
    declared_target: set[str] = set()
    for dataset in datasets.values():
        vocabulary = dataset.get("vocabulary") if isinstance(dataset.get("vocabulary"), Mapping) else {}
        declared_context.update(str(token) for token in list(vocabulary.get("context_tokens") or []))
        declared_target.update(str(token) for token in list(vocabulary.get("target_tokens") or []))

    axes: dict[str, Any] = {slot: {"train": _axis_summary(all_rows, slot, "train"), "holdout": _axis_summary(all_rows, slot, "implementation_holdout")} for slot in TARGET_SLOTS}
    if any(not axis["train"]["value_count"] or not axis["holdout"]["value_count"] for axis in axes.values()):
        failures.append("axis_without_split_coverage")

    # Group only abstract surface tokens; failure/repair tokens are excluded
    # from the key so paired examples can be identified without copying rows.
    pair_groups: dict[str, set[str]] = defaultdict(set)
    failure_rows = repair_rows = 0
    for row in all_rows:
        values = _slot_values(row)
        context = _tokens(row, "context_tokens")
        failure = _context_failure(context) or values.get("question") == "ask_failure" or values.get("next_action") == "repair"
        repair = values.get("repair_action", "none") != "none" or values.get("next_action") == "repair"
        failure_rows += int(failure)
        repair_rows += int(repair)
        surface = [token for token in context if not token.startswith(("failure_", "waf_", "belief_"))]
        key = _sha([str(row.get("split")), surface, values.get("syntax_category_ref"), values.get("payload_shape_ref")])
        role = "failure_repair" if failure and repair else ("repair" if repair else ("failure" if failure else "baseline"))
        pair_groups[key].add(role)
    paired_groups = sum(1 for labels in pair_groups.values() if labels & {"failure", "failure_repair"} and labels & {"repair", "failure_repair"})
    if paired_groups == 0:
        failures.append("failure_repair_pairs_missing")
    # Mirror PG-372's holdout-precedence view for a like-for-like diagnostic.
    deduped_rows: dict[str, Mapping[str, Any]] = {}
    for row in all_rows:
        signature = _sha([_tokens(row, "context_tokens"), _tokens(row, "target_tokens")])
        previous = deduped_rows.get(signature)
        if previous is None or (str(previous.get("split")) == "train" and str(row.get("split")) != "train"):
            deduped_rows[signature] = row
    effective_pair_groups: dict[str, set[str]] = defaultdict(set)
    for row in deduped_rows.values():
        values = _slot_values(row)
        context = _tokens(row, "context_tokens")
        failure = _context_failure(context) or values.get("question") == "ask_failure" or values.get("next_action") == "repair"
        repair = values.get("repair_action", "none") != "none" or values.get("next_action") == "repair"
        structural = [token for token in context if token.startswith(("document_", "dom_", "element_", "navigation_", "javascript_", "js_"))]
        key = _sha([str(row.get("split")), structural, values.get("syntax_category_ref"), values.get("payload_shape_ref")])
        effective_pair_groups[key].add("failure_repair" if failure and repair else ("repair" if repair else ("failure" if failure else "baseline")))
    paired_after_precedence = sum(1 for labels in effective_pair_groups.values() if labels & {"failure", "failure_repair"} and labels & {"repair", "failure_repair"})

    entropy_drop = None
    baseline_summary: dict[str, Any] = {
        "kind": "missing",
        "trained_baseline_required": True,
        "random_initialization_suspected": False,
        "comparison_valid": False,
    }
    if isinstance(candidate_report, Mapping):
        entropy_drop = (candidate_report.get("worst_seed") or {}).get("entropy_relative_drop_max") if isinstance(candidate_report.get("worst_seed"), Mapping) else None
        training_doc = candidate_report.get("training") if isinstance(candidate_report.get("training"), Mapping) else {}
        candidates = [item for item in list(candidate_report.get("candidates") or []) if isinstance(item, Mapping)]
        baselines = [item.get("baseline") for item in candidates if isinstance(item.get("baseline"), Mapping)]
        baseline_entropy = [float(item.get("predictive_entropy")) for item in baselines if isinstance(item.get("predictive_entropy"), (int, float))]
        baseline_kind = str(training_doc.get("baseline_kind") or training_doc.get("initialization") or "unspecified")
        random_suspected = bool(baselines and all(float(item.get("sequence_exact", 1.0) or 0.0) == 0.0 and float(item.get("token_accuracy", 1.0) or 0.0) < 0.01 for item in baselines))
        baseline_summary = {
            "kind": baseline_kind,
            "trained_baseline_required": True,
            "random_initialization_suspected": random_suspected or baseline_kind in {"random", "random_init", "untrained", "unspecified"},
            "baseline_entropy_mean": round(sum(baseline_entropy) / len(baseline_entropy), 6) if baseline_entropy else None,
            "candidate_count": len(candidates),
            "comparison_valid": bool(baseline_kind == "train_only_next_token_pretrain" and baseline_entropy),
        }
        if baseline_summary["random_initialization_suspected"]:
            failures.append("entropy_baseline_random_or_unspecified")
        try:
            if entropy_drop is not None and float(entropy_drop) > 0.25:
                failures.append("candidate_entropy_drop_over_25pct")
        except (TypeError, ValueError):
            failures.append("candidate_entropy_unreadable")

    status = "passed_diagnostic" if not failures else "blocked_entropy_or_leakage"
    return {
        "schema_version": "pg371-representation-entropy-audit-v1",
        "status": status,
        "sources": source_counts,
        "counts": {"records": len(all_rows), "train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "raw_hits": raw_hits, "exact_split_overlap": exact_split_overlap},
        "declared_ontology_inventory": {"context_token_count": len(declared_context), "target_token_count": len(declared_target), "context_inventory_sha256": _sha(sorted(declared_context)), "target_inventory_sha256": _sha(sorted(declared_target)), "slot_order": list(TARGET_SLOTS)},
        "train_only_vocabulary": {"context_size": len(train_context), "target_size": len(train_target), "holdout_context_unknown_count": len(holdout_context_unknown), "holdout_target_unknown_count": len(holdout_target_unknown), "unknown_token_hash": _sha([holdout_context_unknown, holdout_target_unknown]), "built_from_train_only": True},
        "axes": axes,
        "failure_repair": {"failure_rows": failure_rows, "repair_rows": repair_rows, "paired_surface_groups": paired_groups, "paired_surface_groups_after_holdout_precedence": paired_after_precedence, "pairing_key_abstract_only": True},
        "candidate_entropy_relative_drop_max": entropy_drop,
        "entropy_baseline": baseline_summary,
        "recommended_training_sequence": {
            "stage_1": "train_only_next_token_pretrain",
            "stage_2": "low_lr_multitask_heads_and_structured_slots",
            "compare_post_entropy_to": "trained_stage_1_baseline",
            "random_baseline_comparison_allowed": False,
        },
        "holdout_contract": {"train_only_vocab": True, "exact_context_target_overlap": exact_split_overlap, "holdout_precedence_dedupe": True, "effective_overlap_after_precedence": 0, "raw_context_allowed": False},
        "failures": sorted(set(failures)),
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "raw_material_available": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-362/PG-367 representation entropy")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg371_representation_entropy_audit_v1.json")
    args = parser.parse_args()
    datasets = {key: json.loads(path.read_text(encoding="utf-8-sig")) for key, path in SOURCES.items()}
    candidate_path = ROOT / "research" / "pg370_multitask_moe_candidate_v1.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8-sig")) if candidate_path.exists() else None
    result = audit(datasets, candidate)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed_diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())
