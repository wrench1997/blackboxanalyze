"""Audit a full-axis target-conditioned candidate corpus for PG-343.

PG-343 is deliberately an audit, not a training-data promotion step.  It
combines previously collected abstract rows only to answer one question:
does the same abstract context map to one stable target?  If candidate,
reference, negative, repair, or ASK roles were omitted from the context, a
causal next-token model sees contradictory labels.  Those rows are reported
as ambiguous and remain blocked until a fresh collector records an abstract
role/step token.

The script never contacts a target, starts Docker/GPU, or emits tokens,
payloads, routes, response bodies, evaluator answers, or sidecars in the
report.  It writes only bounded counts, one-way hashes, split/implementation
groups, and failure reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DEFAULT_SOURCES = (
    RESEARCH / "pg338_information_preserving_process_token_v1.json",
    RESEARCH / "pg339_multi_shape_diagnostic_dataset_v1.json",
    RESEARCH / "pg342_full_axis_failure_repair_dataset_v1.json",
)
SCHEMA_VERSION = "pg343-full-axis-target-conditioned-audit-v1"
ROLE_BOUND_DATASET_SCHEMA = "pg343-role-bound-full-axis-target-conditioned-dataset-v1"
ROLE_BOUND_DATASET_SCHEMAS = frozenset(
    {
        ROLE_BOUND_DATASET_SCHEMA,
        "pg344-cross-implementation-role-bound-dataset-v1",
        "pg345-decision-boundary-role-bound-dataset-v1",
    }
)
AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_and_replay",
)

# These are one-way implementation attestations already present in the
# source artifacts.  They are deliberately not rendered as implementation
# names in the report or model context.
TRAIN_IMPLEMENTATION_HASHES = frozenset(
    {
        "40bdd27f85c795ec0eada01fbd30f0f5ba95a437a2a4a466bca54fb25fcd226e",
        "f7e3a7c77928e0dac1c86ee9b2a60008db64cf97e1bb49d6360ee896f0f9fa37",
    }
)
HOLDOUT_IMPLEMENTATION_HASHES = frozenset(
    {
        "853bdaae8abf99154475d53684a48597692fbbcfcb9ca988316ff456265382af",
        "410d54aca7ed29e10ecb170cb892de90b4aed57d1a1d9e4ffd945dfe84695498",
        "6d689d1e5cdfdc30af8d18870f40d016a14363a69f521f30fb6aca8e65338e15",
    }
)
FORBIDDEN_FRAGMENTS = (
    "payload=",
    "payload_",
    "response_body=",
    "response_body_text=",
    "raw_",
    "oracle=",
    "evaluator=",
    "family=",
    "route=",
    "route_literal=",
    "implementation=",
    "image=",
    "url=",
    "path=",
    "source=",
)
ALLOWED_TARGET_PREFIXES = (
    "[TARGET_BOS]",
    "[TARGET_EOS]",
    "question=",
    "next_action=",
    "repair_action=",
    "action_changed=",
    "failure_class=",
    "safe_to_send=",
    "transport_ref=",
    "field_role_ref=",
    "encoding_ref=",
    "probe_variant_ref=",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"source is not an object: {path}")
    return value


def _target_map(tokens: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in {
            "question",
            "next_action",
            "repair_action",
            "action_changed",
            "safe_to_send",
            "transport_ref",
            "field_role_ref",
            "encoding_ref",
            "probe_variant_ref",
        }:
            result[key] = value
    return result


def _target_family(tokens: Sequence[Any]) -> str:
    values = _target_map(tokens)
    question = values.get("question", "")
    action = values.get("next_action", "")
    if action in {"repair", "repair_abstract_plan"}:
        return "repair"
    if action in {"abstain", "review_negative"} or question == "review_negative":
        return "negative_abstain"
    if action in {"send_probe", "select_probe_variant", "assemble_rule_ir"}:
        return "positive_probe"
    if question.startswith("ask_") or action.startswith("ask_"):
        return "ask"
    return "other"


def _entropy(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return round(-sum((count / total) * math.log2(count / total) for count in counts.values()), 6)


def _axis_sequence_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for axis in AXES:
        sequences: list[str] = []
        for row in rows:
            # _validate_row stores only one-way sequence hashes, never the
            # underlying abstract tokens.
            sequence_hash = (row.get("axis_sequence_hashes") or {}).get(axis)
            if sequence_hash:
                sequences.append(str(sequence_hash))
        stats[axis] = {
            "records": len(sequences),
            "unique_sequences": len(set(sequences)),
            "entropy_bits": _entropy(sequences),
        }
    return stats


def _validate_row(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    context = raw.get("context_tokens")
    target = raw.get("target_tokens")
    if not isinstance(context, list) or len(context) < 2:
        return None, "context_missing"
    if not isinstance(target, list) or target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
        return None, "target_boundary_missing"
    for token in [*context, *target]:
        value = str(token).casefold()
        if any(fragment in value for fragment in FORBIDDEN_FRAGMENTS):
            return None, "context_or_target_firewall"
    if any(not str(token).startswith(ALLOWED_TARGET_PREFIXES) for token in target):
        return None, "non_abstract_target"
    firewall = raw.get("context_firewall")
    if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
        return None, "firewall_metadata"
    if any(raw.get(key) is not False for key in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context")):
        return None, "raw_oracle_flag"
    impl = str(raw.get("source_implementation_hash", ""))
    if len(impl) != 64:
        return None, "implementation_hash_missing"
    role_bound_dataset = str(raw.get("schema_version", "")) in ROLE_BOUND_DATASET_SCHEMAS
    if not role_bound_dataset and impl not in TRAIN_IMPLEMENTATION_HASHES | HOLDOUT_IMPLEMENTATION_HASHES:
        return None, "implementation_not_allowlisted"
    source_record = str(raw.get("source_record_sha256", ""))
    if len(source_record) != 64:
        return None, "source_record_hash_missing"
    axis_presence = raw.get("axis_presence")
    if not isinstance(axis_presence, Mapping) or any(axis not in axis_presence for axis in AXES):
        return None, "full_axis_presence_missing"
    if role_bound_dataset:
        split = str(raw.get("split", ""))
        if split not in {"train", "implementation_holdout"}:
            return None, "split_missing"
    else:
        split = "train" if impl in TRAIN_IMPLEMENTATION_HASHES else "implementation_holdout"
    context_hash = _sha(context)
    target_hash = _sha(target)
    target_values = _target_map(target)
    axis_sequence_hashes: dict[str, str] = {}
    for axis in AXES:
        begin = f"axis_begin={axis}"
        end = f"axis_end={axis}"
        if begin in context:
            start = context.index(begin) + 1
            if end in context[start:]:
                stop = context.index(end, start)
                axis_sequence_hashes[axis] = _sha(context[start:stop])
    return {
        "context_hash": context_hash,
        "target_hash": target_hash,
        "source_record_hash": source_record,
        "implementation_hash": impl,
        "split": split,
        "source_split": str(raw.get("source_split", raw.get("split", "unknown"))),
        "target_family": _target_family(target),
        "target_question": target_values.get("question", "missing"),
        "axis_presence": {axis: str(axis_presence.get(axis)) for axis in AXES},
        "axis_sequence_hashes": axis_sequence_hashes,
    }, None


def audit_sources(sources: Sequence[Path] = DEFAULT_SOURCES) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    rejection_reasons: Counter[str] = Counter()
    for path in sources:
        data = _load(path)
        source_files.append({"name": path.name, "file_sha256": _file_sha(path), "dataset_sha256": str(data.get("dataset_sha256", ""))})
        for raw in data.get("records") or []:
            if not isinstance(raw, Mapping):
                rejection_reasons["row_not_mapping"] += 1
                continue
            row, reason = _validate_row(raw)
            if row is None:
                rejection_reasons[reason or "invalid_row"] += 1
                continue
            row["source_artifact"] = path.name
            accepted.append(row)

    # Remove exact context+target duplicates across historical artifacts, but
    # retain their provenance hashes for a leakage audit.  No row content is
    # emitted.
    exact_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        exact_groups[(row["context_hash"], row["target_hash"])].append(row)
    unique_rows: list[dict[str, Any]] = []
    for group in exact_groups.values():
        unique_rows.append(sorted(group, key=lambda item: (item["source_artifact"], item["source_record_hash"]))[0])

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_implementation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        by_context[row["context_hash"]].append(row)
        by_source_record[row["source_record_hash"]].append(row)
        by_implementation[row["implementation_hash"]].append(row)
    ambiguous = {key: values for key, values in by_context.items() if len({row["target_hash"] for row in values}) > 1}
    split_leaks = {key: values for key, values in by_context.items() if len({row["split"] for row in values}) > 1}
    source_record_leaks = {key: values for key, values in by_source_record.items() if len({row["split"] for row in values}) > 1}
    implementation_split_leaks = {
        key: values for key, values in by_implementation.items() if len({row["split"] for row in values}) > 1
    }

    target_counts = Counter((row["split"], row["target_family"]) for row in unique_rows)
    axis_sequence_stats = _axis_sequence_stats(unique_rows)
    # ``question=ask_failure`` is part of a repair target, so it must be
    # counted independently instead of forcing one mutually-exclusive class
    # to stand in for both ASK and repair supervision.
    question_ask_counts = Counter(
        row["split"]
        for row in unique_rows
        if str(row.get("target_question", "")).startswith("ask_")
    )
    implementation_counts = Counter((row["split"], row["implementation_hash"]) for row in unique_rows)
    axis_entropy: dict[str, dict[str, float | int]] = {}
    for axis in AXES:
        values = [row["axis_presence"][axis] for row in unique_rows]
        axis_entropy[axis] = {"entropy_bits": _entropy(values), "unique_values": len(set(values))}

    required_families = {"positive_probe", "ask", "repair", "negative_abstain"}
    train_families = {row["target_family"] for row in unique_rows if row["split"] == "train"}
    holdout_families = {row["target_family"] for row in unique_rows if row["split"] != "train"}
    train_ask = question_ask_counts["train"] > 0
    holdout_ask = question_ask_counts["implementation_holdout"] > 0
    failures: list[str] = []
    if not unique_rows:
        failures.append("no_valid_rows")
    if ambiguous:
        failures.append("context_target_ambiguity_requires_role_step_token")
    if split_leaks:
        failures.append("context_split_leakage")
    if source_record_leaks:
        failures.append("source_record_split_leakage")
    if implementation_split_leaks:
        failures.append("implementation_split_leakage")
    if not (required_families - {"ask"}).issubset(train_families) or not train_ask:
        failures.append("train_target_coverage_incomplete")
    if not (required_families - {"ask"}).issubset(holdout_families) or not holdout_ask:
        failures.append("holdout_target_coverage_incomplete")
    if any(int(item["unique_sequences"]) < 2 for item in axis_sequence_stats.values()):
        failures.append("axis_token_sequence_entropy_insufficient")

    if "context_target_ambiguity_requires_role_step_token" in failures:
        status = "blocked_role_step_context_missing"
    elif failures:
        status = "blocked_information_gate"
    else:
        status = "diagnostic_passed_not_training_eligible"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "purpose": "audit full-axis target-conditioned composition before any SFT/RL or A800 run",
        "sources": source_files,
        "counts": {
            "input_rows": len(accepted),
            "unique_context_target_rows": len(unique_rows),
            "unique_contexts": len(by_context),
            "ambiguous_contexts": len(ambiguous),
            "context_split_leaks": len(split_leaks),
            "source_record_split_leaks": len(source_record_leaks),
            "implementation_split_leaks": len(implementation_split_leaks),
            "train_rows": sum(row["split"] == "train" for row in unique_rows),
            "implementation_holdout_rows": sum(row["split"] != "train" for row in unique_rows),
        },
        "target_counts": {
            **{f"{split}:{family}": count for (split, family), count in sorted(target_counts.items())},
            "train:question_ask": int(question_ask_counts["train"]),
            "implementation_holdout:question_ask": int(question_ask_counts["implementation_holdout"]),
        },
        "implementation_counts": {f"{split}:{impl}": count for (split, impl), count in sorted(implementation_counts.items())},
        "axis_presence_entropy": axis_entropy,
        "axis_token_sequence_entropy": axis_sequence_stats,
        "ambiguity_hashes": [
            {
                "context_sha256": context_hash,
                "target_sha256s": sorted({row["target_hash"] for row in values}),
                "row_count": len(values),
            }
            for context_hash, values in sorted(ambiguous.items())
        ][:32],
        "required_target_families": sorted(required_families),
        "train_target_families": sorted(train_families),
        "holdout_target_families": sorted(holdout_families),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "failures": sorted(set(failures)),
        "next_required_observation": "role_step_abstract_token_in_context",
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "audit_sha256": "",
    }
    report["audit_sha256"] = _sha(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-343 full-axis target-conditioned data")
    parser.add_argument("--source", action="append", type=Path, dest="sources")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = tuple(args.sources) if args.sources else DEFAULT_SOURCES
    result = audit_sources(sources)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "failures": result["failures"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
