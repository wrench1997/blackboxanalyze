"""Read-only information audit for fresh PG-331 typed source rows.

The historical :mod:`audit_pg331_information_preservation` report was written
for the old PG-323 rows.  It can report a few coarse token axes, but it cannot
prove that a fresh ``pg331_source_row`` dataset covers every ontology field or
that the append-only vocabulary/capacity contracts contain those fields.  This
module is the stricter, dataset-oriented diagnostic used before any future
training request.

It consumes serialized abstract source rows and optional vocabulary/capacity
reports only.  It never starts Docker, opens a socket, loads a checkpoint, or
changes a training gate.  Raw values are counted as firewall failures and are
never copied to the output.  The report deliberately has no ``passed`` state:
``blocked`` means the input contract is absent/unreadable, ``incomplete``
means required observations or artifacts are missing, and ``diagnostic`` means
the measurements are available but remain evaluator/research diagnostics.  In
all states promotion is false; ``operator_reviewed`` and a single
implementation cannot make a dataset training-eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATASET = RESEARCH / "pg331_source_row_collection_v1.json"
DEFAULT_ONTOLOGY = RESEARCH / "pg331_web_token_ontology_v1.json"
DEFAULT_VOCABULARY = RESEARCH / "pg331_web_token_vocabulary_v1.json"
DEFAULT_CAPACITY = RESEARCH / "pg331_model_capacity_audit_v1.json"
DEFAULT_REPORT = RESEARCH / "pg331_dataset_information_audit_v1.json"

SCHEMA_VERSION = "pg331-dataset-information-audit-v1"
PROTOCOL_ID = "pg-pk-331-dataset-information-audit-v1"
REQUIRED_STATUS = ("observed", "absent", "not_observed", "unknown")
FORBIDDEN_MARKERS = (
    "raw_payload",
    "raw_response",
    "response_body",
    "oracle_answer",
    "evaluator_answer",
    "family_label",
    "route_literal",
    "credential",
    "cookie_value",
    "authorization_value",
    "secret",
)
FORBIDDEN_TOKEN_KEYS = {
    "raw_payload",
    "raw_response",
    "response_body",
    "response_body_text",
    "oracle_answer",
    "evaluator_answer",
    "family_label",
    "route_literal",
    "route_name",
    "credential",
    "cookie_value",
    "authorization_value",
    "secret",
}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _rows(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, Mapping):
        values = document.get("records")
    elif isinstance(document, Sequence) and not isinstance(document, (str, bytes, bytearray)):
        values = document
    else:
        values = []
    return [dict(item) for item in values or [] if isinstance(item, Mapping)]


def _ontology_axes(document: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(document, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    axes = document.get("axes")
    if not isinstance(axes, Mapping):
        return result
    for axis, raw in axes.items():
        if not isinstance(raw, Mapping):
            continue
        fields = tuple(str(field) for field in (raw.get("fields") or []) if str(field))
        presence = str(raw.get("presence_token") or f"{axis}_presence")
        result[str(axis)] = {"presence_token": presence, "fields": fields}
    return result


def _token_map(tokens: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes, bytearray)):
        return {}
    for token in tokens:
        text = str(token)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if key:
            result[key].append(value)
    return dict(result)


def _first(mapping: Mapping[str, Sequence[str]], key: str) -> str | None:
    values = mapping.get(key)
    return str(values[0]) if values else None


def _entropy(values: Sequence[str]) -> dict[str, Any]:
    clean = [str(value) for value in values]
    if not clean:
        return {"status": "missing", "bits": None, "nats": None, "count": 0, "unique": 0, "unique_ratio": None}
    counts = Counter(clean)
    total = sum(counts.values())
    nats = -sum((count / total) * math.log(count / total) for count in counts.values())
    return {
        "status": "measured",
        "bits": round(nats / math.log(2), 6),
        "nats": round(nats, 6),
        "count": total,
        "unique": len(counts),
        "unique_ratio": round(len(counts) / max(total, 1), 6),
    }


def _sequence_metrics(values: Sequence[str]) -> dict[str, Any]:
    clean = [str(value) for value in values]
    unique = len(set(clean))
    return {
        "count": len(clean),
        "unique": unique,
        "unique_ratio": round(unique / max(len(clean), 1), 6) if clean else None,
        "status": "measured" if clean else "missing",
    }


def _field_key(axis: str, field: str) -> str:
    return f"{axis}_field_{field}"


def _safe_field_value(value: str | None) -> str:
    # Values are tokenizer projections.  Keep bounded enum/bucket information
    # but never let a malicious dataset make its literal appear in a report.
    if value is None:
        return "missing"
    folded = str(value).casefold()
    if any(marker in folded for marker in FORBIDDEN_MARKERS):
        return "forbidden"
    return str(value)


def _axis_signature(axis: str, spec: Mapping[str, Any], token_map: Mapping[str, Sequence[str]]) -> str:
    presence = _safe_field_value(_first(token_map, str(spec.get("presence_token", ""))))
    fields = [
        f"{field}={_safe_field_value(_first(token_map, _field_key(axis, field)))}"
        for field in spec.get("fields") or []
    ]
    return "|".join([f"presence={presence}", *fields])


def _field_metric(values: Sequence[str], *, rows: int, missing: int, status_counts: Counter[str]) -> dict[str, Any]:
    metric = {
        "coverage": round((rows - missing) / max(rows, 1), 6),
        "missing_rows": int(missing),
        "status_counts": dict(sorted((str(key), int(value)) for key, value in status_counts.items())),
        "entropy": _entropy(values),
        "sequence": _sequence_metrics(values),
        "status": "measured" if values else "missing",
    }
    if missing:
        metric["status"] = "incomplete"
    return metric


def _field_ablation(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    eligible = 0
    changed = 0
    removed = 0
    before: set[str] = set()
    after: set[str] = set()
    for row in rows:
        tokens = [str(token) for token in (row.get("context_tokens") or [])]
        if not any(token.split("=", 1)[0] == key for token in tokens if "=" in token):
            continue
        eligible += 1
        original_digest = sha256_json(tokens)
        reduced = [token for token in tokens if token.split("=", 1)[0] != key]
        before.add(original_digest)
        after.add(sha256_json(reduced))
        removed += len(tokens) - len(reduced)
        changed += int(reduced != tokens)
    return {
        "eligible_rows": eligible,
        "observable_delta_rows": changed,
        "observable_delta_rate": round(changed / max(eligible, 1), 6) if eligible else None,
        "removed_token_count": removed,
        "unique_sequences_before": len(before),
        "unique_sequences_after": len(after),
        "status": "measured" if eligible else "unused",
    }


def _axis_ablation(rows: Sequence[Mapping[str, Any]], axis: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    keys = {str(spec.get("presence_token", "")), "axis_begin", "axis_end"}
    keys.update(_field_key(axis, str(field)) for field in spec.get("fields") or [])
    eligible = changed = removed = 0
    before: set[str] = set()
    after: set[str] = set()
    for row in rows:
        tokens = [str(token) for token in (row.get("context_tokens") or [])]
        axis_tokens = [token for token in tokens if "=" in token and token.split("=", 1)[0] in keys]
        if not axis_tokens:
            continue
        eligible += 1
        reduced = [token for token in tokens if not ("=" in token and token.split("=", 1)[0] in keys)]
        before.add(sha256_json(tokens))
        after.add(sha256_json(reduced))
        removed += len(tokens) - len(reduced)
        changed += int(reduced != tokens)
    return {
        "eligible_rows": eligible,
        "observable_delta_rows": changed,
        "observable_delta_rate": round(changed / max(eligible, 1), 6) if eligible else None,
        "removed_token_count": removed,
        "unique_sequences_before": len(before),
        "unique_sequences_after": len(after),
        "status": "measured" if eligible else "unused",
    }


def _raw_context_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        for token in row.get("context_tokens") or []:
            text = str(token)
            key = text.split("=", 1)[0].casefold()
            lowered = text.casefold()
            # Structural ``response_body_shape``/``response_body_length``
            # tokens are valid ontology projections.  Only a literal raw key
            # (or a raw-prefixed key) is forbidden here.
            count += int(key in FORBIDDEN_TOKEN_KEYS or key.startswith("raw_") or any(marker in lowered for marker in ("family=", "route_literal=", "oracle=", "evaluator=")))
    return count


def _required_inventory(ontology: Mapping[str, Any]) -> set[str]:
    inventory = {
        "[BOS]",
        "[EOS]",
        "[TARGET_BOS]",
        "[TARGET_EOS]",
        "unknown",
        "not_observed",
        "empty",
        "present",
        "absent",
        "blocked",
        "chunk_boundary=begin",
        "chunk_boundary=end",
        *{f"chunk_shape={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
        *{f"chunk_index={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
        *{f"chunk_count={bucket}" for bucket in ("zero", "one", "two", "few", "many")},
        *{f"chunk_digest=b{value:02x}" for value in range(256)},
    }
    reserved = ontology.get("reserved_tokens")
    if isinstance(reserved, Mapping):
        inventory.update(str(token) for token in (reserved.get("universal") or []))
        inventory.update(str(token) for token in (reserved.get("bucket_policy") or []))
    axes = ontology.get("axes") if isinstance(ontology, Mapping) else {}
    for axis, raw in (axes.items() if isinstance(axes, Mapping) else []):
        if not isinstance(raw, Mapping):
            continue
        presence = str(raw.get("presence_token") or f"{axis}_presence")
        inventory.update({f"{presence}=observed", f"{presence}=not_observed"})
        for field in raw.get("fields") or []:
            key = _field_key(str(axis), str(field))
            # ``absent`` is a valid field-capture status; reserving it here is
            # important even when an old builder only emitted observed/
            # not_observed/unknown slots.
            inventory.update({f"{key}={status}" for status in REQUIRED_STATUS})
    return inventory


def _axis_inventory_coverage(ontology: Mapping[str, Any], context: set[str]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    axes = ontology.get("axes") if isinstance(ontology, Mapping) else {}
    for axis, raw in (axes.items() if isinstance(axes, Mapping) else []):
        if not isinstance(raw, Mapping):
            continue
        axis_name = str(axis)
        required = {
            f"{str(raw.get('presence_token') or f'{axis_name}_presence')}={status}"
            for status in ("observed", "not_observed")
        }
        for field in raw.get("fields") or []:
            key = _field_key(axis_name, str(field))
            required.update(f"{key}={status}" for status in REQUIRED_STATUS)
        missing = sorted(required - context)
        coverage[axis_name] = {
            "required_count": len(required),
            "missing_count": len(missing),
            "missing_sha256": sha256_json(missing),
            "coverage": round((len(required) - len(missing)) / max(len(required), 1), 6),
            "status": "measured" if not missing else "incomplete",
        }
    return coverage


def _vocabulary_audit(
    rows: Sequence[Mapping[str, Any]], ontology: Mapping[str, Any], vocabulary: Any,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if not isinstance(vocabulary, Mapping):
        return {"status": "missing", "context_vocab_count": 0, "target_vocab_count": 0}, ["missing:vocabulary"]
    context = {str(token) for token in vocabulary.get("context_tokens") or []}
    target = {str(token) for token in vocabulary.get("target_tokens") or []}
    observed_context = [str(token) for row in rows for token in row.get("context_tokens") or []]
    observed_target = [str(token) for row in rows for token in row.get("target_tokens") or []]
    context_missing = [token for token in observed_context if token not in context]
    target_missing = [token for token in observed_target if token not in target]
    required = _required_inventory(ontology)
    inventory_missing = sorted(required - context)
    if context_missing:
        failures.append("vocabulary_context_coverage")
    if target_missing:
        failures.append("vocabulary_target_coverage")
    if inventory_missing:
        failures.append("vocabulary_ontology_inventory")
    # Never expose unknown token strings: only bounded counts and digests are
    # needed to reproduce the diagnosis without leaking a literal.
    return {
        "status": "measured" if not failures else "incomplete",
        "manifest_status": str(vocabulary.get("status", "unknown")),
        "context_vocab_count": len(context),
        "target_vocab_count": len(target),
        "observed_context_token_count": len(observed_context),
        "observed_target_token_count": len(observed_target),
        "context_missing_token_count": len(context_missing),
        "target_missing_token_count": len(target_missing),
        "context_coverage": round((len(observed_context) - len(context_missing)) / max(len(observed_context), 1), 6) if observed_context else None,
        "target_coverage": round((len(observed_target) - len(target_missing)) / max(len(observed_target), 1), 6) if observed_target else None,
        "ontology_inventory_count": len(required),
        "ontology_inventory_missing_count": len(inventory_missing),
        "ontology_inventory_missing_sha256": sha256_json(inventory_missing),
        "axis_inventory": _axis_inventory_coverage(ontology, context),
        "observed_context_unknown_sha256": sha256_json(sorted(set(context_missing))),
        "observed_target_unknown_sha256": sha256_json(sorted(set(target_missing))),
        "failures": sorted(set(failures)),
    }, failures


def _capacity_audit(capacity: Any, rows: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(capacity, Mapping):
        return {"status": "missing", "variants": []}, ["missing:capacity"]
    variants = capacity.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes, bytearray)):
        variants = []
    normalized: list[dict[str, Any]] = []
    for value in variants:
        if not isinstance(value, Mapping):
            continue
        normalized.append(
            {
                "id": str(value.get("config", {}).get("id", value.get("id", "unknown"))) if isinstance(value.get("config", {}), Mapping) else str(value.get("id", "unknown")),
                "context_window_pass": value.get("context_window_pass") is True,
                "capacity_pass": value.get("capacity_pass") is True,
                "truncation_risk": value.get("truncation_risk") is True,
            }
        )
    required_window = capacity.get("required_context_window")
    context_lengths = [len(list(row.get("context_tokens") or [])) for row in rows]
    target_lengths = [len(list(row.get("target_tokens") or [])) for row in rows]
    observed_max_context = max(context_lengths, default=0)
    observed_target_budget = max(max(target_lengths, default=0), 32)
    dataset_required_window = int(math.ceil((observed_max_context + observed_target_budget) * 1.25)) if rows else None
    if not normalized or required_window is None:
        return {
            "status": "incomplete",
            "capacity_report_status": str(capacity.get("status", "unknown")),
            "required_context_window": required_window,
            "dataset_required_context_window": dataset_required_window,
            "dataset_context_length_max": observed_max_context,
            "dataset_target_length_max": max(target_lengths, default=0),
            "variants": normalized,
        }, ["capacity_contract"]
    failures = [] if any(item["capacity_pass"] and not item["truncation_risk"] for item in normalized) else ["capacity_no_passing_variant"]
    if dataset_required_window is not None and int(required_window) < dataset_required_window:
        failures.append("capacity_dataset_window")
    return {
        "status": "measured" if not failures else "incomplete",
        "capacity_report_status": str(capacity.get("status", "unknown")),
        "required_context_window": required_window,
        "dataset_required_context_window": dataset_required_window,
        "dataset_context_length_max": observed_max_context,
        "dataset_target_length_max": max(target_lengths, default=0),
        "variants": normalized,
        "passing_variant_count": sum(int(item["capacity_pass"]) for item in normalized),
    }, failures


def _row_validation(rows: Sequence[Mapping[str, Any]], axes: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    valid = 0
    invalid = 0
    typed_complete = 0
    fresh_complete = 0
    negative_complete = 0
    replay_complete = 0
    review_count = 0
    claimed_training = 0
    implementations: set[str] = set()
    source_ids: set[str] = set()
    split_groups: dict[str, dict[str, set[str]]] = {"source_id": defaultdict(set), "implementation": defaultdict(set), "family_id": defaultdict(set)}

    # Import lazily so this script remains usable as a standalone JSON audit.
    try:
        from app.pg331_source_row import validate_pg331_source_row
    except Exception:  # pragma: no cover - import failure is itself diagnostic
        validate_pg331_source_row = None

    for row in rows:
        if bool(row.get("operator_reviewed")):
            review_count += 1
        claimed_training += int(row.get("training_eligible") is True)
        meta = row.get("source_meta") if isinstance(row.get("source_meta"), Mapping) else {}
        implementation = str(meta.get("implementation", "missing"))
        source_id = str(meta.get("source_id", "missing"))
        source_ids.add(source_id)
        implementations.add(implementation)
        split = str(row.get("split", "missing"))
        for key in split_groups:
            split_groups[key][str(meta.get(key, "missing"))].add(split)
        if validate_pg331_source_row is None:
            result = {"valid": False, "failures": ["validator_unavailable"]}
        else:
            try:
                result = validate_pg331_source_row(row)
            except Exception as error:  # malformed rows are incomplete, never fatal
                result = {"valid": False, "failures": [f"validator_exception:{type(error).__name__}"]}
        if result.get("valid"):
            valid += 1
        else:
            invalid += 1
            failures.append("invalid_source_row")
        evaluator = row.get("evaluator_sidecar") if isinstance(row.get("evaluator_sidecar"), Mapping) else {}
        typed = all(evaluator.get(key) is True for key in ("typed_available", "reference_present", "candidate_present"))
        fresh = bool(evaluator.get("fresh_reset") is True and isinstance(row.get("reset"), Mapping) and row["reset"].get("fresh_reset") is True)
        negative = evaluator.get("negative_control") is True
        replay = False
        for token in row.get("context_tokens") or []:
            text = str(token)
            if "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key in {"replay_ready", "belief_replay_ready", "belief_and_replay_field_replay_ready"} and value == "present":
                replay = True
                break
        typed_complete += int(typed)
        fresh_complete += int(fresh)
        negative_complete += int(negative)
        replay_complete += int(replay)
    if not rows:
        failures.append("empty:records")
    if invalid:
        failures.append("invalid_rows")
    if typed_complete < len(rows):
        failures.append("typed_evaluator_incomplete")
    if fresh_complete < len(rows):
        failures.append("fresh_reset_incomplete")
    if negative_complete < len(rows):
        failures.append("negative_control_incomplete")
    if replay_complete < len(rows):
        failures.append("replay_state_incomplete")
    cross_split = {key: {sha256_json({"key": key, "value": value})[:16]: sorted(splits) for value, splits in groups.items() if len(splits) > 1} for key, groups in split_groups.items()}
    if any(cross_split.values()):
        failures.append("split_isolation")
    if len(implementations - {"missing"}) < 2:
        failures.append("single_implementation_diagnostic")
    return {
        "record_count": len(rows),
        "valid_row_count": valid,
        "invalid_row_count": invalid,
        "typed_complete_count": typed_complete,
        "fresh_reset_complete_count": fresh_complete,
        "negative_control_complete_count": negative_complete,
        "replay_state_complete_count": replay_complete,
        "operator_reviewed_count": review_count,
        "claimed_training_eligible_count": claimed_training,
        "accepted_training_eligible_count": 0,
        "implementation_count": len(implementations - {"missing"}),
        "source_count": len(source_ids - {"missing"}),
        "cross_split_groups": cross_split,
        "failures": sorted(set(failures)),
    }, failures


def audit_document(
    document: Any,
    *,
    ontology: Any,
    vocabulary: Any = None,
    capacity: Any = None,
    dataset_path: str = "",
    ontology_path: str = "",
    vocabulary_path: str = "",
    capacity_path: str = "",
) -> dict[str, Any]:
    """Audit an in-memory dataset and optional manifests without side effects."""

    rows = _rows(document)
    axes = _ontology_axes(ontology)
    failures: list[str] = []
    if not axes:
        failures.append("missing:ontology_axes")
    if not rows:
        failures.append("empty:records")

    validation, validation_failures = _row_validation(rows, axes)
    failures.extend(validation_failures)
    raw_count = _raw_context_count(rows)
    if raw_count:
        failures.append("context_firewall")

    all_sequences = [sha256_json({"context": row.get("context_tokens") or [], "target": row.get("target_tokens") or []}) for row in rows]
    axis_report: dict[str, Any] = {}
    for axis, spec in axes.items():
        field_specs = list(spec.get("fields") or [])
        presence_key = str(spec.get("presence_token", ""))
        signatures: list[str] = []
        token_maps: list[dict[str, list[str]]] = []
        for row in rows:
            token_map = _token_map(row.get("context_tokens") or [])
            token_maps.append(token_map)
            signatures.append(_axis_signature(axis, spec, token_map))
        presence_values = [_safe_field_value(_first(token_map, presence_key)) for token_map in token_maps]
        field_report: dict[str, Any] = {}
        for field in field_specs:
            key = _field_key(axis, str(field))
            values: list[str] = []
            missing = 0
            status_counts: Counter[str] = Counter()
            for token_map in token_maps:
                value = _first(token_map, key)
                if value is None:
                    missing += 1
                    status_counts["missing"] += 1
                    continue
                safe = _safe_field_value(value)
                values.append(safe)
                status_counts[safe] += 1
            field_report[str(field)] = {
                **_field_metric(values, rows=len(rows), missing=missing, status_counts=status_counts),
                "ablation": _field_ablation(rows, key),
            }
            if missing:
                failures.append(f"field_missing:{axis}.{field}")
        presence_missing = sum(value == "missing" for value in presence_values)
        if presence_missing:
            failures.append(f"axis_presence_missing:{axis}")
        if any(value in {"not_observed", "unknown", "missing"} for value in presence_values):
            failures.append(f"axis_not_complete:{axis}")
        axis_report[axis] = {
            "presence_token": presence_key,
            "presence": _entropy(presence_values),
            "presence_missing_rows": presence_missing,
            "entropy": _entropy(signatures),
            "sequence": _sequence_metrics(signatures),
            "unique_sequence_count": len(set(signatures)),
            "unique_sequence_ratio": round(len(set(signatures)) / max(len(signatures), 1), 6) if signatures else None,
            "field_count": len(field_specs),
            "fields": field_report,
            "field_ablation": _axis_ablation(rows, axis, spec),
            "status": "measured" if signatures and not presence_missing else "incomplete",
        }

    vocabulary_report, vocabulary_failures = _vocabulary_audit(rows, ontology if isinstance(ontology, Mapping) else {}, vocabulary)
    failures.extend(vocabulary_failures)
    capacity_report, capacity_failures = _capacity_audit(capacity, rows)
    failures.extend(capacity_failures)

    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    structural_incomplete = any(value == "incomplete" for value in axis_report.values())
    if structural_incomplete:
        failures.append("axis_information_incomplete")
    # ``diagnostic`` is intentionally the best state.  Even a complete row
    # set remains evaluator-side evidence until an independent training gate
    # approves it; this script never grants that authority.
    diagnostic_only = {"single_implementation_diagnostic"}
    hard_failures = [item for item in failures if item not in diagnostic_only]
    hard_block = any(item in {"missing:dataset", "missing:ontology", "missing:ontology_axes"} or item.startswith("empty:") for item in hard_failures)
    status = "blocked" if hard_block else "incomplete" if hard_failures else "diagnostic"
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "dataset": dataset_path,
        "ontology": ontology_path,
        "vocabulary": vocabulary_path,
        "capacity": capacity_path,
        "input_sha256": {
            "dataset": sha256_json(document) if document is not None else "",
            "ontology": sha256_json(ontology) if ontology is not None else "",
            "vocabulary": sha256_json(vocabulary) if vocabulary is not None else "",
            "capacity": sha256_json(capacity) if capacity is not None else "",
        },
        "record_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_sequence": _sequence_metrics(all_sequences),
        "unique_sequence_count": len(set(all_sequences)),
        "unique_sequence_ratio": round(len(set(all_sequences)) / max(len(all_sequences), 1), 6) if all_sequences else None,
        "axes": axis_report,
        # Alias retained for readers of the older audit report.
        "axis_quality": axis_report,
        "validation": validation,
        "context_firewall": {"forbidden_token_count": raw_count},
        "vocabulary_coverage": vocabulary_report,
        "capacity_coverage": capacity_report,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
            "operator_review_is_not_promotion": True,
            "single_implementation_is_not_training": True,
        },
        "training_eligibility": {
            "accepted_count": 0,
            "claimed_count": validation["claimed_training_eligible_count"],
            "reason": "diagnostic-only information audit; operator review and one implementation cannot authorize training",
        },
        "failures": sorted(set(failures)),
        "interpretation": "七轴/ontology field 信息保真与词表/容量覆盖仅作诊断；缺字段、typed/fresh/negative/replay 或跨实现证据时必须 ASK/incomplete，不得补默认值。",
    }
    report["dataset_information_sha256"] = ""
    report["dataset_information_sha256"] = sha256_json(report)
    return report


def audit_dataset_information(
    dataset: Path | str | Mapping[str, Any] | Sequence[Any] = DEFAULT_DATASET,
    *,
    ontology: Path | str | Mapping[str, Any] = DEFAULT_ONTOLOGY,
    vocabulary: Path | str | Mapping[str, Any] | None = DEFAULT_VOCABULARY,
    capacity: Path | str | Mapping[str, Any] | None = DEFAULT_CAPACITY,
) -> dict[str, Any]:
    """Load input paths (or use in-memory mappings) and return a diagnostic."""

    def load_input(value: Any, label: str) -> tuple[Any, str, bool]:
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.exists():
                return None, _relative(path), False
            try:
                return _load(path), _relative(path), True
            except (OSError, ValueError, json.JSONDecodeError):
                return None, _relative(path), False
        return value, f"<in_memory:{label}>", True

    dataset_value, dataset_label, dataset_ok = load_input(dataset, "dataset")
    ontology_value, ontology_label, ontology_ok = load_input(ontology, "ontology")
    vocabulary_value, vocabulary_label, vocabulary_ok = load_input(vocabulary, "vocabulary") if vocabulary is not None else (None, "", False)
    capacity_value, capacity_label, capacity_ok = load_input(capacity, "capacity") if capacity is not None else (None, "", False)
    report = audit_document(
        dataset_value if dataset_ok else {},
        ontology=ontology_value if ontology_ok else {},
        vocabulary=vocabulary_value if vocabulary_ok else None,
        capacity=capacity_value if capacity_ok else None,
        dataset_path=dataset_label,
        ontology_path=ontology_label,
        vocabulary_path=vocabulary_label,
        capacity_path=capacity_label,
    )
    if not dataset_ok:
        report["failures"] = sorted(set([*report.get("failures", []), "missing:dataset"]))
        report["status"] = "blocked"
    if not ontology_ok:
        report["failures"] = sorted(set([*report.get("failures", []), "missing:ontology"]))
        report["status"] = "blocked"
    if vocabulary is not None and not vocabulary_ok:
        report["failures"] = sorted(set([*report.get("failures", []), "missing:vocabulary"]))
        if report["status"] != "blocked":
            report["status"] = "incomplete"
    if capacity is not None and not capacity_ok:
        report["failures"] = sorted(set([*report.get("failures", []), "missing:capacity"]))
        if report["status"] != "blocked":
            report["status"] = "incomplete"
    report["dataset_information_sha256"] = ""
    report["dataset_information_sha256"] = sha256_json(report)
    return report


# Short aliases make the script convenient for focused tests and callers that
# already use the older audit modules' ``audit`` convention.
audit_dataset = audit_dataset_information
audit = audit_dataset_information


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-331 typed source-row information without Docker/GPU")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--vocab", "--vocabulary", dest="vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--capacity", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_dataset_information(args.dataset, ontology=args.ontology, vocabulary=args.vocabulary, capacity=args.capacity)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else f"{report['status']}: {', '.join(report.get('failures') or []) or 'diagnostic only'}")
    # A diagnostic audit is useful output but is never a training success.
    return 0 if report["status"] == "diagnostic" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "audit",
    "audit_dataset",
    "audit_dataset_information",
    "audit_document",
    "sha256_json",
]
