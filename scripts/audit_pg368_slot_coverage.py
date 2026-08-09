"""Read-only PG-368 target-slot coverage audit.

The PG-367 v2 target-token inventory is the reference schema.  This audit
compares it with the PG-368 plan and the existing PG-333/PG-337/PG-342 source
rows without editing or reclassifying any row.  It never starts Docker,
imports a model runtime, opens a network connection, or copies wire values.

The report is intentionally bounded: it contains counts and SHA-256 digests
of abstract value sets, not raw payloads, URLs, request/response bodies, or
evaluator answer literals.  A source row missing a required slot remains
``blocked``/``incomplete``; the auditor does not synthesize a value from a
nearby field (for example, ``js_syntax_shape`` is not
``syntax_category_ref``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "pg368-slot-coverage-audit-v1"
REFERENCE_PATH = ROOT / "research" / "pg367_waf_staircase_dataset_v2.json"
DATASET_PATHS = {
    "pg368_plan": ROOT / "research" / "pg368_second_implementation_plan_v1.json",
    "pg333_webgoat": ROOT / "research" / "pg333_webgoat_typed_method_shape_source_rows_v1.json",
    "pg337_dvwa": ROOT / "research" / "pg337_dvwa_failure_repair_source_rows_v1.json",
    "pg342_webgoat": ROOT / "research" / "pg342_webgoat_failure_repair_source_rows_v1.json",
}

# These are the slots required to decode a target action.  They are names,
# not literal payloads.  ``next_action``/``repair_action`` are also checked as
# the failure-repair axis requested by PG-368.
REQUIRED_SLOTS = (
    "syntax_category_ref",
    "payload_shape_ref",
    "encoding_ref",
    "field_role_ref",
    "oracle_ref",
    "next_action",
    "repair_action",
)
FAILURE_CONTEXT_SLOTS = (
    "failure_failure_class",
    "failure_failure_stage",
    "failure_error_shape",
    "failure_next_action",
    "failure_repair_delta_axis",
    "failure_repair_outcome",
)
ROLES = ("candidate", "reference", "negative", "replay")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = {
    "payload",
    "raw_payload",
    "probe_value",
    "raw_value",
    "request_body",
    "response_body",
    "raw_response",
    "url",
    "uri",
    "evaluator_answer",
}

# These are the only model-facing values this audit recommends preserving.
# They are abstract slot names; evaluator evidence remains sidecar-only.
SAFE_ABSTRACT_PROJECTION = (
    "transport_ref",
    "request_method",
    "request_placement",
    "encoding_ref",
    "field_role_ref",
    "response_shape",
    "redirect_shape",
    "failure_class",
    "failure_stage",
    "error_shape",
    "previous_action",
    "next_action",
    "repair_delta_axis",
    "repair_outcome",
    "belief_delta_axis",
    "replay_state",
    "typed_available_presence",
    "evidence_hash_presence",
)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _token_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float, bool))]


def _slot_values(tokens: Sequence[str], slot: str) -> list[str]:
    prefix = slot + "="
    return [token.split("=", 1)[1] for token in tokens if token.startswith(prefix)]


def _value_digest(values: Sequence[str]) -> str:
    # Values are abstract enums/buckets.  Persist only a digest and count.
    return sha256_json(sorted(set(str(value) for value in values)))


def _contains_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if key_text in _FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_contains_forbidden_keys(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_contains_forbidden_keys(item, f"{path}[{index}]"))
    return found


def _slot_stat(*, exact_values: Sequence[str], projection_values: Sequence[str], alias_values: Sequence[str], units: int) -> dict[str, Any]:
    exact = len(exact_values)
    projection = len(projection_values)
    aliases = len(alias_values)
    return {
        "units": int(units),
        "exact_target_count": exact,
        "exact_target_rate": round(exact / units, 6) if units else 0.0,
        "projection_count": projection,
        "projection_rate": round(projection / units, 6) if units else 0.0,
        "abstract_alias_count": aliases,
        "abstract_alias_rate": round(aliases / units, 6) if units else 0.0,
        "missing_exact_count": max(0, units - exact),
        "distinct_value_count": len(set(str(value) for value in exact_values)),
        "distinct_value_sha256": _value_digest(exact_values),
    }


def _empty_slot_map(units: int) -> dict[str, dict[str, Any]]:
    return {
        slot: _slot_stat(exact_values=(), projection_values=(), alias_values=(), units=units)
        for slot in REQUIRED_SLOTS
    }


def _audit_reference() -> dict[str, Any]:
    reference = _load(REFERENCE_PATH)
    records = list(reference.get("records") or [])
    target_values: dict[str, list[str]] = {slot: [] for slot in REQUIRED_SLOTS}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        tokens = _token_list(record.get("target_tokens"))
        for slot in REQUIRED_SLOTS:
            target_values[slot].extend(_slot_values(tokens, slot))
    slot_schema = {
        slot: {
            "required": True,
            "reference_exact_count": len(values),
            "reference_distinct_value_count": len(set(values)),
            "reference_distinct_value_sha256": _value_digest(values),
        }
        for slot, values in target_values.items()
    }
    missing = [slot for slot, values in target_values.items() if not values]
    return {
        "dataset": "pg367_waf_staircase_v2",
        "path": "research/pg367_waf_staircase_dataset_v2.json",
        "file_sha256": sha256_file(REFERENCE_PATH),
        "record_count": len(records),
        "target_slot_schema": slot_schema,
        "missing_reference_slots": missing,
        "reference_status": "passed" if not missing else "blocked",
    }


def _audit_source_rows(name: str, path: Path) -> dict[str, Any]:
    data = _load(path)
    records = [record for record in list(data.get("records") or []) if isinstance(record, Mapping)]
    units = len(records)
    slot_data: dict[str, dict[str, Any]] = {}
    alias_by_slot = {
        "field_role_ref": "parameter_role_ref",
    }
    for slot in REQUIRED_SLOTS:
        exact_values: list[str] = []
        projection_values: list[str] = []
        alias_values: list[str] = []
        for record in records:
            target_tokens = _token_list(record.get("target_tokens"))
            exact_values.extend(_slot_values(target_tokens, slot))
            projection = record.get("target_projection")
            if isinstance(projection, Mapping) and projection.get(slot) not in (None, ""):
                projection_values.append(str(projection.get(slot)))
            alias = alias_by_slot.get(slot)
            if alias and isinstance(projection, Mapping) and projection.get(alias) not in (None, ""):
                alias_values.append(str(projection.get(alias)))
        slot_data[slot] = _slot_stat(
            exact_values=exact_values,
            projection_values=projection_values,
            alias_values=alias_values,
            units=units,
        )

    context_data: dict[str, dict[str, Any]] = {}
    for slot in FAILURE_CONTEXT_SLOTS:
        values: list[str] = []
        for record in records:
            values.extend(_slot_values(_token_list(record.get("context_tokens")), slot))
        context_data[slot] = {
            "units": units,
            "observed_count": len(values),
            "observed_rate": round(len(values) / units, 6) if units else 0.0,
            "distinct_value_count": len(set(values)),
            "distinct_value_sha256": _value_digest(values),
        }

    sidecars = [record.get("evaluator_sidecar") for record in records]
    typed_available = sum(bool(isinstance(sidecar, Mapping) and sidecar.get("typed_available")) for sidecar in sidecars)
    evidence_present = sum(bool(isinstance(sidecar, Mapping) and _HEX64.fullmatch(str(sidecar.get("evidence_hash", "")))) for sidecar in sidecars)
    confirmed_positive = sum(bool(isinstance(sidecar, Mapping) and sidecar.get("confirmed_positive")) for sidecar in sidecars)
    observed_repairs = 0
    for record in records:
        projection = record.get("target_projection")
        if isinstance(projection, Mapping) and projection.get("next_action") == "repair" and projection.get("repair_action") == "observe":
            observed_repairs += 1
    methods = Counter()
    for record in records:
        method_values = _slot_values(_token_list(record.get("context_tokens")), "request_method")
        if method_values:
            methods.update(method_values[-1:])
    failures: list[str] = []
    for slot in ("syntax_category_ref", "payload_shape_ref", "oracle_ref"):
        if slot_data[slot]["exact_target_count"] == 0:
            failures.append(f"missing_exact_target_slot:{slot}")
    if typed_available and slot_data["oracle_ref"]["exact_target_count"] == 0:
        failures.append("typed_oracle_sidecar_not_model_slot")
    if not observed_repairs:
        failures.append("missing_observed_failure_repair")
    if not data.get("promotion", {}).get("training_allowed", False):
        failures.append("promotion_training_closed")
    source_training_eligible_count = sum(bool(record.get("training_eligible")) for record in records)
    if source_training_eligible_count and not data.get("promotion", {}).get("training_allowed", False):
        failures.append("source_rows_marked_training_eligible_but_promotion_closed")
    return {
        "dataset": name,
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "unit_kind": "source_row",
        "unit_count": units,
        "methods": dict(methods),
        "target_slot_coverage": slot_data,
        "failure_context_coverage": context_data,
        "evaluator_sidecar": {
            "typed_available_count": typed_available,
            "typed_available_rate": round(typed_available / units, 6) if units else 0.0,
            "evidence_hash_count": evidence_present,
            "evidence_hash_rate": round(evidence_present / units, 6) if units else 0.0,
            "confirmed_positive_count": confirmed_positive,
            "confirmed_positive_rate": round(confirmed_positive / units, 6) if units else 0.0,
            "model_context_allowed": False,
        },
        "failure_repair": {
            "observed_repair_count": observed_repairs,
            "observed_repair_rate": round(observed_repairs / units, 6) if units else 0.0,
            "model_context_allowed": True,
        },
        "blocked_reasons": failures,
        "source_training_eligible_count": source_training_eligible_count,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def _audit_pg368_plan(path: Path) -> dict[str, Any]:
    """Audit the plan's 24 role contracts without treating them as observations."""

    data = _load(path)
    units: list[tuple[Mapping[str, Any], Mapping[str, Any], list[str], Mapping[str, Any] | None, str]] = []
    methods = Counter()
    for episode in list(data.get("episodes") or []):
        if not isinstance(episode, Mapping):
            continue
        method = str(episode.get("method", "")).casefold()
        methods[method] += len(dict(episode.get("roles") or {}))
        for role in ROLES:
            contract = dict(dict(episode.get("roles") or {}).get(role) or {})
            units.append((contract, dict(contract.get("model_projection") or {}), [], contract.get("typed_oracle"), "planned_unobserved"))
    count = len(units)
    slot_data: dict[str, dict[str, Any]] = {}
    alias_by_slot = {"field_role_ref": "parameter_role_ref"}
    for slot in REQUIRED_SLOTS:
        exact_values: list[str] = []
        projection_values: list[str] = []
        alias_values: list[str] = []
        for _contract, projection, _tokens, _oracle, _status in units:
            if projection.get(slot) not in (None, ""):
                projection_values.append(str(projection.get(slot)))
            alias = alias_by_slot.get(slot)
            if alias and projection.get(alias) not in (None, ""):
                alias_values.append(str(projection.get(alias)))
        slot_data[slot] = _slot_stat(exact_values=exact_values, projection_values=projection_values, alias_values=alias_values, units=count)
    failures = [f"missing_exact_target_slot:{slot}" for slot in ("syntax_category_ref", "payload_shape_ref", "oracle_ref")]
    failures.append("typed_oracle_planned_unobserved")
    failures.append("missing_observed_failure_repair")
    if data.get("status") != "planning_only":
        failures.append("plan_not_planning_only")
    return {
        "dataset": "pg368_plan",
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256_file(path),
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "unit_kind": "planned_role_contract",
        "unit_count": count,
        "methods": dict(methods),
        "target_slot_coverage": slot_data,
        "failure_context_coverage": {
            slot: {"units": count, "observed_count": 0, "observed_rate": 0.0, "distinct_value_count": 0, "distinct_value_sha256": _value_digest(())}
            for slot in FAILURE_CONTEXT_SLOTS
        },
        "evaluator_sidecar": {
            "typed_available_count": 0,
            "typed_available_rate": 0.0,
            "evidence_hash_count": 0,
            "evidence_hash_rate": 0.0,
            "confirmed_positive_count": 0,
            "confirmed_positive_rate": 0.0,
            "planned_unobserved_count": count,
            "model_context_allowed": False,
        },
        "failure_repair": {
            "observed_repair_count": 0,
            "observed_repair_rate": 0.0,
            "planned_repair_count": 0,
            "model_context_allowed": True,
        },
        "blocked_reasons": failures,
        "source_training_eligible_count": 0,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }


def build_pg368_slot_coverage_audit() -> dict[str, Any]:
    """Build a deterministic, read-only audit report."""

    reference = _audit_reference()
    datasets = [_audit_pg368_plan(DATASET_PATHS["pg368_plan"])]
    datasets.extend(_audit_source_rows(name, path) for name, path in DATASET_PATHS.items() if name != "pg368_plan")
    all_failures: list[str] = []
    for dataset in datasets:
        all_failures.extend(f"{dataset['dataset']}:{reason}" for reason in dataset["blocked_reasons"])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if all_failures or reference["reference_status"] != "passed" else "passed",
        "reference": reference,
        "datasets": datasets,
        "required_slots": list(REQUIRED_SLOTS),
        "safe_abstract_projection": {
            "allowed_slots": list(SAFE_ABSTRACT_PROJECTION),
            "evaluator_sidecar_off_context": True,
            "unsafe_literal_presence": False,
            "transport_literal_presence": False,
            "route_literal_presence": False,
            "reclassification_performed": False,
            "new_training_rows_generated": False,
        },
        "blocked_reasons": all_failures,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "interpretation": (
            "PG-367 v2 supplies the target-slot reference. PG-333/337/342 are audited as-is: "
            "missing exact slots remain missing, evaluator-side typed evidence does not become model input, "
            "and no old row is reclassified or promoted."
        ),
    }
    report["report_sha256"] = sha256_json(report)
    return report


def _assert_no_forbidden_keys(value: Any) -> None:
    forbidden = _contains_forbidden_keys(value)
    if forbidden:
        raise ValueError("forbidden keys in PG-368 slot report: " + ", ".join(forbidden))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "research" / "pg368_slot_coverage_audit_v1.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_pg368_slot_coverage_audit()
    _assert_no_forbidden_keys(report)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
