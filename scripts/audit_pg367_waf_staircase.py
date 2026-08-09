"""Audit the PG-367 abstract WAF staircase dataset.

The audit checks that the dataset contains the *process* (probe, filter
observation, failure, one-axis repair and matched roles), rather than merely
answer labels.  It is diagnostic-only: even a clean audit never enables
training, memory or payload promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg367_waf_staircase import ALLOWED_METHODS, ALLOWED_ROLES, POLICIES


RAW_FRAGMENTS = (
    "payload=", "raw_payload", "response_body", "wire=", "http://", "https://",
    "<script", "select ", "union ", "oracle=", "evaluator=", "route_literal=",
)
REQUIRED_WAF_KEYS = (
    "filter_stage", "filter_action", "transform_class", "failure_signature",
    "repair_axis", "typed_effect_confirmed", "negative_control_clean",
)
AXIS_PREFIXES = (
    "request_method=", "request_field_role=", "request_encoding=",
    "failure_filter_stage=", "failure_filter_action=", "failure_transform_class=",
    "failure_signature=", "failure_repair_axis=", "failure_process_step=",
    "waf_policy_stage=", "waf_filter_observation=",
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(len(values))
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def _row_hash_without_digest(row: Mapping[str, Any]) -> str:
    unsigned = dict(row)
    unsigned.pop("record_sha256", None)
    return _sha(unsigned)


def audit(dataset: Mapping[str, Any]) -> dict[str, Any]:
    failures: set[str] = set()
    records = dataset.get("records") or []
    split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    token_values: dict[str, list[str]] = defaultdict(list)
    policy_hashes: set[str] = set()
    repair_rows = 0
    changed_repairs = 0
    failure_rows = 0
    negative_rows = 0
    negative_clean_rows = 0
    positive_rows = 0
    raw_hits = 0
    valid_rows = 0

    known_policy_hashes = {
        _sha(policy.policy_id): policy.policy_id for policy in POLICIES
    }
    for index, row in enumerate(records):
        prefix = f"row_{index}"
        if not isinstance(row, Mapping):
            failures.add(f"{prefix}:not_mapping")
            continue
        valid_rows += 1
        split = str(row.get("split", ""))
        role = str((row.get("evaluator_projection") or {}).get("role", ""))
        method = str((row.get("evaluator_projection") or {}).get("method", "")).upper()
        step = "repair" if "failure_transition" in row else "baseline"
        split_counts[split] += 1
        role_counts[role] += 1
        method_counts[method] += 1
        step_counts[step] += 1
        if split not in {"train", "implementation_holdout"}:
            failures.add(f"{prefix}:split")
        if role not in ALLOWED_ROLES:
            failures.add(f"{prefix}:role")
        if method not in ALLOWED_METHODS:
            failures.add(f"{prefix}:method")
        projection = row.get("evaluator_projection")
        if not isinstance(projection, Mapping):
            failures.add(f"{prefix}:projection")
            projection = {}
        for key in REQUIRED_WAF_KEYS:
            if key not in projection:
                failures.add(f"{prefix}:projection_{key}")
        policy_hash = str(projection.get("policy_id_hash", ""))
        policy_hashes.add(policy_hash)
        if policy_hash not in known_policy_hashes:
            failures.add(f"{prefix}:policy_hash")
        context = [str(token) for token in row.get("context_tokens") or []]
        target = [str(token) for token in row.get("target_tokens") or []]
        if not context or context[0] != "chunk_boundary=begin" or context[-1] != "chunk_boundary=end":
            failures.add(f"{prefix}:context_boundary")
        if target[:1] != ["[TARGET_BOS]"] or target[-1:] != ["[TARGET_EOS]"]:
            failures.add(f"{prefix}:target_boundary")
        missing_axes = [axis for axis in AXIS_PREFIXES if not any(token.startswith(axis) for token in context)]
        failures.update(f"{prefix}:missing_axis:{axis[:-1]}" for axis in missing_axes)
        for token in context:
            if "=" in token:
                key, value = token.split("=", 1)
                if key in {"request_method", "request_field_role", "request_encoding", "failure_filter_stage", "failure_filter_action", "failure_transform_class", "failure_signature", "failure_repair_axis", "failure_process_step", "waf_policy_stage", "waf_filter_observation"}:
                    token_values[key].append(value)
        firewall = row.get("context_firewall")
        if firewall != {"forbidden_token_count": 0, "sidecars_off_context": True}:
            failures.add(f"{prefix}:firewall")
        for flag in ("raw_payload_stored", "raw_response_body_stored", "oracle_answer_in_context"):
            if row.get(flag) is not False:
                failures.add(f"{prefix}:{flag}")
        if any(any(fragment in token.casefold() for fragment in RAW_FRAGMENTS) for token in [*context, *target]):
            raw_hits += 1
            failures.add(f"{prefix}:raw_token")
        if row.get("record_sha256") != _row_hash_without_digest(row):
            failures.add(f"{prefix}:record_hash")
        if projection.get("failure_signature") != "none":
            failure_rows += 1
        if role == "negative":
            negative_rows += 1
            negative_clean_rows += int(projection.get("negative_control_clean") is True)
        positive_rows += int(projection.get("typed_effect_confirmed") is True)
        if step == "repair":
            repair_rows += 1
            transition = row.get("failure_transition")
            if not isinstance(transition, Mapping):
                failures.add(f"{prefix}:transition")
            else:
                changed = transition.get("action_changed") is True
                changed_repairs += int(changed)
                if not changed:
                    failures.add(f"{prefix}:repair_action_unchanged")
                if transition.get("before_failure_signature") == "none":
                    failures.add(f"{prefix}:repair_without_failure")

    expected_roles = set(ALLOWED_ROLES)
    if set(role_counts) != expected_roles:
        failures.add("role_coverage")
    if set(method_counts) != set(ALLOWED_METHODS):
        failures.add("get_post_coverage")
    if not split_counts.get("train") or not split_counts.get("implementation_holdout"):
        failures.add("split_coverage")
    if repair_rows == 0 or changed_repairs != repair_rows:
        failures.add("failure_repair_coverage")
    if negative_rows == 0 or negative_clean_rows != negative_rows:
        failures.add("negative_control_clean")
    if not policy_hashes:
        failures.add("policy_coverage")
    promotion = dataset.get("promotion")
    if not isinstance(promotion, Mapping) or any(bool(promotion.get(key)) for key in ("training_allowed", "memory_promotion_allowed", "payload_catalog_promotion_allowed", "vulnerability_claim_allowed")):
        failures.add("promotion_not_fail_closed")
    failures = set(sorted(failures))
    status = "passed_diagnostic_only" if not failures else "blocked_incomplete"
    axis_entropy = {key: round(_entropy(values), 6) for key, values in sorted(token_values.items())}
    result: dict[str, Any] = {
        "schema_version": "pg367-waf-staircase-audit-v1",
        "status": status,
        "counts": {
            "records": len(records),
            "valid_rows": valid_rows,
            "train_rows": split_counts.get("train", 0),
            "implementation_holdout_rows": split_counts.get("implementation_holdout", 0),
            "get_rows": method_counts.get("GET", 0),
            "post_rows": method_counts.get("POST", 0),
            "baseline_rows": step_counts.get("baseline", 0),
            "failure_rows": failure_rows,
            "repair_rows": repair_rows,
            "repair_action_changed": changed_repairs,
            "roles": dict(sorted(role_counts.items())),
            "positive_effect_rows": positive_rows,
            "negative_rows": negative_rows,
            "negative_clean_rows": negative_clean_rows,
            "raw_hits": raw_hits,
            "training_eligible_rows": 0,
        },
        "axis_entropy_bits": axis_entropy,
        "policy_hash_count": len(policy_hashes),
        "failure_reasons": sorted(failures),
        "predictive_entropy": "not_run",
        "interpretation": "WAF过程合同通过也只证明本地抽象过程数据完整；不等于任意网址漏洞或可迁移原始payload能力。",
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    result["audit_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PG-367 WAF staircase data")
    parser.add_argument("--dataset", type=Path, default=ROOT / "research" / "pg367_waf_staircase_dataset_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "pg367_waf_staircase_audit_v1.json")
    args = parser.parse_args()
    result = audit(json.loads(args.dataset.read_text(encoding="utf-8-sig")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed_diagnostic_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
