"""Read-only cross-implementation audit for PG-388 Rule-IR source-row artifacts.

The auditor joins the implementation-A and implementation-B wrapper files only
through bounded metadata and hashes.  It never emits source-row records,
context/target tokens, response data, payloads, wire data, or evaluator
answers.  Passing this audit is evidence of a candidate-only collection
contract, not authorization to train or claim a general vulnerability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_ROWS = ROOT / "research" / "pg388_logic_rule_ir_source_rows_live_rows_v1.json"
DEFAULT_B_ROWS = ROOT / "research" / "pg388_logic_holdout_b_source_rows_rows_v1.json"
DEFAULT_B_DOCKER = ROOT / "research" / "pg388_logic_holdout_b_docker_smoke_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg388_logic_cross_implementation_audit_v1.json"
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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _load_wrappers(path: Path) -> list[dict[str, Any]]:
    document = _load(path)
    value = document.get("rows")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"source-row wrapper list missing: {path}")
    return [item for item in value if isinstance(item, dict)]


def _source_row(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    value = wrapper.get("source_row")
    if not isinstance(value, Mapping):
        raise ValueError("source-row wrapper missing nested source row")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bool(value: Any) -> bool:
    return value is True


def _count_true(wrappers: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(1 for wrapper in wrappers if _bool(wrapper.get(key)))


def _token_digest_set(wrappers: Iterable[Mapping[str, Any]], key: str) -> set[str]:
    result: set[str] = set()
    for wrapper in wrappers:
        row = _source_row(wrapper)
        value = row.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            result.add(_digest(value))
    return result


def _assert_safe(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden audit projection key: {key}")
            _assert_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_safe(item)


def audit(
    *,
    a_rows_path: Path = DEFAULT_A_ROWS,
    b_rows_path: Path = DEFAULT_B_ROWS,
    b_docker_path: Path = DEFAULT_B_DOCKER,
) -> dict[str, Any]:
    sources = [("pg388-logic-lab-backend-a", _load_wrappers(a_rows_path)), ("pg388-logic-lab-backend-b", _load_wrappers(b_rows_path))]
    all_wrappers = [wrapper for _, wrappers in sources for wrapper in wrappers]
    implementation_counts = {implementation: len(wrappers) for implementation, wrappers in sources}
    split_counts = Counter()
    source_digest_counts = Counter()
    record_ref_counts = Counter()
    strict_valid = 0
    typed = 0
    fresh = 0
    negative_clean = 0
    negative_violation = 0
    raw_flag_failures = 0
    context_firewall_failures = 0
    target_slot_counts = Counter()
    source_contracts: dict[str, dict[str, Any]] = {}
    context_signatures: dict[str, set[str]] = {}
    target_signatures: dict[str, set[str]] = {}

    for implementation, wrappers in sources:
        context_signatures[implementation] = _token_digest_set(wrappers, "context_tokens")
        target_signatures[implementation] = _token_digest_set(wrappers, "logic_rule_ir_target_tokens")
        for wrapper in wrappers:
            row = _source_row(wrapper)
            split_counts[str(row.get("split", "unknown"))] += 1
            record_ref = str(wrapper.get("record_ref_sha256", ""))
            if record_ref:
                record_ref_counts[record_ref] += 1
            source_meta = row.get("source_meta") if isinstance(row.get("source_meta"), Mapping) else {}
            source_digest = str(source_meta.get("source_digest", ""))
            if source_digest:
                source_digest_counts[source_digest] += 1
            source_contracts[implementation] = {
                "source_id_present": bool(source_meta.get("source_id")),
                "authorization_present": bool(source_meta.get("authorization_id")),
                "row_bound_typed_evidence": _bool((row.get("evaluator_sidecar") or {}).get("typed_available")),
                "fresh_role_reset": _bool((row.get("reset") or {}).get("fresh_reset")),
                "operator_reviewed": _bool(row.get("operator_reviewed")),
                "training_eligible": _bool(row.get("training_eligible")),
            }
            strict_valid += int(_bool(wrapper.get("strict_valid")))
            typed += int(_bool((row.get("evaluator_sidecar") or {}).get("typed_available")))
            fresh += int(_bool((row.get("reset") or {}).get("fresh_reset")))
            if _bool((row.get("evaluator_sidecar") or {}).get("negative_control")):
                negative_clean += 1
            target = row.get("logic_rule_ir_target") if isinstance(row.get("logic_rule_ir_target"), Mapping) else {}
            if str(target.get("oracle_ref")) == "negative_no_effect" and str(target.get("safe_to_send")) == "True":
                negative_violation += 1
            target_tokens = row.get("logic_rule_ir_target_tokens")
            if isinstance(target_tokens, list):
                target_slot_counts[len(target_tokens)] += 1
            if row.get("raw_payload_stored") is True or row.get("raw_response_body_stored") is True or row.get("oracle_answer_in_context") is True:
                raw_flag_failures += 1
            firewall = row.get("context_firewall") if isinstance(row.get("context_firewall"), Mapping) else {}
            if int(firewall.get("forbidden_token_count", 0) or 0) != 0 or firewall.get("sidecars_off_context") is not True:
                context_firewall_failures += 1

    docker = _load(b_docker_path) if b_docker_path.exists() else {}
    observed = docker.get("observed") if isinstance(docker.get("observed"), Mapping) else {}
    execution = docker.get("execution") if isinstance(docker.get("execution"), Mapping) else {}
    context_overlap = len(context_signatures.get("pg388-logic-lab-backend-a", set()) & context_signatures.get("pg388-logic-lab-backend-b", set()))
    target_overlap = len(target_signatures.get("pg388-logic-lab-backend-a", set()) & target_signatures.get("pg388-logic-lab-backend-b", set()))
    failures: list[str] = []
    if any(split != "implementation_holdout" for split in split_counts):
        failures.append("unexpected_split")
    if "train" not in split_counts:
        failures.append("train_split_missing")
    if strict_valid != len(all_wrappers):
        failures.append("strict_invalid_rows")
    if typed != len(all_wrappers):
        failures.append("typed_evidence_missing")
    if fresh != len(all_wrappers):
        failures.append("fresh_reset_missing")
    if negative_violation:
        failures.append("negative_violation")
    if raw_flag_failures:
        failures.append("raw_context_flag")
    if context_firewall_failures:
        failures.append("context_firewall_failure")
    failures.extend(["operator_review_missing", "image_attestation_missing"])
    status = "passed_candidate_cross_implementation_logic_audit" if not any(item in failures for item in ("unexpected_split", "strict_invalid_rows", "typed_evidence_missing", "fresh_reset_missing", "negative_violation", "raw_context_flag", "context_firewall_failure")) else "blocked_cross_implementation_logic_audit"
    output = {
        "schema_version": "pg388-logic-cross-implementation-audit-v1",
        "status": status,
        "purpose": "bounded metadata audit for two PG-388 logic implementations",
        "sources": {
            "implementation_count": len(sources),
            "implementation_counts": implementation_counts,
            "split_counts": dict(sorted(split_counts.items())),
            "source_digest_groups": len(source_digest_counts),
            "duplicate_record_ref_count": sum(1 for count in record_ref_counts.values() if count > 1),
            "cross_implementation_context_signature_overlap": context_overlap,
            "cross_implementation_target_signature_overlap": target_overlap,
        },
        "coverage": {
            "source_row_count": len(all_wrappers),
            "strict_valid": strict_valid,
            "typed_evidence": typed,
            "fresh_resets": fresh,
            "negative_clean": negative_clean,
            "negative_violations": negative_violation,
            "target_slot_lengths": dict(sorted((str(key), value) for key, value in target_slot_counts.items())),
            "context_firewall_failures": context_firewall_failures,
            "raw_context_flag_failures": raw_flag_failures,
        },
        "implementation_contracts": source_contracts,
        "docker_smoke": {
            "status": str(docker.get("status", "missing")),
            "health_http_status": int(observed.get("health_http_status", 0) or 0),
            "docker_started": _bool(execution.get("docker_started")),
            "target_contacted": _bool(execution.get("target_contacted")),
            "external_network": _bool(execution.get("network_contacted")),
            "persistent_storage": _bool(observed.get("persistent_storage")),
            "negative_control_clean": _bool(observed.get("negative_control_clean")),
        },
        "hard_gate": {
            "train_split_present": "train" in split_counts,
            "implementation_holdout_present": split_counts.get("implementation_holdout", 0) > 0,
            "operator_reviewed": False,
            "image_attested": False,
            "training_eligible": 0,
            "capability_training_allowed": False,
        },
        "failures": failures,
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    _assert_safe(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-rows", type=Path, default=DEFAULT_A_ROWS)
    parser.add_argument("--b-rows", type=Path, default=DEFAULT_B_ROWS)
    parser.add_argument("--b-docker", type=Path, default=DEFAULT_B_DOCKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = audit(a_rows_path=args.a_rows, b_rows_path=args.b_rows, b_docker_path=args.b_docker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "output": str(args.output), "source_row_count": output["coverage"]["source_row_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
