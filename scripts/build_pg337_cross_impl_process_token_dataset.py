"""Build a bounded cross-implementation process-token diagnostic dataset.

PG-336 supplies source-grounded Pikachu failure/ASK/negative process rows.
PG-337 supplies a real DVWA failure->repair->abstain replay.  The output
contains abstract context/target tokens only; implementation, route and
evaluator details remain in hash-bound metadata and never enter context.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PG336 = ROOT / "research" / "pg336_real_failure_process_token_v1.json"
PG337 = ROOT / "research" / "pg337_dvwa_failure_repair_source_rows_v1.json"
OUTPUT = ROOT / "research" / "pg337_cross_impl_process_token_v1.json"

SCHEMA_VERSION = "pg337-cross-impl-process-token-v1"
FORBIDDEN = ("family=", "implementation=", "route=", "route_literal=", "source=", "image=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=", "response_body_text=", "oracle=", "evaluator=", "canary=")
SAFE_CONTEXT_KEYS = frozenset({
    "typed_available", "feedback_state", "replay_ready", "evidence_present", "negative_control", "fresh_reset",
    "surface_method", "surface_field_role", "surface_encoding", "history_action", "failure_class", "step_budget",
    "process_stage", "missing_slot", "missing_count", "observation_state", "review_required",
})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_map(tokens: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in tokens:
        token = str(raw)
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def _safe_context(tokens: list[str]) -> list[str]:
    output = ["[BOS]"]
    for raw in tokens:
        token = str(raw)
        folded = token.casefold()
        if token in {"[BOS]", "[EOS]"}:
            continue
        if any(fragment in folded for fragment in FORBIDDEN):
            raise ValueError(f"forbidden context token: {token}")
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in SAFE_CONTEXT_KEYS:
            output.append(f"{key}={value}")
    output.append("[EOS]")
    return list(dict.fromkeys(output))


def _normalize_target(tokens: list[str]) -> list[str]:
    values = _token_map(tokens)
    question = str(values.get("question", "none"))
    if question in {"ask_typed_availability", "ask_typed"}:
        question = "ask_typed"
    elif question == "review_negative":
        question = "review_negative"
    elif values.get("next_action") in {"repair_abstract_plan", "repair"}:
        question = "ask_failure"
    else:
        question = "none"
    action = str(values.get("next_action", "abstain"))
    action = {"assemble_abstract_plan": "assemble_rule_ir", "assemble_rule_ir": "assemble_rule_ir", "repair_abstract_plan": "repair", "repair": "repair", "ask_typed": "ask_typed", "abstain": "abstain", "send_probe": "send_probe"}.get(action, "abstain")
    repair = str(values.get("repair_action", "none"))
    repair = "observe" if repair in {"retry_bounded_variant", "observe"} else "none" if repair == "none" else "unknown"
    failure = str(values.get("failure_class", "none"))
    failure = {"effect_not_confirmed": "candidate_without_typed_effect", "environment_response_mismatch": "candidate_without_typed_effect", "negative_control_clear": "negative_control_clear", "missing_observation": "missing_observation"}.get(failure, failure if failure else "none")
    changed = "1" if str(values.get("action_changed", "0")) in {"1", "true", "True"} else "0"
    safe = "1" if str(values.get("safe_to_send", "0")) in {"1", "true", "True"} else "0"
    return ["[TARGET_BOS]", f"question={question}", f"next_action={action}", f"repair_action={repair}", f"failure_class={failure}", f"action_changed={changed}", f"safe_to_send={safe}", "[TARGET_EOS]"]


def _manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("field_capture_manifest")
    if isinstance(value, Mapping):
        normalized = dict(value)
        # PG-331 whole-page rows call this axis belief_and_replay; the
        # process-token contract uses the shorter historical alias.  The
        # rename is metadata-only and does not alter context tokens.
        if "belief_and_replay" in normalized and "belief_replay" not in normalized:
            normalized["belief_replay"] = normalized.pop("belief_and_replay")
        return normalized
    return {axis: {"presence": "not_observed"} for axis in ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_replay")}


def _base_record(*, context: list[str], target: list[str], split: str, source_ref: str, kind: str, metadata: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if any(any(fragment in token.casefold() for fragment in FORBIDDEN) for token in context):
        raise ValueError("context firewall failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"pg337-process-{_sha({'source': source_ref, 'kind': kind})[:24]}",
        "split": split,
        "diagnostic_kind": kind,
        "source_grounded": True,
        "synthetic_counterfactual": False,
        "source_record_sha256": source_ref,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
        "context_tokens": context,
        "target_tokens": target,
        "field_capture_manifest": dict(manifest),
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "process_metadata": dict(metadata),
        "evaluator_sidecar_ref": {"evidence_hash_present": bool(metadata.get("evidence_hash_present_sidecar")), "source_record_sha256": source_ref},
    }


def build_dataset(*, pg336_path: Path = PG336, pg337_path: Path = PG337) -> dict[str, Any]:
    old = json.loads(pg336_path.read_text(encoding="utf-8-sig"))
    dvwa = json.loads(pg337_path.read_text(encoding="utf-8-sig"))
    old_rows = list(old.get("records") or [])
    dvwa_rows = list(dvwa.get("records") or [])
    if not old_rows or not dvwa_rows:
        raise ValueError("PG-337 requires PG-336 and PG-337 records")
    records: list[dict[str, Any]] = []
    for row in old_rows:
        split = "train" if row.get("split") == "train" else "implementation_holdout"
        metadata = dict(row.get("process_metadata") or {})
        metadata.update({"source_track": "pg336_pikachu", "independent_implementation_holdout": split == "implementation_holdout", "source_dataset_sha256": str(old.get("dataset_sha256", ""))})
        record = _base_record(context=_safe_context(list(row.get("context_tokens") or [])), target=_normalize_target(list(row.get("target_tokens") or [])), split=split, source_ref=str(row.get("source_record_sha256", "")), kind=str(row.get("diagnostic_kind", "unknown")), metadata=metadata, manifest=_manifest(row))
        records.append(record)
    for row in dvwa_rows:
        target_values = _token_map(list(row.get("target_tokens") or []))
        role = "negative" if str(row.get("record_id", "")).endswith("-negative") else "candidate"
        kind = "negative_review" if role == "negative" else "failure_repair"
        context = [
            "[BOS]", "typed_available=1", "feedback_state=negative_control_clear" if role == "negative" else "observable_progress",
            "replay_ready=1", "evidence_present=1", "negative_control=1", "fresh_reset=1", "surface_method=POST",
            "surface_field_role=stored_text", "surface_encoding=form_urlencoded", "history_action=candidate_failed",
            "failure_class=candidate_without_typed_effect", "step_budget=present", "process_stage=" + kind,
            "failure_observed=1", "[EOS]",
        ]
        target = _normalize_target(list(row.get("target_tokens") or []))
        # Strict PG-331 source rows carry action transition in the failure
        # axis rather than in target_projection.  Make the process target
        # explicit without copying any evaluator literal.
        if "action_changed=1" not in target:
            target.insert(-1, "action_changed=1")
        metadata = {"source_track": "pg337_dvwa", "independent_implementation_holdout": True, "real_failure_trace": role != "negative", "real_negative_evaluator_trace": role == "negative", "real_ask_preflight": False, "evidence_hash_present_sidecar": True, "source_dataset_sha256": str(dvwa.get("dataset_sha256", "")), "target_next_action_observed": str(target_values.get("next_action", ""))}
        records.append(_base_record(context=context, target=target, split="implementation_holdout", source_ref=_sha(row), kind=kind, metadata=metadata, manifest=_manifest(row)))
    counts = {"total": len(records), "train": sum(int(r["split"] == "train") for r in records), "implementation_holdout": sum(int(r["split"] == "implementation_holdout") for r in records), "failure_repair": sum(int(r["diagnostic_kind"] == "failure_repair") for r in records), "negative_review": sum(int(r["diagnostic_kind"] == "negative_review") for r in records), "ask_preflight": sum(int(r["diagnostic_kind"] == "ask_preflight") for r in records), "real_dvwa_failure_rows": sum(int(r["process_metadata"].get("source_track") == "pg337_dvwa" and r["diagnostic_kind"] == "failure_repair") for r in records)}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_only_cross_implementation",
        "purpose": "cross-implementation process-token diagnostic with real DVWA failure-repair and abstain",
        "source": {"pg336_sha256": _file_sha(pg336_path), "pg337_sha256": _file_sha(pg337_path), "independent_implementation_holdout": True, "train_track": "pg336_pikachu", "holdout_track": "pg337_dvwa"},
        "records": records,
        "counts": counts,
        "context_tokens": sorted({token for row in records for token in row["context_tokens"]}),
        "target_tokens": sorted({token for row in records for token in row["target_tokens"]}),
        "process_policy": {"real_failure_required": True, "real_dvwa_failure_rows": counts["real_dvwa_failure_rows"], "synthetic_counterfactual_allowed": False, "raw_context_allowed": False, "accepted_training_rows": 0},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    result["dataset_sha256"] = _sha(result)
    return result


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build_dataset(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "diagnostic_only_cross_implementation", "output": str(OUTPUT), "sha256": _file_sha(OUTPUT)}, ensure_ascii=False))
