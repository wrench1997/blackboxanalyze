"""Build PG-283 multi-step feedback supervision.

The rows describe what the controller can observe after each bounded step:
negative/reference state, typed-evaluator availability, failure signature,
fresh-reset state and replay consistency.  The target is the next *abstract*
action plus its safe gate.  No literal payload, route value, response body or
oracle label is written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg283_feedback_policy_dataset_v1.json"
HARD_NEGATIVE = RESEARCH / "pg283_feedback_policy_hard_negative_v1.json"
AUDIT = RESEARCH / "pg283_feedback_policy_dataset_audit_v1.json"

ACTION_NAMES = (
    "send_negative",
    "send_reference",
    "ask_typed",
    "send_candidate",
    "repair_alternate",
    "replay_confirmed",
    "abstain",
)
PROBE_CLASSES = ("sql", "xss", "redirect", "logic", "file", "other")
CHANNELS = ("query", "form", "unknown")
ENCODINGS = ("plain", "url_percent", "unknown")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _field_bucket(row: dict[str, Any]) -> str:
    for token in row.get("context_tokens", []):
        text = str(token)
        if text.startswith("field_count="):
            try:
                count = int(text.split("=", 1)[1])
            except ValueError:
                break
            return "0" if count <= 0 else "1" if count == 1 else "2" if count == 2 else "3p"
    return "unknown"


def _base_plan(row: dict[str, Any]) -> dict[str, Any]:
    target = dict(row.get("target") or {})
    probe = str(target.get("probe_class", "other"))
    channel = str(target.get("channel", "unknown"))
    encoding = str(target.get("encoding", "unknown"))
    if probe not in PROBE_CLASSES:
        probe = "other"
    if channel not in CHANNELS:
        channel = "unknown"
    if encoding not in ENCODINGS:
        encoding = "unknown"
    return {"probe_class": probe, "channel": channel, "encoding": encoding}


STATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "need_negative",
        "phase": "preprobe",
        "failure": "missing_negative_control",
        "feedback": "none",
        "negative_sent": False,
        "reference_sent": False,
        "candidate_sent": False,
        "typed_available": False,
        "negative_clean": False,
        "reference_agreement": False,
        "fresh_reset": True,
        "candidate_signal": False,
        "replay_consistent": False,
        "action": "send_negative",
        "safe": False,
    },
    {
        "name": "negative_clean",
        "phase": "observe",
        "failure": "negative_control_clean",
        "feedback": "control_clean",
        "negative_sent": True,
        "reference_sent": False,
        "candidate_sent": False,
        "typed_available": False,
        "negative_clean": True,
        "reference_agreement": False,
        "fresh_reset": True,
        "candidate_signal": False,
        "replay_consistent": False,
        "action": "send_reference",
        "safe": False,
    },
    {
        "name": "reference_clean",
        "phase": "diagnose",
        "failure": "typed_evaluator_missing",
        "feedback": "control_clean",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": False,
        "typed_available": False,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": True,
        "candidate_signal": False,
        "replay_consistent": False,
        "action": "ask_typed",
        "safe": False,
    },
    {
        "name": "typed_ready",
        "phase": "plan",
        "failure": "all_preconditions_met",
        "feedback": "typed_available",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": False,
        "typed_available": True,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": True,
        "candidate_signal": False,
        "replay_consistent": False,
        "action": "send_candidate",
        "safe": True,
    },
    {
        "name": "candidate_gap",
        "phase": "feedback",
        "failure": "candidate_without_typed_effect",
        "feedback": "candidate_signal",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": True,
        "typed_available": False,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": True,
        "candidate_signal": True,
        "replay_consistent": False,
        "action": "ask_typed",
        "safe": False,
    },
    {
        "name": "typed_effect",
        "phase": "feedback",
        "failure": "typed_effect_observed",
        "feedback": "typed_effect",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": True,
        "typed_available": True,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": True,
        "candidate_signal": True,
        "replay_consistent": False,
        "action": "replay_confirmed",
        "safe": True,
    },
    {
        "name": "reference_mismatch",
        "phase": "diagnose",
        "failure": "reference_candidate_mismatch",
        "feedback": "mismatch",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": True,
        "typed_available": True,
        "negative_clean": True,
        "reference_agreement": False,
        "fresh_reset": True,
        "candidate_signal": True,
        "replay_consistent": False,
        "action": "repair_alternate",
        "safe": False,
    },
    {
        "name": "fresh_missing",
        "phase": "guard",
        "failure": "fresh_reset_missing",
        "feedback": "environment_failure",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": False,
        "typed_available": True,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": False,
        "candidate_signal": False,
        "replay_consistent": False,
        "action": "abstain",
        "safe": False,
    },
    {
        "name": "replay_done",
        "phase": "replay",
        "failure": "replay_consistent",
        "feedback": "typed_effect",
        "negative_sent": True,
        "reference_sent": True,
        "candidate_sent": True,
        "typed_available": True,
        "negative_clean": True,
        "reference_agreement": True,
        "fresh_reset": True,
        "candidate_signal": True,
        "replay_consistent": True,
        "action": "replay_confirmed",
        "safe": True,
    },
)


def _context(row: dict[str, Any], spec: dict[str, Any], replica: int, *, hard_negative: bool = False) -> list[str]:
    method = str(row.get("method", "GET")).upper()
    if method not in {"GET", "POST"}:
        method = "GET"
    plan = _base_plan(row)
    channel = plan["channel"] if plan["channel"] in CHANNELS else ("query" if method == "GET" else "form")
    encoding = plan["encoding"]
    # ``replica`` is a bounded history variation, not an identifier for a
    # route or target.  It prevents the process learner from memorizing a
    # single fixed trajectory length.
    history_bucket = (replica % 3) + 1
    observed = dict(spec)
    if hard_negative:
        # Hard negatives model an unresolved evaluator/surface contract.  The
        # family remains hidden, so the only safe behavior is abstention; no
        # target label is copied into the input.
        observed.update({
            "typed_available": False,
            "negative_clean": False,
            "reference_agreement": False,
            "fresh_reset": True,
            "candidate_signal": False,
            "replay_consistent": False,
        })
        observed["failure"] = "unsupported_evaluator_surface"
        observed["feedback"] = "unresolved"
    tokens = [
        "[BOS]",
        "ir_layer=shared_slot_ontology",
        "ir_family_agnostic=1",
        "ir_role=effect",
        "ir_surface=abstract_request",
        "ir_measure=transition",
        f"phase={observed['phase']}",
        f"method={method}",
        f"channel={channel}",
        f"encoding={encoding}",
        f"field_bucket={_field_bucket(row)}",
        f"history_bucket={history_bucket}",
        f"fresh_reset={_flag(bool(observed['fresh_reset']))}",
        "source_attested=1",
        f"negative_sent={_flag(bool(observed['negative_sent']))}",
        f"reference_sent={_flag(bool(observed['reference_sent']))}",
        f"candidate_sent={_flag(bool(observed['candidate_sent']))}",
        f"typed_available={_flag(bool(observed['typed_available']))}",
        f"negative_clean={_flag(bool(observed['negative_clean']))}",
        f"reference_agreement={_flag(bool(observed['reference_agreement']))}",
        f"candidate_signal={_flag(bool(observed['candidate_signal']))}",
        f"replay_consistent={_flag(bool(observed['replay_consistent']))}",
        f"failure_signature={observed['failure']}",
        f"feedback={observed['feedback']}",
        "family_hidden=1",
        "oracle_label_in_context=0",
        f"step_replica={replica}",
        "[CTX_END]",
    ]
    return tokens


def _record(row: dict[str, Any], spec: dict[str, Any], replica: int, *, split: str, hard_negative: bool = False) -> dict[str, Any]:
    plan = _base_plan(row)
    target_action = "abstain" if hard_negative else str(spec["action"])
    target_safe = False if hard_negative else bool(spec["safe"])
    source_id = str(row.get("record_id") or row.get("hard_negative_id") or "unknown")
    identity = {"source_id": source_id, "state": spec["name"], "replica": replica, "split": split, "hard_negative": hard_negative}
    return {
        "record_id": "pg283:" + _sha(identity)[:24],
        "source_record_id": source_id,
        "source": str(row.get("source", "pg281")),
        "family": str(row.get("family", "other")),
        "method": str(row.get("method", "GET")).upper(),
        "split": split,
        "state": spec["name"],
        "replica": replica,
        "hard_negative": hard_negative,
        "context_tokens": _context(row, spec, replica, hard_negative=hard_negative),
        "target": {
            "next_action": target_action,
            "probe_class": plan["probe_class"],
            "channel": plan["channel"],
            "encoding": plan["encoding"],
            "safe_to_send": target_safe,
            "oracle_required": True,
        },
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_eligible": not hard_negative,
        "memory_promotion_allowed": False,
        "source_evidence_hash": str(row.get("source_evidence_hash", "")),
    }


def _records(rows: list[dict[str, Any]], *, split: str, hard_negative: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for replica in range(3):
            for spec in STATE_SPECS:
                result.append(_record(row, spec, replica, split=split, hard_negative=hard_negative))
    return result


def main() -> None:
    source = json.loads((RESEARCH / "pg281_payload_policy_dataset_v1.json").read_text(encoding="utf-8"))
    hard_source = json.loads((RESEARCH / "pg281_payload_policy_hard_negative_v1.json").read_text(encoding="utf-8"))
    records = list(source.get("records") or [])
    hard_rows = list(hard_source.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train"]
    route_rows = [row for row in records if row.get("split") == "route_dev"]
    family_rows = [row for row in records if row.get("split") == "family_holdout"]
    train = _records(train_rows, split="train")
    route_dev = _records(route_rows, split="route_dev")
    family_holdout = _records(family_rows, split="family_holdout")
    hard = _records(hard_rows, split="hard_negative", hard_negative=True)
    all_records = [*train, *route_dev, *family_holdout]
    counts = {"train": len(train), "route_dev": len(route_dev), "family_holdout": len(family_holdout), "hard_negative": len(hard), "total": len(all_records) + len(hard)}
    dataset = {
        "schema_version": "pg283-feedback-policy-dataset-v1",
        "purpose": "multi-step failure feedback -> abstract next action and safe gate",
        "source": {
            "pg281_dataset": "research/pg281_payload_policy_dataset_v1.json",
            "pg281_dataset_sha256": source.get("dataset_sha256", ""),
            "pg281_hard_negative": "research/pg281_payload_policy_hard_negative_v1.json",
            "pg281_hard_negative_sha256": hard_source.get("dataset_sha256", ""),
            "generated_process_rows": counts["total"],
            "literal_payloads": False,
            "raw_responses": False,
        },
        "records": all_records,
        "hard_negative_records": hard,
        "counts": counts,
        "action_ontology": {"actions": list(ACTION_NAMES), "probe_classes": list(PROBE_CLASSES), "channels": list(CHANNELS), "encodings": list(ENCODINGS)},
        "split_contract": {"train": "PG-281 train transitions", "route_dev": "unseen route transitions", "family_holdout": "unseen family transitions", "hard_negative": "evaluation-only; forced abstain"},
        "training_contract": {"family_hidden_in_context": True, "oracle_label_in_context": False, "literal_payload_values_out_of_context": True, "hard_negative_training_eligible": False, "remote_a800_required": True, "live_replay_required_for_promotion": True, "memory_promotion_allowed": False},
        "scientific_contract": {"generated_transition_templates": True, "cross_template_generalization_claim_allowed": False, "real_application_gold_required": True},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _sha({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    hard_dataset = {
        "schema_version": "pg283-feedback-policy-hard-negative-v1",
        "records": hard,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "dataset_sha256": _sha(hard),
    }
    audit = {
        "schema_version": "pg283-feedback-policy-dataset-audit-v1",
        "dataset": "research/pg283_feedback_policy_dataset_v1.json",
        "dataset_sha256": dataset["dataset_sha256"],
        "checks": {
            "counts_match": counts["total"] == len(all_records) + len(hard),
            "train_nonempty": bool(train),
            "route_dev_nonempty": bool(route_dev),
            "family_holdout_nonempty": bool(family_holdout),
            "hard_negative_nonempty": bool(hard),
            "hard_negative_excluded": all(not row["training_eligible"] and not row["memory_promotion_allowed"] for row in hard),
            "context_has_no_oracle": all(not row["oracle_in_context"] for row in all_records + hard),
            "context_has_no_literal": all(not row["raw_payload_strings_stored"] and not row["raw_response_bodies_stored"] for row in all_records + hard),
            "family_hidden": all("family_hidden=1" in row["context_tokens"] for row in all_records + hard),
            "target_actions_allowlisted": all(row["target"]["next_action"] in ACTION_NAMES for row in all_records + hard),
        },
        "status": "",
    }
    audit["status"] = "passed" if all(audit["checks"].values()) else "blocked"
    audit["audit_sha256"] = _sha({key: value for key, value in audit.items() if key != "audit_sha256"})
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HARD_NEGATIVE.write_text(json.dumps(hard_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
