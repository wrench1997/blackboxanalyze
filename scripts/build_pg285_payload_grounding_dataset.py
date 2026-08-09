"""Build PG-285 failure-driven structured payload-grounding data.

PG-285 is the bridge between an abstract probe policy and a human-readable
wire plan.  The model is trained to emit bounded Rule-IR slots (method,
channel, encoding, field slot and action), never a literal exploit string.
The context contains surface/response-shape tokens and failure feedback so a
repair is causally conditioned on what failed.  Literal values, raw bodies,
oracle labels and family names stay outside the model context.

This builder is deterministic and CPU-only; the actual model training runner
is remote-A800-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg281_payload_policy_dataset_v1.json"
SOURCE_HARD = RESEARCH / "pg281_payload_policy_hard_negative_v1.json"
DATASET = RESEARCH / "pg285_payload_grounding_dataset_v1.json"
HARD = RESEARCH / "pg285_payload_grounding_hard_negative_v1.json"
AUDIT = RESEARCH / "pg285_payload_grounding_dataset_audit_v1.json"

ACTION_NAMES = (
    "negative_control",
    "reference_probe",
    "candidate_probe",
    "repair_alternate",
    "replay_confirmed",
    "abstain",
)
PROBE_CLASSES = ("sql", "xss", "redirect", "logic", "file", "ssrf", "rce", "other")
CHANNELS = ("query", "form", "header", "path", "unknown")
ENCODINGS = ("plain", "url_percent", "html_entity", "json", "unknown")
WIRE_KINDS = ("query_param", "form_field", "header_value", "path_segment", "none")
METHODS = ("GET", "POST")

SURFACE_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("form_fetch", ("markup=form", "layout=table", "js=fetch", "response=reflective", "redirect=none")),
    ("card_xhr", ("markup=card", "layout=grid", "js=xhr", "response=delayed", "redirect=none")),
    ("search_jquery", ("markup=search", "layout=list", "js=jquery", "response=error_shape", "redirect=one")),
    ("detail_native", ("markup=detail", "layout=stack", "js=native", "response=opaque", "redirect=two")),
    ("wizard_json", ("markup=wizard", "layout=stepper", "js=json_fetch", "response=json", "redirect=one")),
    ("legacy_table", ("markup=legacy", "layout=table", "js=inline_handler", "response=reflective", "redirect=none")),
)

STATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "preflight",
        "phase": "preprobe",
        "failure": "missing_negative_control",
        "feedback": "none",
        "last_action": "none",
        "negative_clean": False,
        "reference_agreement": False,
        "typed_available": False,
        "candidate_sent": False,
        "fresh_reset": True,
        "replay_consistent": False,
        "action": "negative_control",
        "safe": False,
        "repair_delta": "none",
    },
    {
        "name": "negative_clean",
        "phase": "observe",
        "failure": "negative_control_clean",
        "feedback": "control_clean",
        "last_action": "negative_control",
        "negative_clean": True,
        "reference_agreement": False,
        "typed_available": False,
        "candidate_sent": False,
        "fresh_reset": True,
        "replay_consistent": False,
        "action": "reference_probe",
        "safe": False,
        "repair_delta": "none",
    },
    {
        "name": "reference_aligned",
        "phase": "diagnose",
        "failure": "reference_agreement",
        "feedback": "reference_clean",
        "last_action": "reference_probe",
        "negative_clean": True,
        "reference_agreement": True,
        "typed_available": True,
        "candidate_sent": False,
        "fresh_reset": True,
        "replay_consistent": False,
        "action": "candidate_probe",
        "safe": True,
        "repair_delta": "none",
    },
    {
        "name": "candidate_failed",
        "phase": "feedback",
        "failure": "candidate_no_typed_effect",
        "feedback": "candidate_gap",
        "last_action": "candidate_probe",
        "negative_clean": True,
        "reference_agreement": True,
        "typed_available": False,
        "candidate_sent": True,
        "fresh_reset": True,
        "replay_consistent": False,
        "action": "repair_alternate",
        "safe": False,
        "repair_delta": "encoding_or_channel",
    },
    {
        "name": "repair_candidate",
        "phase": "repair",
        "failure": "bounded_repair_ready",
        "feedback": "minimal_delta",
        "last_action": "repair_alternate",
        "negative_clean": True,
        "reference_agreement": True,
        "typed_available": True,
        "candidate_sent": False,
        "fresh_reset": True,
        "replay_consistent": False,
        "action": "candidate_probe",
        "safe": True,
        "repair_delta": "encoding_or_channel",
    },
    {
        "name": "effect_replay",
        "phase": "replay",
        "failure": "typed_effect_replay_consistent",
        "feedback": "typed_effect",
        "last_action": "candidate_probe",
        "negative_clean": True,
        "reference_agreement": True,
        "typed_available": True,
        "candidate_sent": True,
        "fresh_reset": True,
        "replay_consistent": True,
        "action": "replay_confirmed",
        "safe": True,
        "repair_delta": "none",
    },
    {
        "name": "oracle_gap",
        "phase": "guard",
        "failure": "typed_evaluator_missing",
        "feedback": "unresolved",
        "last_action": "candidate_probe",
        "negative_clean": True,
        "reference_agreement": True,
        "typed_available": False,
        "candidate_sent": True,
        "fresh_reset": False,
        "replay_consistent": False,
        "action": "abstain",
        "safe": False,
        "repair_delta": "none",
    },
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _normal(value: str, choices: tuple[str, ...], fallback: str) -> str:
    return value if value in choices else fallback


def _plan(row: dict[str, Any]) -> dict[str, str]:
    target = dict(row.get("target") or {})
    method = _normal(str(row.get("method", "GET")).upper(), METHODS, "GET")
    probe = _normal(str(target.get("probe_class", row.get("family", "other"))), PROBE_CLASSES, "other")
    channel = _normal(str(target.get("channel", "query" if method == "GET" else "form")), CHANNELS, "unknown")
    encoding = _normal(str(target.get("encoding", "plain")), ENCODINGS, "unknown")
    if channel == "query" and method == "POST":
        channel = "form"
    if channel == "form" and method == "GET":
        channel = "query"
    wire = {"query": "query_param", "form": "form_field", "header": "header_value", "path": "path_segment"}.get(channel, "none")
    return {"method": method, "probe_class": probe, "channel": channel, "encoding": encoding, "wire_kind": wire}


def _surface_tokens(index: int) -> tuple[str, ...]:
    _, tokens = SURFACE_VARIANTS[index % len(SURFACE_VARIANTS)]
    return tokens


def _context(row: dict[str, Any], spec: dict[str, Any], replica: int, surface_index: int, *, hard_negative: bool) -> list[str]:
    plan = _plan(row)
    field_count = 2 if plan["method"] == "POST" else 1
    if hard_negative:
        failure = "unsupported_surface_evaluator"
        feedback = "unresolved"
        state = dict(spec)
        state.update({"typed_available": False, "reference_agreement": False, "negative_clean": False, "fresh_reset": True, "candidate_sent": False, "replay_consistent": False, "action": "abstain"})
    else:
        failure = str(spec["failure"])
        feedback = str(spec["feedback"])
        state = spec
    history_bucket = (replica % 4) + 1
    tokens = [
        "[BOS]",
        "ir_layer=payload_grounding",
        "ir_family_agnostic=1",
        "ir_role=effect",
        "ir_surface=abstract_request",
        "ir_measure=wire_composition",
        *[str(item) for item in _surface_tokens(surface_index)],
        f"phase={state['phase']}",
        f"method={plan['method']}",
        f"channel_observed={plan['channel']}",
        f"field_bucket={'1' if field_count == 1 else '2'}",
        f"history_bucket={history_bucket}",
        f"fresh_reset={_flag(bool(state['fresh_reset']))}",
        "source_attested=1",
        f"negative_clean={_flag(bool(state['negative_clean']))}",
        f"reference_agreement={_flag(bool(state['reference_agreement']))}",
        f"typed_available={_flag(bool(state['typed_available']))}",
        f"candidate_sent={_flag(bool(state['candidate_sent']))}",
        f"replay_consistent={_flag(bool(state['replay_consistent']))}",
        f"last_action={state['last_action']}",
        f"failure_signature={failure}",
        f"feedback={feedback}",
        "family_hidden=1",
        "oracle_label_in_context=0",
        "literal_probe_in_context=0",
        f"step_replica={replica}",
        "[CTX_END]",
    ]
    return tokens


def _target(row: dict[str, Any], spec: dict[str, Any], *, hard_negative: bool) -> tuple[list[str], dict[str, Any]]:
    plan = _plan(row)
    if hard_negative:
        action = "abstain"
        safe = False
        repair_delta = "none"
    else:
        action = str(spec["action"])
        safe = bool(spec["safe"])
        repair_delta = str(spec["repair_delta"])
    # The sequence is deliberately a structured wire plan, not a literal
    # payload.  <RUNTIME_CANARY> is an execution-time placeholder only.
    tokens = [
        "[TARGET_BOS]",
        f"plan={action}",
        f"method={plan['method']}",
        f"probe_class={plan['probe_class'] if not hard_negative else 'other'}",
        f"channel={plan['channel'] if not hard_negative else 'unknown'}",
        f"encoding={plan['encoding'] if not hard_negative else 'unknown'}",
        f"wire={plan['wire_kind'] if not hard_negative and action != 'abstain' else 'none'}",
        "field_slot=observed_or_runtime_canary" if action != "abstain" else "field_slot=none",
        f"repair_delta={repair_delta}",
        "family_agnostic=1",
        f"final_action={action}",
        f"safe_to_send={_flag(safe)}",
        "[TARGET_EOS]",
    ]
    structured = {
        "next_action": action,
        "method": plan["method"],
        "probe_class": plan["probe_class"] if not hard_negative else "other",
        "channel": plan["channel"] if not hard_negative else "unknown",
        "encoding": plan["encoding"] if not hard_negative else "unknown",
        "wire_kind": plan["wire_kind"] if not hard_negative and action != "abstain" else "none",
        "repair_delta": repair_delta,
        "safe_to_send": safe,
        "oracle_required": True,
    }
    return tokens, structured


def _record(row: dict[str, Any], spec: dict[str, Any], *, split: str, replica: int, surface_index: int, hard_negative: bool) -> dict[str, Any]:
    source_id = str(row.get("record_id") or row.get("hard_negative_id") or "unknown")
    target_tokens, target = _target(row, spec, hard_negative=hard_negative)
    identity = {"source_id": source_id, "state": spec["name"], "replica": replica, "surface": surface_index, "split": split, "hard_negative": hard_negative}
    return {
        "record_id": "pg285:" + _sha(identity)[:24],
        "source_record_id": source_id,
        "source_group_id": str(row.get("group_id", source_id)),
        "source": str(row.get("source", "pg281")),
        "family": str(row.get("family", "other")),
        "method": target["method"],
        "split": split,
        "surface_variant": SURFACE_VARIANTS[surface_index % len(SURFACE_VARIANTS)][0],
        "state": str(spec["name"]),
        "replica": replica,
        "hard_negative": hard_negative,
        "context_tokens": _context(row, spec, replica, surface_index, hard_negative=hard_negative),
        "target_tokens": target_tokens,
        "target": target,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_eligible": not hard_negative,
        "memory_promotion_allowed": False,
        "source_evidence_hash": str(row.get("source_evidence_hash", "")),
    }


def _records(rows: list[dict[str, Any]], *, split: str, hard_negative: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for replica in range(3):
            for surface_index in range(len(SURFACE_VARIANTS)):
                for spec in STATE_SPECS:
                    result.append(_record(row, spec, split=split, replica=replica, surface_index=surface_index, hard_negative=hard_negative))
    return result


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_hard = json.loads(SOURCE_HARD.read_text(encoding="utf-8"))
    rows = list(source.get("records") or [])
    hard_rows = list(source_hard.get("records") or [])
    train_rows = [row for row in rows if row.get("split") == "train"]
    route_rows = [row for row in rows if row.get("split") == "route_dev"]
    family_rows = [row for row in rows if row.get("split") == "family_holdout"]
    train = _records(train_rows, split="train", hard_negative=False)
    route_dev = _records(route_rows, split="route_dev", hard_negative=False)
    family_holdout = _records(family_rows, split="family_holdout", hard_negative=False)
    hard = _records(hard_rows, split="hard_negative", hard_negative=True)
    records = [*train, *route_dev, *family_holdout]
    counts = {"train": len(train), "route_dev": len(route_dev), "family_holdout": len(family_holdout), "hard_negative": len(hard), "total": len(records) + len(hard)}
    dataset = {
        "schema_version": "pg285-payload-grounding-dataset-v1",
        "purpose": "failure-conditioned structured wire-plan next-token decoding",
        "source": {
            "pg281_dataset": "research/pg281_payload_policy_dataset_v1.json",
            "pg281_dataset_sha256": source.get("dataset_sha256", ""),
            "pg281_hard_negative": "research/pg281_payload_policy_hard_negative_v1.json",
            "pg281_hard_negative_sha256": source_hard.get("dataset_sha256", ""),
            "surface_variants": len(SURFACE_VARIANTS),
            "generated_states": len(STATE_SPECS),
            "replicas": 3,
        },
        "records": records,
        "hard_negative_records": hard,
        "counts": counts,
        "action_ontology": {"actions": list(ACTION_NAMES), "probe_classes": list(PROBE_CLASSES), "channels": list(CHANNELS), "encodings": list(ENCODINGS), "wire_kinds": list(WIRE_KINDS), "methods": list(METHODS)},
        "split_contract": {"train": "PG-281 source train with independent surface variants", "route_dev": "unseen routes", "family_holdout": "unseen families/implementations", "hard_negative": "evaluation-only; forced abstain"},
        "training_contract": {
            "family_hidden_in_context": True,
            "oracle_label_in_context": False,
            "literal_probe_values_out_of_context": True,
            "raw_response_bodies_out_of_context": True,
            "hard_negative_training_eligible": False,
            "remote_a800_required": True,
            "live_replay_required_for_promotion": True,
            "memory_promotion_allowed": False,
        },
        "scientific_contract": {"generated_surface_and_state_templates": True, "cross_template_generalization_claim_allowed": False, "real_application_gold_required": True},
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _sha({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    hard_dataset = {
        "schema_version": "pg285-payload-grounding-hard-negative-v1",
        "records": hard,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "dataset_sha256": _sha(hard),
    }
    all_rows = [*records, *hard]
    allowed_target_prefixes = ("[TARGET_BOS]", "[TARGET_EOS]")
    forbidden_context_tokens = ("family=", "oracle=", "payload=", "<script", "javascript:", "union", "drop", "raw_body")
    checks = {
        "counts_match": counts["total"] == len(all_rows),
        "train_nonempty": bool(train),
        "route_dev_nonempty": bool(route_dev),
        "family_holdout_nonempty": bool(family_holdout),
        "hard_negative_nonempty": bool(hard),
        "hard_negative_excluded": all(not row["training_eligible"] and not row["memory_promotion_allowed"] for row in hard),
        "context_has_no_oracle": all(not row["oracle_in_context"] and "oracle_label_in_context=0" in row["context_tokens"] for row in all_rows),
        "context_has_no_literal": all(not row["raw_payload_strings_stored"] and not row["raw_response_bodies_stored"] and not any(any(token.casefold() in item.casefold() for token in forbidden_context_tokens) for item in row["context_tokens"]) for row in all_rows),
        "family_hidden": all("family_hidden=1" in row["context_tokens"] for row in all_rows),
        "target_boundaries": all(row["target_tokens"][0] == allowed_target_prefixes[0] and row["target_tokens"][-1] == allowed_target_prefixes[1] for row in all_rows),
        "target_actions_allowlisted": all(row["target"]["next_action"] in ACTION_NAMES for row in all_rows),
        "source_group_split_disjoint": not (set(row["source_group_id"] for row in train) & set(row["source_group_id"] for row in family_holdout)),
        "safe_abstain_hard": all(row["target"]["next_action"] == "abstain" and row["target"]["safe_to_send"] is False for row in hard),
    }
    audit = {
        "schema_version": "pg285-payload-grounding-dataset-audit-v1",
        "dataset": str(DATASET.relative_to(ROOT).as_posix()),
        "dataset_sha256": dataset["dataset_sha256"],
        "counts": counts,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "blocked",
        "interpretation": "PG-285 仅证明结构化 wire plan 的失败条件化解码管线；硬负例拒答评估与真实靶场 evaluator gold 仍分开。",
    }
    audit["audit_sha256"] = _sha(audit)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HARD.write_text(json.dumps(hard_dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "counts": counts, "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
