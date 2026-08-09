"""Build PG-336 from the real PG-325 failure/replay trace.

PG-325 already contains a bounded, authorized local Docker replay with real
GET/POST candidate/reference/negative roles and a recorded failure followed by
an action-changing repair.  This builder turns only the abstract trace into a
process-token diagnostic dataset.  Route names, family/implementation names,
wire values, payloads, response bodies and evaluator literals remain outside
the model context.  The result is intentionally diagnostic-only: it is a
better process signal, not a gold vulnerability dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research" / "pg325_sql_family_holdout_trace_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg336_real_failure_process_token_v1.json"

AXES = (
    "document_structure",
    "navigation",
    "request_transport",
    "response_transport",
    "javascript_surface",
    "failure_feedback",
    "belief_replay",
)

TRACE_SCHEMA = "pg325-sql-family-holdout-trace-v1"
SCHEMA_VERSION = "pg336-real-failure-process-token-v1"

CONTEXT_KEYS = {
    "typed_available",
    "feedback_state",
    "replay_ready",
    "evidence_present",
    "negative_control",
    "fresh_reset",
    "surface_method",
    "surface_field_role",
    "surface_encoding",
    "history_action",
    "failure_class",
    "step_budget",
    "process_stage",
    "missing_slot",
    "missing_count",
    "observation_state",
    "review_required",
}

ALLOWED_VALUES = {
    "typed_available": {"0", "1"},
    "feedback_state": {"negative_control_clear", "observable_progress", "unknown"},
    "replay_ready": {"0", "1"},
    "evidence_present": {"0", "1"},
    "negative_control": {"0", "1"},
    "fresh_reset": {"0", "1"},
    "surface_method": {"GET", "POST"},
    "surface_field_role": {"query_param", "form_field", "unknown"},
    "surface_encoding": {"url_percent", "form_urlencoded", "unknown"},
    "history_action": {"candidate_request", "reference_request", "negative_control", "candidate_failed", "preflight"},
    "failure_class": {"none", "effect_not_confirmed", "missing_observation", "negative_control_clear", "unknown"},
    "step_budget": {"present", "unknown"},
    "process_stage": {"probe_observed", "failure_feedback", "negative_review", "ask_preflight"},
    "missing_slot": {"typed_available", "feedback_state", "replay_ready", "evidence_present", "negative_control", "fresh_reset", "surface_method", "surface_field_role", "surface_encoding", "step_budget"},
    "missing_count": {"one", "two", "few", "many"},
    "observation_state": {"observed", "not_observed"},
    "review_required": {"0", "1"},
}

TARGET_KEYS = {
    "question",
    "next_action",
    "repair_action",
    "transport_ref",
    "field_role_ref",
    "encoding_ref",
    "stop_condition",
    "safe_to_send",
    "probe_variant_ref",
    "encoding_chain_ref",
    "failure_class",
    "action_changed",
}

TARGET_VALUES = {
    "question": {"none", "ask_typed_availability", "review_negative"},
    "next_action": {"assemble_abstract_plan", "repair_abstract_plan", "ask_typed", "abstain"},
    "repair_action": {"none", "retry_bounded_variant", "observe"},
    "transport_ref": {"surface_method"},
    "field_role_ref": {"surface_field_role"},
    "encoding_ref": {"surface_encoding"},
    "stop_condition": {"typed_effect_or_abstain", "repair_feedback_or_abstain", "ask_typed_or_abstain"},
    "safe_to_send": {"0", "1"},
    "probe_variant_ref": {"source_attested_candidate", "reference_canary", "negative_control", "none"},
    "encoding_chain_ref": {"surface_encoding", "none"},
    "failure_class": {"none", "effect_not_confirmed", "missing_observation", "negative_control_clear"},
    "action_changed": {"0", "1"},
}

FORBIDDEN_PREFIXES = (
    "family=", "implementation=", "route=", "route_literal=", "source=", "image=",
    "record=", "path=", "url=", "payload=", "payload_", "raw_", "response_body=",
    "response_body_text=", "oracle=", "evaluator=", "canary=",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kv(tokens: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in tokens:
        text = str(raw)
        if "=" in text:
            key, value = text.split("=", 1)
            result[key] = value
    return result


def _normalize_context_token(token: Any) -> str | None:
    text = str(token)
    if text in {"[BOS]", "[EOS]"}:
        return text
    folded = text.casefold()
    if any(folded.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return None
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    if key not in CONTEXT_KEYS:
        return None
    allowed = ALLOWED_VALUES.get(key, set())
    if value not in allowed:
        return f"{key}=unknown" if "unknown" in allowed else None
    return f"{key}={value}"


def abstract_context(tokens: list[Any], *, stage: str | None = None, extra: list[str] | None = None) -> list[str]:
    """Keep only the stable process ontology, never route or evaluator text."""
    result: list[str] = []
    for token in tokens:
        normalized = _normalize_context_token(token)
        if normalized is not None:
            result.append(normalized)
    if stage:
        result.append(f"process_stage={stage}")
    result.extend(str(item) for item in (extra or []) if _normalize_context_token(item) is not None)
    # Keep a deterministic BOS/EOS envelope and preserve first occurrence order.
    result = [item for item in result if item not in {"[BOS]", "[EOS]"}]
    return ["[BOS]", *list(dict.fromkeys(result)), "[EOS]"]


def _target_from_source(tokens: list[Any], *, failure: bool = False) -> list[str]:
    parsed = _kv(tokens)
    selected: list[str] = ["[TARGET_BOS]"]
    for key in ("question", "next_action", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "stop_condition", "safe_to_send", "probe_variant_ref", "encoding_chain_ref"):
        value = parsed.get(key)
        if key not in TARGET_KEYS or value not in TARGET_VALUES.get(key, set()):
            continue
        selected.append(f"{key}={value}")
    if failure:
        # The trace is the authority for the action-changing repair; do not
        # copy its oracle/canary/evaluator values.
        selected = [
            "[TARGET_BOS]",
            "question=none",
            "next_action=repair_abstract_plan",
            "repair_action=retry_bounded_variant",
            "transport_ref=surface_method",
            "field_role_ref=surface_field_role",
            "encoding_ref=surface_encoding",
            "stop_condition=repair_feedback_or_abstain",
            "safe_to_send=0",
            "probe_variant_ref=none",
            "encoding_chain_ref=none",
            "failure_class=effect_not_confirmed",
            "action_changed=1",
        ]
    if not any(item.startswith("safe_to_send=") for item in selected):
        selected.append("safe_to_send=0")
    selected.append("[TARGET_EOS]")
    return list(dict.fromkeys(selected))


def _target_ask() -> list[str]:
    return [
        "[TARGET_BOS]", "question=ask_typed_availability", "next_action=ask_typed",
        "repair_action=observe", "stop_condition=ask_typed_or_abstain",
        "failure_class=missing_observation", "action_changed=1", "safe_to_send=0", "[TARGET_EOS]",
    ]


def _target_negative_review() -> list[str]:
    return [
        "[TARGET_BOS]", "question=review_negative", "next_action=abstain", "repair_action=none",
        "stop_condition=typed_effect_or_abstain", "failure_class=negative_control_clear",
        "action_changed=1", "safe_to_send=0", "[TARGET_EOS]",
    ]


def _manifest(kind: str, context: list[str], missing_slots: list[str] | None = None) -> dict[str, dict[str, str]]:
    parsed = _kv(context)
    missing = set(missing_slots or [])
    request_status = "observed" if all(parsed.get(key) not in {None, "unknown"} for key in ("surface_method", "surface_field_role", "surface_encoding")) else "not_observed"
    failure_status = "observed" if kind == "failure_repair" else ("not_observed" if "feedback_state" in missing else "absent")
    belief_status = "observed" if kind in {"probe_observed", "failure_repair", "negative_review"} and parsed.get("replay_ready") == "1" else "not_observed"
    return {
        "document_structure": {"presence": "not_observed"},
        "navigation": {"presence": "not_observed"},
        "request_transport": {"presence": request_status},
        "response_transport": {"presence": "not_observed"},
        "javascript_surface": {"presence": "not_observed"},
        "failure_feedback": {"presence": failure_status},
        "belief_replay": {"presence": belief_status},
    }


def _base(source_kind: str, source_id: Any, source_index: int, seed: Any, split: str, trace_sha: str, record_hash: str) -> dict[str, Any]:
    identity = digest({"source_kind": source_kind, "source_index": source_index, "source_record_hash": record_hash, "seed": seed})
    return {
        "schema_version": "pg336-real-process-token-row-v1",
        "record_id": f"pg336:{identity}",
        "split": split,
        "diagnostic_kind": source_kind,
        "source_grounded": True,
        "synthetic_counterfactual": False,
        "source_trace_sha256": trace_sha,
        "source_record_sha256": record_hash,
        "source_index": source_index,
        "source_seed": int(seed) if str(seed).isdigit() else None,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "payload_catalog_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }


def _row(base: dict[str, Any], context: list[str], target: list[str], *, kind: str, missing_slots: list[str] | None = None, real_failure: bool = False, real_negative: bool = False, source_evidence: str | None = None) -> dict[str, Any]:
    return {
        **base,
        "context_tokens": context,
        "target_tokens": target,
        "field_capture_manifest": _manifest(kind, context, missing_slots),
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
        "process_metadata": {
            "real_failure_trace": real_failure,
            "real_negative_evaluator_trace": real_negative,
            "real_ask_preflight": kind == "ask_preflight",
            "evidence_hash_present_sidecar": bool(source_evidence and len(str(source_evidence)) == 64),
        },
        "evaluator_sidecar_ref": {"evidence_sha256": source_evidence if source_evidence and len(str(source_evidence)) == 64 else None, "off_context": True},
    }


def build_dataset(source: Mapping[str, Any], *, source_trace_sha256: str = "") -> dict[str, Any]:
    if source.get("schema_version") != TRACE_SCHEMA:
        raise ValueError("unexpected PG-325 trace schema")
    episodes = [item for item in list(source.get("episodes") or []) if isinstance(item, Mapping)]
    records: list[dict[str, Any]] = []
    real_failures = 0
    real_negatives = 0
    for index, episode in enumerate(episodes):
        raw_id = str(episode.get("record_id", ""))
        role = raw_id.rsplit(":", 1)[-1]
        seed = episode.get("seed")
        split = "train" if str(seed) == "31901" else "seed_holdout"
        record_hash = digest({"record_id": raw_id, "context_tokens": episode.get("context_tokens"), "target_tokens": episode.get("target_tokens"), "evidence_sha256": episode.get("evidence_sha256")})
        context = abstract_context(list(episode.get("context_tokens") or []), stage="failure_feedback" if role == "failure-repair" else "probe_observed")
        if role == "failure-repair":
            transition = episode.get("failure_transition") or {}
            if transition.get("action_changed") is not True:
                raise ValueError("PG-325 failure trace lacks action change")
            real_failures += 1
            records.append(_row(_base("failure_repair", raw_id, index, seed, split, source_trace_sha256, record_hash), context, _target_from_source(list(episode.get("target_tokens") or []), failure=True), kind="failure_repair", real_failure=True, source_evidence=str(episode.get("evidence_sha256") or "")))
        elif role in {"candidate_request", "reference_request", "negative_control"}:
            records.append(_row(_base("probe_observed", raw_id, index, seed, split, source_trace_sha256, record_hash), context, _target_from_source(list(episode.get("target_tokens") or [])), kind="probe_observed", real_negative=role == "negative_control", source_evidence=str(episode.get("evidence_sha256") or "")))
            if role == "negative_control":
                real_negatives += 1
                review_context = abstract_context(list(episode.get("context_tokens") or []), stage="negative_review", extra=["review_required=1"])
                records.append(_row(_base("negative_review", raw_id, index, seed, split, source_trace_sha256, record_hash), review_context, _target_negative_review(), kind="negative_review", real_negative=True, source_evidence=str(episode.get("evidence_sha256") or "")))

    preflight = [item for item in list(source.get("multi_missing_preflight") or []) if isinstance(item, Mapping)]
    for index, item in enumerate(preflight):
        missing_slots = [str(slot) for slot in list(item.get("missing_slots") or []) if str(slot) in ALLOWED_VALUES["missing_slot"]]
        if not missing_slots or item.get("question_correct") is not True or item.get("safe_actual") is not False:
            raise ValueError("PG-325 ASK preflight is not a verified safe row")
        method = str(item.get("method")) if str(item.get("method")) in {"GET", "POST"} else "unknown"
        missing_tokens = [f"missing_slot={slot}" for slot in missing_slots]
        missing_count = "one" if len(missing_slots) == 1 else "two" if len(missing_slots) == 2 else "few" if len(missing_slots) <= 3 else "many"
        context = abstract_context(["[BOS]", f"surface_method={method}", "history_action=preflight", "step_budget=present"], stage="ask_preflight", extra=[*missing_tokens, f"missing_count={missing_count}", "observation_state=not_observed"])
        source_hash = digest({"method": method, "missing_slots": sorted(missing_slots), "question": item.get("question_actual"), "safe": item.get("safe_actual"), "index": index})
        records.append(_row(_base("ask_preflight", source_hash, index, "31901", "train" if index % 3 == 0 else "seed_holdout", source_trace_sha256, source_hash), context, _target_ask(), kind="ask_preflight", missing_slots=missing_slots))

    context_tokens = sorted({str(token) for item in records for token in item["context_tokens"]})
    target_tokens = sorted({str(token) for item in records for token in item["target_tokens"]})
    counts = Counter(str(item["diagnostic_kind"]) for item in records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "real PG-325 failure/ASK/negative process tokens with evaluator sidecars off-context",
        "source": {
            "trace_schema": TRACE_SCHEMA,
            "trace_path": "research/pg325_sql_family_holdout_trace_v1.json",
            "trace_sha256": source_trace_sha256,
            "episode_count": len(episodes),
            "real_failure_trace_count": real_failures,
            "real_negative_trace_count": real_negatives,
            "ask_preflight_count": len(preflight),
            "real_gold_rows": 0,
            "synthetic_counterfactual_rows": 0,
            "implementation_count": 1,
            "independent_implementation_holdout": False,
        },
        "records": records,
        "counts": {
            "total": len(records),
            "probe_observed": counts.get("probe_observed", 0),
            "failure_repair": counts.get("failure_repair", 0),
            "negative_review": counts.get("negative_review", 0),
            "ask_preflight": counts.get("ask_preflight", 0),
            "train": sum(item["split"] == "train" for item in records),
            "seed_holdout": sum(item["split"] == "seed_holdout" for item in records),
            "get": sum("surface_method=GET" in item["context_tokens"] for item in records),
            "post": sum("surface_method=POST" in item["context_tokens"] for item in records),
        },
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "process_policy": {
            "real_failure_trace_used": True,
            "failure_requires_action_change": True,
            "ask_rows_are_verified_preflight": True,
            "negative_review_is_evaluator_derived": True,
            "seed_holdout_is_not_independent_implementation": True,
            "raw_wire_and_evaluator_answers_off_context": True,
        },
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    payload["dataset_sha256"] = digest(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-336 real failure process-token diagnostics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = build_dataset(source, source_trace_sha256=file_digest(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "diagnostic_only", "records": len(result["records"]), "counts": result["counts"], "dataset_sha256": result["dataset_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
