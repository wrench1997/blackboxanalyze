"""Build PG-301 abstract multi-step Rule-IR assembly records."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg301_payload_assembly import audit_assembly_records, project_assembly_record  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg301_payload_assembly_dataset_v1.json"
AUDIT = RESEARCH / "pg301_payload_assembly_audit_v1.json"


SURFACES = (
    ("GET", "query_param", "url_percent"),
    ("POST", "form_field", "form_urlencoded"),
    ("GET", "form_field", "url_percent"),
    ("POST", "query_param", "form_urlencoded"),
    ("GET", "form_field", "form_urlencoded"),
    ("POST", "query_param", "url_percent"),
    ("GET", "query_param", "form_urlencoded"),
    ("POST", "form_field", "url_percent"),
)
SURFACE_HOLDOUT = {SURFACES[6], SURFACES[7]}


def _raw(index: int, state: str, surface: tuple[str, str, str], split: str, *, hard_negative: bool = False, history: str = "observe", failure: str = "none", counterfactual_group: str = "none") -> dict:
    method, field, encoding = surface
    observations = {
        "missing_typed": {"typed_available": "unknown", "feedback_state": "unknown", "replay_ready": "unknown", "evidence_present": "unknown", "negative_control": "unknown", "fresh_reset": "unknown"},
        "missing_feedback": {"typed_available": "1", "feedback_state": "unknown", "replay_ready": "1", "evidence_present": "1", "negative_control": "1", "fresh_reset": "1"},
        "missing_replay": {"typed_available": "1", "feedback_state": "observable_progress", "replay_ready": "unknown", "evidence_present": "1", "negative_control": "1", "fresh_reset": "1"},
        "missing_evidence": {"typed_available": "1", "feedback_state": "observable_progress", "replay_ready": "1", "evidence_present": "unknown", "negative_control": "1", "fresh_reset": "1"},
        "ready": {"typed_available": "1", "feedback_state": "observable_progress", "replay_ready": "1", "evidence_present": "1", "negative_control": "1", "fresh_reset": "1"},
        "repair": {"typed_available": "1", "feedback_state": "observable_no_effect", "replay_ready": "1", "evidence_present": "1", "negative_control": "1", "fresh_reset": "1"},
    }[state]
    tokens = [
        "[BOS]",
        f"surface_method={method}",
        f"surface_field_role={field}",
        f"surface_encoding={encoding}",
        *[f"{key}={value}" for key, value in observations.items()],
        f"history_action={history}",
        f"failure_class={failure}",
        "step_budget=present",
        "[EOS]",
    ]
    return {"record_id": f"pg301:raw:{index}", "split": split, "training_eligible": split == "train", "hard_negative": hard_negative, "context_tokens": tokens, "trace_step": index % 4, "counterfactual_group": counterfactual_group}


def main() -> None:
    records: list[dict] = []
    index = 0
    states = ("missing_typed", "missing_feedback", "missing_replay", "missing_evidence", "ready", "repair")
    for state, surface in itertools.product(states, SURFACES):
        # Hold out complete surface combinations for every state, while also
        # holding a changed history presentation for the missing typed case.
        split = "implementation_holdout" if surface in SURFACE_HOLDOUT else "train"
        history = "candidate_failed" if state == "repair" else "observe"
        failure = "candidate_mismatch" if state == "repair" else "none"
        for copy_index in range(2):
            raw = _raw(index, state, surface, split, history=history, failure=failure, counterfactual_group=f"cf:{state}:{index}")
            records.append(project_assembly_record(raw))
            index += 1
    # Counterfactual hidden worlds share exactly the same visible context and
    # therefore must ask for the missing observation rather than guess a plan.
    for surface in SURFACES[:4]:
        for world in ("world_a", "world_b"):
            raw = _raw(index, "missing_typed", surface, "train", counterfactual_group=f"cf-hidden:{surface}:{world}")
            records.append(project_assembly_record(raw))
            index += 1
    # Hard negatives: all observations are available, but the previous
    # candidate failed.  A model may propose repair, never safe_to_send=1.
    for surface in SURFACES:
        for _ in range(2):
            raw = _raw(index, "repair", surface, "hard_negative_eval", hard_negative=True, history="candidate_failed", failure="candidate_mismatch", counterfactual_group="hard:repair")
            records.append(project_assembly_record(raw))
            index += 1
    audit = audit_assembly_records(records)
    dataset = {
        "schema_version": "pg301-payload-assembly-dataset-v1",
        "purpose": "causal next-token composition of question, abstract transport, field, encoding, oracle and stop slots",
        "source": {"kind": "synthetic_authorized_rule_ir", "literal_payload_strings_stored": False, "response_bodies_stored": False, "wire_emission": False},
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row.get("split") == "train" for row in records),
            "implementation_holdout": sum(row.get("split") == "implementation_holdout" for row in records),
            "hard_negative_eval": sum(row.get("split") == "hard_negative_eval" for row in records),
            "surface_holdout": [list(item) for item in sorted(SURFACE_HOLDOUT)],
            "counterfactual_rows": sum(1 for row in records if str(row.get("counterfactual_group", "")).startswith("cf-hidden")),
        },
        "contract": {
            "causal_next_token": True,
            "abstract_slots_only": True,
            "question_first": True,
            "transport_slot_bounded_to_get_post": True,
            "runtime_canary_is_placeholder": True,
            "typed_oracle_required": True,
            "fresh_reset_and_negative_control_required": True,
            "literal_payload_strings_stored": False,
            "wire_emission_allowed": False,
            "memory_promotion_allowed": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg301-payload-assembly-audit-v1", "schema_version": "pg301-payload-assembly-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit}
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
