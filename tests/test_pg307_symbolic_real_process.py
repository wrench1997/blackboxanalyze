from __future__ import annotations

import json
from pathlib import Path

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS
from app.pg301_payload_assembly import target_map
from app.pg302_symbolic_assembly import bind_symbolic_plan


ROOT = Path(__file__).resolve().parents[1]


def _row_tokens() -> tuple[list[str], list[str]]:
    context = [
        "[BOS]",
        "typed_available=1",
        "feedback_state=negative_control_clear",
        "replay_ready=1",
        "evidence_present=1",
        "negative_control=1",
        "fresh_reset=1",
        "surface_method=POST",
        "surface_field_role=form_field",
        "surface_encoding=form_urlencoded",
        "history_action=none",
        "failure_class=none",
        "step_budget=present",
        "[EOS]",
    ]
    symbolic = [
        TARGET_BOS,
        "question=none",
        "next_action=assemble_abstract_plan",
        "repair_action=none",
        "transport_ref=surface_method",
        "field_role_ref=surface_field_role",
        "encoding_ref=surface_encoding",
        "canary=runtime",
        "oracle=typed",
        "stop_condition=typed_effect_or_abstain",
        "safe_to_send=1",
        TARGET_EOS,
    ]
    return context, symbolic


def test_pg307_dataset_audit_is_passed_and_information_complete() -> None:
    dataset = json.loads(
        (ROOT / "research/pg307_symbolic_real_process_dataset_v1.json").read_text(encoding="utf-8-sig")
    )
    audit = json.loads(
        (ROOT / "research/pg307_symbolic_real_process_dataset_audit_v1.json").read_text(encoding="utf-8-sig")
    )
    assert audit["status"] == "passed"
    assert dataset["counts"]["real_process_rows"] > 0
    assert dataset["counts"]["missing_counterfactual_rows"] > 0
    assert dataset["contract"]["oracle_target_off_input"] is True
    assert dataset["contract"]["payload_strings_excluded"] is True


def test_symbolic_binder_copies_only_visible_surface_slots() -> None:
    context, symbolic = _row_tokens()
    bound = bind_symbolic_plan(symbolic, context)
    assert bound is not None
    values = target_map(bound)
    assert values["transport"] == "POST"
    assert values["field_role"] == "form_field"
    assert values["encoding"] == "form_urlencoded"
    assert values["safe_to_send"] == "1"


def test_malformed_or_unknown_reference_fails_closed() -> None:
    context, symbolic = _row_tokens()
    malformed = list(symbolic)
    malformed[4] = "transport_ref=route_oracle"
    assert bind_symbolic_plan(malformed, context) is None

    unknown_surface = [token.replace("surface_method=POST", "surface_method=unknown") for token in context]
    bound = bind_symbolic_plan(symbolic, unknown_surface)
    assert bound is not None
    assert target_map(bound)["safe_to_send"] == "0"
