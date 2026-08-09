from app.pg301_payload_assembly import TARGET_KEYS
from app.pg302_symbolic_assembly import SYMBOLIC_TARGET_KEYS, audit_symbolic_records, bind_symbolic_plan, symbolic_record, symbolic_target_for_context


def _context(typed="1", method="GET", field="query_param", encoding="url_percent"):
    return [
        f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}",
        f"typed_available={typed}", "feedback_state=observable_progress", "replay_ready=1", "evidence_present=1", "negative_control=1", "fresh_reset=1", "history_action=observe", "failure_class=none", "step_budget=present",
    ]


def test_model_target_uses_symbolic_slot_references():
    target = symbolic_target_for_context(_context())
    assert len(target) == len(SYMBOLIC_TARGET_KEYS) + 2
    assert "transport_ref=surface_method" in target
    assert "field_role_ref=surface_field_role" in target
    assert "encoding_ref=surface_encoding" in target
    assert "transport=GET" not in target


def test_binder_resolves_abstract_values_without_literal_wire():
    context = _context(method="POST", field="form_field", encoding="form_urlencoded")
    symbolic = symbolic_target_for_context(context)
    bound = bind_symbolic_plan(symbolic, context)
    assert bound is not None
    assert "transport=POST" in bound
    assert "field_role=form_field" in bound
    assert "encoding=form_urlencoded" in bound
    assert all("http" not in token.lower() for token in bound)


def test_missing_slot_symbolic_plan_cannot_be_safe():
    context = _context(typed="unknown")
    symbolic = symbolic_target_for_context(context)
    bound = bind_symbolic_plan(symbolic, context)
    assert bound is not None
    assert "question=ask_typed_availability" in bound
    assert "safe_to_send=0" in bound


def test_symbolic_dataset_audit_accepts_projected_records():
    rows = [
        symbolic_record({"record_id": "r1", "split": "train", "training_eligible": True, "context_tokens": _context()}),
        symbolic_record({"record_id": "r2", "split": "implementation_holdout", "training_eligible": False, "context_tokens": _context(typed="unknown")}),
        symbolic_record({"record_id": "r3", "split": "hard_negative_eval", "training_eligible": False, "hard_negative": True, "context_tokens": _context()}),
    ]
    audit = audit_symbolic_records(rows)
    assert audit["status"] == "passed"
