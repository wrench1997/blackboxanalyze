from scripts.audit_pg349_decision_boundary import audit_document


def _row(*, split="train", next_action="select_probe_variant", safe=True, role="source_attested_candidate", context=None):
    role_token = {
        "source_attested_candidate": "candidate",
        "reference": "reference",
        "negative_control": "negative",
        "runtime_canary": "replay",
    }.get(role, "candidate")
    return {
        "split": split,
        "context_tokens": context or ["request_transport_presence=observed", f"belief_probe_role={role_token}"],
        "target_tokens": ["[TARGET_BOS]", f"next_action={next_action}", "[TARGET_EOS]"],
        "target_projection": {
            "question": "none",
            "next_action": next_action,
            "repair_action": "none",
            "transport_ref": "get_query",
            "field_role_ref": "query_term",
            "encoding_ref": "identity",
            "probe_variant_ref": role,
            "payload_shape_ref": "query_marker",
            "safe_to_send": safe,
        },
        "source_meta": {"surface_id": "surface-a"},
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
    }


def test_decision_boundary_passes_without_context_conflicts() -> None:
    rows = [
        _row(role="source_attested_candidate"),
        _row(role="reference"),
        _row(role="negative_control", next_action="abstain", safe=False),
        _row(role="runtime_canary", next_action="replay"),
    ]
    report = audit_document({"records": rows})
    assert report["status"] == "passed_decision_boundary_diagnostic"
    assert report["context_target_conflict_count"] == 0
    assert report["decision_conflict_count"] == 0
    assert report["complete_candidate_reference_negative_replay_groups"] == 1
    assert report["promotion"]["training_allowed"] is False


def test_conflicting_target_for_same_context_is_blocked() -> None:
    same_context = ["request_transport_presence=observed", "belief_probe_role=candidate"]
    rows = [_row(context=same_context), _row(next_action="abstain", safe=False, role="negative_control", context=same_context)]
    report = audit_document({"records": rows})
    assert report["status"] == "blocked_context_target_ambiguity"
    assert report["context_target_conflict_count"] == 1
    assert report["decision_conflict_count"] == 1


def test_firewall_failure_is_blocked_without_printing_raw_fields() -> None:
    row = _row()
    row["context_tokens"] = ["request_transport_presence=observed", "raw_payload=forbidden"]
    row["context_firewall"] = {"forbidden_token_count": 1, "sidecars_off_context": False}
    report = audit_document({"records": [row]})
    assert report["status"] == "blocked_context_target_ambiguity"
    assert "forbidden_token" in report["failures"]
    assert "raw_payload" not in str(report)
