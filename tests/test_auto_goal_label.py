from app.auto_goal_label import apply_proposal, make_visible_pair, propose_goal_and_labels


def _step(*, positive: bool, changed: bool = False):
    projection = {
        "status_class": "2xx",
        "content_type_class": "json",
        "body_length_bucket": "256-4095",
        "effect_surface": {
            "array_field_count": 0,
            "boolean_field_count": 1,
            "nonzero_numeric_count": 1 if changed else 0,
            "numeric_field_count": 1,
            "true_boolean_count": 1 if changed else 0,
        },
        "effect_geometry": {
            "array_count": 0,
            "array_item_count": 0,
            "boolean_count": 10 if changed else 8,
            "leaf_count": 16 if changed else 13,
            "max_depth": 2 if changed else 1,
            "nonzero_numeric_count": 1 if changed else 0,
            "numeric_count": 1,
            "object_count": 2 if changed else 1,
            "string_count": 5 if changed else 4,
            "string_length_bucket_sum": 6 if changed else 5,
            "true_boolean_count": 4 if changed else 0,
        },
        "location_origin_changed": False,
        "state_changed": False,
        "transport_error": False,
    }
    return {
        "action_manifest": {
            "method": "GET",
            "encoding_chain": ["identity"],
            "probe_ref": "safe-probe-confirm",
            "safety": {
                "no_external_network": True,
                "does_not_execute": True,
                "no_database_write": True,
                "no_credential_access": True,
            },
        },
        "response_projection": projection,
        "oracle_projection": {"positive": positive, "positive_authority": positive},
    }


def test_visible_pair_uses_only_bounded_observation_deltas():
    control = _step(positive=False, changed=False)
    candidate = _step(positive=True, changed=True)
    visible = make_visible_pair(control, candidate)
    assert visible["has_observed_change"] is True
    assert any(token.startswith("DELTA_SURFACE_") for token in visible["delta_tokens"])
    assert "oracle_projection" not in visible
    assert "probe_ref" not in visible


def test_proposal_is_rejected_if_evaluator_fields_are_visible():
    rows = [{"row_id": "a", "context_group": "c", "delta_tokens": [], "oracle_projection": {}}]
    try:
        propose_goal_and_labels(rows)
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("oracle leakage must fail closed")


def test_proposal_decodes_known_change_and_abstains_on_unseen_prefix():
    rows = [
        {"row_id": "a", "context_group": "c1", "delta_tokens": [], "safe_probe": True},
        {"row_id": "b", "context_group": "c2", "delta_tokens": ["DELTA_SURFACE_TRUE_BOOLEAN_COUNT_INCREASE"], "safe_probe": True},
    ]
    proposal = propose_goal_and_labels(rows)
    assert proposal["proposal_inputs"]["oracle_visible"] is False
    known = apply_proposal(rows[1], proposal)
    assert known["decision"] == "confirm_candidate"
    unseen = apply_proposal({"delta_tokens": ["DELTA_NEW_FAMILY_X_CHANGE"]}, proposal)
    assert unseen["decision"] == "abstain"
