from app.goal_label_decoder import NeuralGoalLabelDecoder


def _rows():
    return [
        {"row_id": "a", "context_group": "c1", "delta_tokens": [], "safe_probe": True},
        {"row_id": "b", "context_group": "c2", "delta_tokens": [], "safe_probe": True},
        {"row_id": "c", "context_group": "c3", "delta_tokens": ["DELTA_SURFACE_TRUE_BOOLEAN_COUNT_INCREASE"], "safe_probe": True},
        {"row_id": "d", "context_group": "c4", "delta_tokens": ["DELTA_SURFACE_NUMERIC_FIELD_COUNT_INCREASE", "DELTA_GEOMETRY_LEAF_COUNT_INCREASE"], "safe_probe": True},
    ]


def test_neural_decoder_learns_generic_effect_cluster_without_oracle_fields():
    decoder = NeuralGoalLabelDecoder(seed=17, epochs=8, device="cpu").fit(_rows())
    proposal = decoder.proposal(design_row_count=4)
    assert proposal["proposal_inputs"]["oracle_visible"] is False
    assert proposal["proposal_inputs"]["family_visible"] is False
    assert proposal["audit"]["training_promotion_allowed"] is False
    assert decoder.predict(_rows()[0])["decision"] == "reject"
    assert decoder.predict(_rows()[3])["decision"] == "confirm_candidate"


def test_neural_decoder_fails_closed_on_leakage_and_unseen_tokens():
    decoder = NeuralGoalLabelDecoder(seed=17, epochs=4, device="cpu").fit(_rows())
    try:
        decoder.predict({"delta_tokens": [], "oracle_projection": {}})
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("oracle leakage must be rejected")
    output = decoder.predict({"delta_tokens": ["DELTA_NEW_PREFIX_VALUE"]})
    assert output["decision"] == "abstain"


def test_neural_decoder_tokenless_signal_is_degenerate_and_abstains():
    tokenless = [{"row_id": "a", "context_group": "c1", "delta_tokens": [], "safe_probe": True} for _ in range(4)]
    decoder = NeuralGoalLabelDecoder(seed=19, epochs=4, device="cpu").fit(tokenless)
    assert decoder.proposal(design_row_count=4)["degenerate_visible_signal"] is True
    assert decoder.predict(tokenless[0])["decision"] == "abstain"
