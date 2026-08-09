from app.active_probe import active_probe_score, choose_active_probe, normalized_entropy


def test_entropy_prefers_ambiguous_distribution():
    assert normalized_entropy({"a": 0.5, "b": 0.5}) > normalized_entropy({"a": 0.99, "b": 0.01})


def test_active_probe_is_deterministic_and_label_free():
    rows = [
        {"path": "/certain", "model_score": 1.0, "rule_ir_decoder": {"confidence": 0.99, "probabilities": {"a": 0.99, "b": 0.01}}},
        {"path": "/ambiguous", "model_score": 0.4, "rule_ir_decoder": {"confidence": 0.51, "probabilities": {"a": 0.5, "b": 0.5}}},
    ]
    assert choose_active_probe(rows)["path"] == "/ambiguous"
    assert active_probe_score(rows[1]) > active_probe_score(rows[0])
