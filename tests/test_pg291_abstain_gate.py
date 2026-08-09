from app.pg291_abstain_gate import build_vocab, encode, predict, train_gate


def _row(safe: bool, marker: str):
    return {"context_tokens": ["method=GET", marker], "target": {"safe_to_send": safe}}


def test_gate_vocab_and_labels_are_context_only():
    rows = [_row(False, "feedback=unresolved"), _row(True, "feedback=typed_effect")]
    vocab = build_vocab(rows)
    values, lengths, labels = encode(rows, vocab)
    assert values.shape[0] == 2
    assert lengths.tolist() == [2, 2]
    assert labels.tolist() == [0.0, 1.0]


def test_gate_can_learn_a_tiny_separable_context():
    rows = [_row(False, "feedback=unresolved")] * 4 + [_row(True, "feedback=typed_effect")] * 4
    vocab = build_vocab(rows)
    model = train_gate(rows, vocab, __import__("torch").device("cpu"), 291, epochs=30, embed_dim=8, hidden_dim=12)
    metrics = predict(model, rows, vocab, __import__("torch").device("cpu"))
    assert metrics["positive_recall"] >= 0.5
    assert metrics["safe_reject_rate"] >= 0.5

