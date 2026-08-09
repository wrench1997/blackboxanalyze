import torch

from app.pg292_feature_gate import build_feature_vocab, encode, predict, train_gate


def _row(safe: bool, typed: str, feedback: str):
    return {"context_tokens": [f"typed_available={typed}", f"feedback={feedback}", "candidate_sent=0", "replay_consistent=0"], "target": {"safe_to_send": safe}}


def test_feature_gate_omits_unknown_high_cardinality_value_but_keeps_shared_keys():
    rows = [_row(False, "0", "unresolved"), _row(True, "1", "typed_effect")]
    vocab = build_feature_vocab(rows)
    values, labels = encode(rows, vocab)
    assert "key:typed_available" in vocab
    assert values.shape[0] == 2
    assert labels.tolist() == [0.0, 1.0]


def test_feature_gate_learns_shared_state_features():
    train_rows = [_row(False, "0", "unresolved")] * 8 + [_row(True, "1", "typed_effect")] * 8
    vocab = build_feature_vocab(train_rows)
    model = train_gate(train_rows, vocab, torch.device("cpu"), 292, epochs=40, hidden_dim=16)
    metrics = predict(model, train_rows, vocab, torch.device("cpu"))
    assert metrics["positive_recall"] >= 0.5
    assert metrics["safe_reject_rate"] >= 0.5

