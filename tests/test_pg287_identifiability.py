from __future__ import annotations

import json
from pathlib import Path

from app.pg285_payload_grounding import build_vocabs, encode_rows, greedy_decode, PayloadGroundingDecoder


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg287_dataset_turns_encoding_collision_into_explicit_ask():
    data = _load("pg287_identifiability_dataset_v1.json")
    audit = _load("pg287_identifiability_dataset_audit_v1.json")
    assert audit["status"] == "passed"
    assert audit["coverage_gate_status"] == "blocked"
    assert audit["coverage"]["family_holdout_resolved_coverage"] is False
    assert audit["coverage_gaps"] == ["family_holdout_resolved_rows"]
    assert data["training_contract"]["ambiguous_rows_train_ask_typed"] is True
    assert data["training_contract"]["family_hidden_in_context"] is True
    assert data["counts"]["ambiguous"] > data["counts"]["resolved"]
    ambiguous = [row for row in data["records"] if row["variant"] == "ambiguous"]
    resolved = [row for row in data["records"] if row["variant"] == "resolved"]
    assert all(row["target"]["next_action"] == "ask_typed" and row["target"]["safe_to_send"] is False for row in ambiguous[:100])
    assert all("encoding_observed=unknown" in row["context_tokens"] for row in ambiguous[:100])
    assert all(row["target"]["encoding"] != "unknown" and any(token == f"encoding_observed={row['target']['encoding']}" for token in row["context_tokens"]) for row in resolved[:100])
    assert all("family=" not in " ".join(row["context_tokens"]) and "oracle=" not in " ".join(row["context_tokens"]) for row in data["records"][:200])
    assert all(row["training_eligible"] is False for row in data["hard_negative_records"])


def test_pg287_target_vocab_and_cpu_decode_are_bounded():
    data = _load("pg287_identifiability_dataset_v1.json")
    train = [row for row in data["records"] if row["split"] == "train"][:64]
    context_vocab, target_vocab = build_vocabs(train)
    model = PayloadGroundingDecoder(len(context_vocab), len(target_vocab), embed_dim=16, hidden_dim=24)
    contexts, lengths, _, _ = encode_rows(train[:2], context_vocab, target_vocab)
    tokens = greedy_decode(model, contexts, lengths, target_vocab, max_tokens=20)
    assert len(tokens) == 2
    assert all(len(item) <= 20 for item in tokens)
    assert "<script" not in " ".join(" ".join(item) for item in tokens).casefold()
