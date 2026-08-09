from __future__ import annotations

import hashlib
import json

import pytest
import torch

from app.open_source_token_embedding import (
    IR_SLOT_IDS,
    MAX_IR_TOKENS,
    PAD_ID,
    RuleIRTokenEmbedding,
    VOCABULARY,
    canonical_ir_token_strings,
    open_source_ir_token_inputs,
)


def _layers() -> list[dict[str, object]]:
    return [
        {
            "tokens": [
                {"slot_id": slot, "value": "html" if slot == "surface.modalities" else "unknown", "weight": 1.0}
                for slot in IR_SLOT_IDS
            ]
        }
    ]


def test_open_source_tokenizer_is_bounded_and_deterministic() -> None:
    embedding_a = RuleIRTokenEmbedding(seed=13201)
    embedding_b = RuleIRTokenEmbedding(seed=13201)
    ids_a, scalars = open_source_ir_token_inputs(embedding_a, _layers())
    ids_b, _ = open_source_ir_token_inputs(embedding_b, _layers())
    assert ids_a == ids_b
    assert len(ids_a) == MAX_IR_TOKENS
    assert len(scalars) == MAX_IR_TOKENS
    assert ids_a[-1] == PAD_ID
    assert embedding_a.provenance.pretrained is False
    assert embedding_a.provenance.tokenizer_vocab_size == len(VOCABULARY)
    # Only canonical pair tokens may be decoded; arbitrary source literals are
    # never inserted into the vocabulary.
    assert all("fetch" not in token for token in VOCABULARY)
    assert canonical_ir_token_strings(_layers())[0].startswith("ir.surface.modalities=")


def test_unknown_ir_values_collapse_to_unknown_without_raw_retention() -> None:
    embedding = RuleIRTokenEmbedding()
    layers = [{"tokens": [{"slot_id": "surface.modalities", "value": "raw-page-secret", "weight": 2.0}]}]
    token_strings = canonical_ir_token_strings(layers)
    assert "raw-page-secret" not in " ".join(token_strings)
    assert token_strings[0] == "ir.surface.modalities=unknown"
    ids, _ = open_source_ir_token_inputs(embedding, layers)
    assert ids[0] != PAD_ID


def test_attested_local_checkpoint_requires_source_license_and_hash(tmp_path) -> None:
    seed_embedding = RuleIRTokenEmbedding(embedding_dim=8)
    path = tmp_path / "embedding.pt"
    torch.save({"weight": seed_embedding.embedding.weight.detach().cpu()}, path)
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        RuleIRTokenEmbedding(embedding_dim=8, pretrained_weights_path=path, expected_sha256=raw_sha)
    loaded = RuleIRTokenEmbedding(
        embedding_dim=8,
        pretrained_weights_path=path,
        expected_sha256=raw_sha,
        source_id="local-test-open-source-model",
        license="Apache-2.0",
    )
    assert loaded.provenance.pretrained is True
    assert loaded.provenance.weights_source == "attested_local_pretrained_checkpoint"
    assert loaded.provenance.source_id == "local-test-open-source-model"
    assert loaded.provenance.license == "Apache-2.0"
    assert torch.allclose(loaded.embedding.weight, seed_embedding.embedding.weight)


def test_scalar_modes_preserve_shape_and_failure_ablation() -> None:
    embedding = RuleIRTokenEmbedding()
    layers = _layers()
    _, weighted = open_source_ir_token_inputs(embedding, layers, mode="weighted")
    _, uniform = open_source_ir_token_inputs(embedding, layers, mode="uniform")
    _, zero = open_source_ir_token_inputs(embedding, layers, mode="zero")
    assert len(weighted) == len(uniform) == len(zero) == MAX_IR_TOKENS
    assert weighted[0][1] == pytest.approx(uniform[0][1])
    assert all(value == 0.0 for row in zero for value in row)


def test_token_only_ablation_keeps_scalar_evidence() -> None:
    embedding = RuleIRTokenEmbedding()
    ids, scalars = open_source_ir_token_inputs(embedding, _layers(), mode="tokens_zeroed")
    assert ids == [PAD_ID] * MAX_IR_TOKENS
    assert any(value != 0.0 for row in scalars for value in row)
