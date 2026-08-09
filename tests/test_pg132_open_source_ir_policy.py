from __future__ import annotations

import torch

from app.open_source_token_embedding import MAX_IR_TOKENS, SCALAR_DIM, open_source_ir_token_inputs
from app.pg132_open_source_ir_policy import OpenSourceIRActionPolicy


def _layers() -> list[dict[str, object]]:
    return [{"tokens": [{"slot_id": "failure.kind", "value": "oracle_unavailable", "weight": 2.0}]}]


def test_pg132_forward_uses_token_ids_and_scalars() -> None:
    model = OpenSourceIRActionPolicy(embedding_dim=16, embedding_seed=13202)
    ids, scalars = open_source_ir_token_inputs(model.token_embedding, _layers())
    logits = model(
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([scalars], dtype=torch.float32),
    )
    assert logits.shape == (1, 7)
    assert model.embedding_provenance["tokenizer_backend"] == "huggingface-tokenizers-wordlevel"
    assert model.embedding_provenance["pretrained"] is False


def test_pg132_zero_mode_masks_pair_ids_and_scalar_channel() -> None:
    model = OpenSourceIRActionPolicy()
    ids, scalars = open_source_ir_token_inputs(model.token_embedding, _layers(), mode="zero")
    assert ids == [0] * MAX_IR_TOKENS
    assert all(value == 0.0 for row in scalars for value in row)
    assert len(scalars[0]) == SCALAR_DIM
