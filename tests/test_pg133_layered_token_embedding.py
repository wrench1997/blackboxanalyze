from __future__ import annotations

import torch

from app.layered_token_embedding import (
    IR_SLOT_IDS,
    LayeredTokenEmbedding,
    MAX_LAYERED_TOKENS,
    PAD_ID,
    PG133_EXTRA_IR_SLOT_IDS,
    SCALAR_DIM,
    SPECIAL_TOKENS,
    VOCABULARY,
    canonical_layered_ir_pair_token,
    canonical_source_atom,
    layered_token_inputs,
)
from app.pg133_layered_token_policy import LayeredTokenActionPolicy


def _step() -> dict[str, object]:
    return {
        "source_token_layers": [
            {"modality": "html", "tokens": [{"kind": "tag", "value": "form"}, {"kind": "attribute", "value": "name", "value_hash": "f" * 64}]},
            {"modality": "javascript", "tokens": [{"kind": "api", "value": "fetch", "count_bucket": "1-4"}]},
            {"modality": "transport", "tokens": [{"kind": "method", "value": "POST"}, {"kind": "placement", "value": "json"}]},
        ],
        "ir_layer": {"tokens": [{"slot_id": slot, "value": "unknown", "weight": 1.0} for slot in IR_SLOT_IDS]},
    }


def test_source_and_ir_atoms_use_special_boundaries_and_no_raw_hash() -> None:
    assert {"[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]", "[STEP]"}.issubset(set(SPECIAL_TOKENS))
    atom = canonical_source_atom("html", {"kind": "attribute", "value": "secret", "value_hash": "a" * 64})
    assert atom == "src.html.attribute=hash_present"
    assert "secret" not in atom
    assert len(VOCABULARY) > len(SPECIAL_TOKENS)
    assert PG133_EXTRA_IR_SLOT_IDS == ("oracle.availability",)
    assert canonical_layered_ir_pair_token("oracle.availability", "typed") != canonical_layered_ir_pair_token("oracle.availability", "unknown")


def test_layered_inputs_are_bounded_and_token_only_ablation_is_separate() -> None:
    embedding = LayeredTokenEmbedding(seed=13302)
    ids, scalars = layered_token_inputs(embedding, [_step(), _step()])
    assert len(ids) == MAX_LAYERED_TOKENS
    assert len(scalars) == MAX_LAYERED_TOKENS
    assert len(scalars[0]) == SCALAR_DIM
    token_ids_zeroed, scalar_kept = layered_token_inputs(embedding, [_step()], mode="tokens_zeroed")
    assert token_ids_zeroed == [PAD_ID] * MAX_LAYERED_TOKENS
    assert any(value != 0.0 for row in scalar_kept for value in row)
    zero_ids, zero_scalars = layered_token_inputs(embedding, [_step()], mode="zero")
    assert zero_ids == [PAD_ID] * MAX_LAYERED_TOKENS
    assert all(value == 0.0 for row in zero_scalars for value in row)


def test_layered_policy_forward_shape() -> None:
    model = LayeredTokenActionPolicy(embedding_dim=16, embedding_seed=13303)
    ids, scalars = layered_token_inputs(model.token_embedding, [_step()])
    logits = model(torch.tensor([ids], dtype=torch.long), torch.tensor([scalars], dtype=torch.float32))
    assert logits.shape == (1, 7)
    assert model.embedding_provenance["tokenizer_backend"] == "huggingface-tokenizers-layered-wordlevel"
    assert model.embedding_provenance["pretrained"] is False
