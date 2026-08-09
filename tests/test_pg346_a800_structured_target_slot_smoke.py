import json
from pathlib import Path

import torch

from app.pg295_causal_moe import CausalMoEConfig
from app.pg346_structured_target_slot import StructuredTargetSlotDecoder, build_slot_candidates
from scripts.run_pg346_a800_structured_target_slot_smoke import _predictive_entropy, _vocabulary_map


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    dataset = json.loads((ROOT / "research" / "pg345_decision_boundary_role_bound_dataset_v1.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((ROOT / "research" / "pg345_decision_boundary_vocabulary_v1.json").read_text(encoding="utf-8"))
    return [row for row in dataset["records"] if row["split"] == "implementation_holdout"][:2], _vocabulary_map(vocabulary)


def test_structured_runner_entropy_uses_context_only():
    rows, vocab = _fixture()
    model = StructuredTargetSlotDecoder(
        vocab_size=len(vocab),
        config=CausalMoEConfig(d_model=24, n_heads=4, n_layers=1, experts=2, expert_hidden=48, max_length=256),
        slot_candidates=build_slot_candidates(vocab),
    )
    value = _predictive_entropy(model, rows, vocab, torch.device("cpu"))
    assert value > 0.0


def test_runner_vocab_map_is_append_only_abstract_union():
    _, vocab = _fixture()
    assert "[TARGET_BOS]" in vocab
    assert any(token.startswith("question=") for token in vocab)
    assert not any("payload=" in token or "response_body=" in token for token in vocab)
