import json
from pathlib import Path

import torch

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig
from app.pg346_structured_target_slot import (
    SLOT_PREFIXES,
    StructuredSlotConfig,
    StructuredTargetSlotDecoder,
    build_slot_candidates,
    evaluate_structured_slots,
    predict_structured_slots,
    train_structured_slot_decoder,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixture():
    dataset = json.loads((ROOT / "research" / "pg345_decision_boundary_role_bound_dataset_v1.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((ROOT / "research" / "pg345_decision_boundary_vocabulary_v1.json").read_text(encoding="utf-8"))
    tokens = [PAD, UNK, *vocabulary["context_tokens"], *vocabulary["target_tokens"]]
    vocab = {token: index for index, token in enumerate(dict.fromkeys(tokens))}
    rows = [row for row in dataset["records"] if row["split"] == "train"][:4]
    return rows, vocab


def test_slot_candidates_cover_exact_rule_ir_slots_without_raw_fields():
    rows, vocab = _fixture()
    candidates = build_slot_candidates(vocab)
    assert set(candidates) == {name for name, _ in SLOT_PREFIXES}
    assert all(candidates[name] for name, _ in SLOT_PREFIXES)
    assert all(not any("payload" in str(token).casefold() or "response_body" in str(token).casefold() for token, index in vocab.items() if index in ids) for name, ids in candidates.items())


def test_structured_decoder_reads_context_only_and_returns_fixed_slots():
    rows, vocab = _fixture()
    candidates = build_slot_candidates(vocab)
    model = StructuredTargetSlotDecoder(
        vocab_size=len(vocab),
        config=CausalMoEConfig(d_model=24, n_heads=4, n_layers=1, experts=2, expert_hidden=48, max_length=256),
        slot_candidates=candidates,
    )
    result = predict_structured_slots(model, rows[:2], vocab, torch.device("cpu"))
    assert len(result) == 2
    assert all(result_row[0] == "[TARGET_BOS]" and result_row[-1] == "[TARGET_EOS]" for result_row in result)
    assert all(len(result_row) == 10 for result_row in result)
    metrics = evaluate_structured_slots(model, rows[:2], vocab, torch.device("cpu"))
    assert metrics["rows"] == 2
    assert set(metrics["slot_accuracy"]) == {name for name, _ in SLOT_PREFIXES}


def test_short_structured_training_keeps_language_model_backbone_and_slot_metrics():
    rows, vocab = _fixture()
    model = train_structured_slot_decoder(
        rows,
        vocab,
        torch.device("cpu"),
        seed=34601,
        config=CausalMoEConfig(d_model=24, n_heads=4, n_layers=1, experts=2, expert_hidden=48, max_length=256),
        slot_config=StructuredSlotConfig(language_model_weight=0.25, slot_weight=1.0),
        epochs=2,
        learning_rate=2e-4,
    )
    assert model.training is False
    result = evaluate_structured_slots(model, rows, vocab, torch.device("cpu"))
    assert result["rows"] == len(rows)
    assert "next_action" in result["slot_accuracy"]
    assert hasattr(model.backbone, "lm_head")
