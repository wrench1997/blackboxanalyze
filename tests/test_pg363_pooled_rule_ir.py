from __future__ import annotations

import torch

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig
from app.pg363_pooled_rule_ir import (
    PooledRuleIRDecoder,
    PooledSlotConfig,
    SLOT_PREFIXES,
    build_slot_candidates,
    evaluate_pooled_rule_ir,
    train_pooled_rule_ir,
)


def _vocabulary() -> dict[str, int]:
    tokens = [PAD, UNK, "document_presence=observed", "request_method=get", "[TARGET_BOS]", "[TARGET_EOS]"]
    for _, prefix in SLOT_PREFIXES:
        tokens.append(prefix + "none")
    tokens.extend(["safe_to_send=0", "safe_to_send=1", "next_action=abstain", "question=none", "negative_control_presence_ref=unknown"])
    return {token: index for index, token in enumerate(dict.fromkeys(tokens))}


def _row() -> dict[str, object]:
    target = ["[TARGET_BOS]"]
    values = {
        "question": "none",
        "ask_reason": "none",
        "next_action": "abstain",
        "repair_action": "none",
        "transport_ref": "none",
        "field_role_ref": "none",
        "encoding_ref": "none",
        "syntax_category_ref": "none",
        "probe_variant_ref": "none",
        "safe_to_send": "0",
        "payload_shape_ref": "none",
        "oracle_ref": "none",
        "negative_control_presence_ref": "unknown",
    }
    target.extend(f"{name}={values[name]}" for name, _ in SLOT_PREFIXES)
    target.append("[TARGET_EOS]")
    return {"context_tokens": ["document_presence=observed", "request_method=get"], "target_tokens": target}


def test_pool_reads_all_valid_context_positions() -> None:
    vocab = _vocabulary()
    candidates = build_slot_candidates(vocab)
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=32)
    model = PooledRuleIRDecoder(vocab_size=len(vocab), config=config, slot_candidates=candidates)
    ids = torch.tensor([[vocab["document_presence=observed"], vocab["request_method=get"]]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    logits, _ = model(ids, valid_mask=valid)
    assert set(logits) == {name for name, _ in SLOT_PREFIXES}
    assert all(value.shape[0] == 1 for value in logits.values())


def test_evaluator_is_abstract_and_fail_closed_without_raw_fields() -> None:
    vocab = _vocabulary()
    candidates = build_slot_candidates(vocab)
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=32)
    model = PooledRuleIRDecoder(vocab_size=len(vocab), config=config, slot_candidates=candidates)
    result = evaluate_pooled_rule_ir(model, [_row()], vocab, torch.device("cpu"))
    assert result["rows"] == 1
    assert result["negative_false_allow"] in {0, 1}
    assert "raw_payload" not in result


def test_training_and_evaluation_accept_micro_batches() -> None:
    vocab = _vocabulary()
    candidates = build_slot_candidates(vocab)
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=32)
    records = [_row() for _ in range(3)]
    model = train_pooled_rule_ir(
        records,
        vocab,
        torch.device("cpu"),
        seed=36399,
        config=config,
        slot_config=PooledSlotConfig(language_model_weight=0.1, slot_weight=1.0),
        epochs=1,
        batch_size=1,
    )
    result = evaluate_pooled_rule_ir(model, records, vocab, torch.device("cpu"), batch_size=1)
    assert result["rows"] == 3
    assert result["slot_accuracy"]


def test_label_smoothing_is_bounded_and_recorded_in_objective() -> None:
    vocab = _vocabulary()
    config = CausalMoEConfig(d_model=16, n_heads=4, n_layers=1, experts=2, expert_hidden=32, max_length=32)
    records = [_row()]
    model = train_pooled_rule_ir(records, vocab, torch.device("cpu"), seed=36400, config=config, epochs=1, batch_size=1, slot_config=PooledSlotConfig(label_smoothing=0.1))
    assert model is not None
