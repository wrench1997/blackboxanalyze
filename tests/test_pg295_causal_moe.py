import json
from pathlib import Path

import torch

from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, _batch, build_vocabulary, evaluate_causal_moe, generate_target, train_causal_moe


ROOT = Path(__file__).resolve().parents[1]


def test_causal_moe_has_lm_head_and_no_action_classifier():
    model = CausalMoELanguageModel(vocab_size=32, config=CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=64))
    ids = torch.randint(0, 32, (2, 8))
    logits, balance = model(ids, valid_mask=torch.ones_like(ids, dtype=torch.bool))
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(balance)
    assert not hasattr(model, "action_head")
    hidden, hidden_balance = model.forward_hidden(ids, valid_mask=torch.ones_like(ids, dtype=torch.bool))
    assert hidden.shape == (2, 8, 32)
    assert torch.isfinite(hidden_balance)


def test_pg295_report_records_question_vs_answer_only_gap():
    report = json.loads((ROOT / "research" / "pg295_causal_moe_training_report_v1_local_morning.json").read_text(encoding="utf-8"))
    assert report["objective"].startswith("causal next-token")
    assert report["selection"]["seed_missing_question_recall_min"] == 1.0
    assert report["answer_only_control"]["missing_holdout"]["missing_question_recall"]["mean"] == 0.0
    assert report["selection"]["hard_negative_false_allow_max"] == 8
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_short_causal_training_evaluates_without_teacher_forcing():
    dataset = json.loads((ROOT / "research" / "pg294_active_repair_dataset_v1.json").read_text(encoding="utf-8"))
    rows = [row for row in dataset["records"] if row["split"] == "train"][:4]
    vocabulary = build_vocabulary(rows)
    model = train_causal_moe(rows, vocabulary, torch.device("cpu"), seed=29599, config=CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=128), epochs=2)
    result = evaluate_causal_moe(model, rows[:2], vocabulary, torch.device("cpu"))
    assert result["count"] == 2
    assert "sequence_exact_accuracy" in result
    assert "missing_question_recall" in result


def test_weighted_loss_can_normalize_by_effective_slot_mass():
    rows = [{"context_tokens": ["page=abstract"] * 16, "target_tokens": ["[TARGET_BOS]", "question=ask_typed", "[TARGET_EOS]"]}]
    vocabulary = build_vocabulary(rows)
    model = train_causal_moe(
        rows,
        vocabulary,
        torch.device("cpu"),
        seed=29598,
        config=CausalMoEConfig(d_model=32, n_heads=4, n_layers=1, experts=2, expert_hidden=64, max_length=64),
        epochs=1,
        token_weights={"page=abstract": 0.0, "question=ask_typed": 1.0, "[TARGET_BOS]": 1.0, "[TARGET_EOS]": 1.0},
        normalize_weighted_loss=True,
    )
    assert model.training is False


def test_batch_respects_large_pg331_context_capacity():
    rows = [{"context_tokens": [f"field_{index}" for index in range(180)], "target_tokens": ["[TARGET_BOS]", "[TARGET_EOS]"]}]
    vocabulary = build_vocabulary(rows)
    ids, valid = _batch(rows, vocabulary, torch.device("cpu"), max_length=256)
    assert ids.shape == (1, 182)
    assert int(valid.sum().item()) == 182


def test_generate_target_predicts_target_bos_after_context_not_before_it():
    class StubModel:
        config = type("Config", (), {"max_length": 32})()

        def __call__(self, input_ids, valid_mask=None):
            import torch

            last = int(input_ids[0, -1].item())
            next_id = {1: 2, 2: 3, 3: 4}[last]
            logits = torch.full((1, input_ids.shape[1], 5), -100.0)
            logits[0, -1, next_id] = 100.0
            return logits, torch.tensor(0.0)

    vocabulary = {"[UNK]": 0, "context=seen": 1, "[TARGET_BOS]": 2, "question=ask_typed": 3, "[TARGET_EOS]": 4}
    prediction = generate_target(
        StubModel(),
        ["context=seen"],
        3,
        vocabulary,
        torch.device("cpu"),
    )
    assert prediction == ["[TARGET_BOS]", "question=ask_typed", "[TARGET_EOS]"]
