import json
from pathlib import Path

import torch

from app.pg257_rule_ir_class_decoder import RULE_CLASSES, RuleIRClassDecoder, class_target, evaluate_decoder


ROOT = Path(__file__).resolve().parents[1]


def test_pg257_report_is_seed_held_out_and_promotion_blocked():
    report = json.loads((ROOT / "research/pg257_widebyte_rule_ir_capacity_training_report_v1.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed_rule_ir_class_capacity_training"
    assert report["counts"]["records"] == 16
    assert report["counts"]["train_rows"] == 8
    assert report["counts"]["holdout_rows"] == 8
    assert report["selected"]["hidden_dim"] == 2048
    assert report["selected"]["metrics"]["seed_holdout"]["rule_accuracy"] == 1.0
    assert report["selected"]["metrics"]["seed_holdout"]["widebyte_escape_boundary_recall"] == 1.0
    assert report["model_input_excludes_oracle_target"] is True
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["honesty"]["general_web_capability_not_established"] is True


def test_pg257_decoder_has_auxiliary_token_and_rule_heads():
    model = RuleIRClassDecoder(d_model=16, hidden_dim=8, token_vocab_size=11)
    context = torch.randn(3, 5, 16)
    targets = torch.zeros(3, 5, dtype=torch.long)
    targets[:, 1] = torch.tensor([1, 2, 3])
    rule_targets = torch.tensor([class_target(name) for name in RULE_CLASSES])
    result = evaluate_decoder(model, context, targets, rule_targets, torch.tensor([1, 1, 1]))
    assert len(result["predicted_classes"]) == 3
    assert set(result["predicted_classes"]).issubset(set(RULE_CLASSES))
    assert result["token_count"] == 3
    assert result["row_count"] == 3
    assert set(model(context, classification_positions=torch.tensor([1, 1, 1]))) == {"token", "rule"}
