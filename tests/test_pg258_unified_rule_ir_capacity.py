import json
from pathlib import Path

import torch

from app.pg258_unified_rule_ir_adapter import (
    FAMILY_CLASSES,
    RULE_IR_CLASSES,
    UnifiedRuleIRCapacityAdapter,
    evaluate_unified_adapter,
    family_target,
    rule_target,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pg258_report_records_negative_generalization_and_freezing():
    report = json.loads((ROOT / "research/pg258_unified_rule_ir_capacity_report_v1.json").read_text(encoding="utf-8"))
    judge = report["independent_final_judge"]
    assert report["counts"]["records"] == 208
    assert report["counts"]["train_rows"] == 164
    assert report["counts"]["holdout_rows"] == 23
    assert report["counts"]["implementation_ood_rows"] == 21
    assert judge["pass"] is False
    assert "implementation_ood_family_accuracy_ge_0_60" in judge["reasons"]
    assert report["catastrophic_forgetting_canary"]["pass"] is True
    assert report["catastrophic_forgetting_canary"]["state_unchanged"] is True
    assert report["model_input_excludes_oracle_target"] is True
    assert report["promotion"]["training_promotion_allowed"] is False
    assert report["honesty"]["general_web_capability_not_established"] is True


def test_pg258_adapter_exposes_token_rule_and_family_heads():
    model = UnifiedRuleIRCapacityAdapter(d_model=12, hidden_dim=8, token_vocab_size=17)
    context = torch.randn(3, 6, 12)
    token_targets = torch.zeros(3, 6, dtype=torch.long)
    token_targets[:, 1] = torch.tensor([1, 2, 3])
    rules = torch.tensor([rule_target(name) for name in RULE_IR_CLASSES[:3]])
    families = torch.tensor([family_target(name) for name in FAMILY_CLASSES])
    result = evaluate_unified_adapter(model, context, token_targets, rules, families, torch.tensor([1, 2, 3]))
    assert set(model(context, classification_positions=torch.tensor([1, 2, 3]))) == {"token", "rule", "family"}
    assert result["token_count"] == 3
    assert result["row_count"] == 3
    assert set(result["predicted_rule_classes"]).issubset(set(RULE_IR_CLASSES))
    assert set(result["predicted_families"]).issubset(set(FAMILY_CLASSES))
