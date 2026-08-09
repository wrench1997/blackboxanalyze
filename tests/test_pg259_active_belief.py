import json
from pathlib import Path

import torch

from app.pg259_active_belief_rule_ir_adapter import (
    ActiveBeliefRuleIRAdapter,
    BELIEF_CLASSES,
    FAMILY_CLASSES,
    PROBE_CLASSES,
    RULE_IR_CLASSES,
    belief_target,
    evaluate_active_adapter,
    family_target,
    probe_target,
    rule_target,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pg259_report_keeps_fresh_and_ood_gates_independent():
    report = json.loads((ROOT / "research/pg259_active_belief_capacity_training_report_v1.json").read_text(encoding="utf-8"))
    assert report["counts"]["records"] == 226
    assert report["counts"]["fresh_train_rows"] == 10
    assert report["counts"]["fresh_holdout_rows"] == 8
    assert report["selected"]["hidden_dim"] == 4096
    assert report["independent_final_judge"]["pass"] is False
    assert "fresh_route_rule_accuracy_ge_0_70" in report["independent_final_judge"]["reasons"]
    assert report["catastrophic_forgetting_canary"]["pass"] is True
    assert report["model_input_excludes_oracle_target"] is True
    assert report["promotion"]["training_promotion_allowed"] is False


def test_pg259_adapter_exposes_active_belief_and_probe_heads():
    model = ActiveBeliefRuleIRAdapter(d_model=12, hidden_dim=8, token_vocab_size=17)
    context = torch.randn(4, 6, 12)
    token_targets = torch.zeros(4, 6, dtype=torch.long)
    token_targets[:, 1] = torch.tensor([1, 2, 3, 4])
    rules = torch.tensor([rule_target(name) for name in RULE_IR_CLASSES[:4]])
    families = torch.tensor([family_target(name) for name in ("sql", "dom", "other", "sql")])
    beliefs = torch.tensor([belief_target(name) for name in BELIEF_CLASSES])
    probes = torch.tensor([probe_target(name) for name in PROBE_CLASSES])
    outputs = model(context, classification_positions=torch.tensor([1, 2, 3, 4]))
    result = evaluate_active_adapter(model, context, token_targets, rules, families, beliefs, probes, torch.tensor([1, 2, 3, 4]))
    assert set(outputs) == {"token", "rule", "family", "belief", "probe"}
    assert result["token_count"] == 4
    assert result["row_count"] == 4
    assert set(result["predicted_beliefs"]).issubset(set(BELIEF_CLASSES))
    assert set(result["predicted_probes"]).issubset(set(PROBE_CLASSES))
