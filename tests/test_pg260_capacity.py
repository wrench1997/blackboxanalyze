import hashlib
import json
from pathlib import Path

import torch

from app.pg260_active_belief_adapter import PG260ActiveBeliefAdapter
from app.research_ops import build_research_ops_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_pg260_report_and_artifact_are_auditable():
    report = json.loads((ROOT / "research" / "pg260_active_belief_capacity_training_report_v1.json").read_text(encoding="utf-8"))
    artifact = ROOT / "artifacts" / "pg260-active-belief-capacity-v1" / "active_belief_hidden4096.pt"
    selected = report["selected"]
    fresh = selected["metrics"]["fresh_route_holdout"]
    judge = report["independent_final_judge"]
    assert report["status"] == "completed_pg260_active_belief_capacity_training"
    assert selected["hidden_dim"] == 4096
    assert selected["adapter_parameter_count"] > 50_000_000
    assert fresh["rule_accuracy"] >= 0.90
    assert fresh["family_accuracy"] >= 0.90
    assert judge["pass"] is False
    assert judge["holdout_support"]["sql_syntax"] >= 2
    assert judge["holdout_support"]["oracle_gap"] >= 2
    assert judge["reasons"] == ["holdout_rule_accuracy_ge_0_80", "implementation_ood_family_accuracy_ge_0_60"]
    assert report["evaluation_audit"]["weights_changed"] is False
    assert report["evaluation_audit"]["artifact_unchanged"] is True
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == selected["artifact_sha256"]


def test_pg260_is_visible_as_blocked_capacity_card_and_trainer_task():
    snapshot = build_research_ops_snapshot()
    card = snapshot["capability"]["model"]["pg260"]
    assert card["selected_hidden_dim"] == 4096
    assert card["fresh_route_rule_accuracy"] >= 0.90
    assert card["judge_pass"] is False
    assert card["promotion_blocked"] is True
    trainer = next(task for task in snapshot["tasks"]["trainer"] if task["id"] == "pg260-active-belief")
    assert trainer["status"] == "promotion_blocked"
    assert trainer["human_required"] is True


def test_pg261_masked_pool_is_invariant_to_batch_padding_width():
    torch.manual_seed(261)
    model = PG260ActiveBeliefAdapter(d_model=8, hidden_dim=16, token_vocab_size=11).eval()
    real = torch.randn(1, 3, 8)
    padded = torch.cat([real, torch.randn(1, 2, 8)], dim=1)
    positions = torch.tensor([1])
    with torch.no_grad():
        short = model(real, classification_positions=positions, attention_mask=torch.tensor([[1, 1, 1]]))
        long = model(padded, classification_positions=positions, attention_mask=torch.tensor([[1, 1, 1, 0, 0]]))
    assert torch.allclose(short["rule"], long["rule"], atol=1e-6)
    assert torch.allclose(short["family"], long["family"], atol=1e-6)
