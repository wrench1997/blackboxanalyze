import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_pg36_failure_is_classified_as_experiment_problem_without_authority():
    result = _load("research/pg36_formal_rule_ir_triage_v1.json")
    assert result["classification"] == "experiment_problem"
    assert result["triage"]["engineering_path"]["signals"] == []
    assert result["decision"]["model_change_authorized_by_triage"] is False
    assert result["decision"]["checkpoint_promotion"] is False
    assert result["decision"]["memory_promotion"] is False
    assert result["decision"]["infrastructure_scale"] is False
    assert result["decision"]["next_experiment"].startswith("PG-37")


def test_pg36_failure_triage_policy_requires_new_ablation_and_blocks_scale():
    rules = _load("research/improvement_rules.json")
    policy = rules["pg36_failure_triage_policy"]
    assert policy["family_holdout_failure_is_experiment_signal"] is True
    assert policy["full_regression_required_before_model_change"] is True
    assert policy["model_change_requires_new_preregistered_ablation"] is True
    assert policy["compute_scaling_on_family_failure"] is False
    assert policy["checkpoint_promotion_on_failure"] is False
