import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg34_curriculum_is_local_only_and_gate_first():
    protocol = _load("pg34_multi_family_blackbox_agent_curriculum_v1.json")
    scope = protocol["scope"]
    assert scope["targets"] == ["127.0.0.1", "localhost", "workspace_authorized_docker_only"]
    assert scope["methods"] == ["GET", "POST"]
    assert scope["external_network"] is False
    assert scope["credentials"] is False
    assert scope["raw_probe_or_response_persistence"] is False
    assert len(protocol["vulnerability_families"]) >= 8
    assert len(protocol["experiments"]) == 8
    assert protocol["promotion_gate"]["zero_negative_false_accept"] is True
    assert protocol["promotion_gate"]["capability_gain_over_baseline_required"] is True
    assert protocol["first_run"]["experiment_id"] == "PG34-E01"


def test_improvement_rules_reference_pg34_without_relaxing_old_gates():
    rules = _load("improvement_rules.json")
    policy = rules["multi_family_blackbox_agent_curriculum_policy"]
    assert policy["protocol_file"].endswith("pg34_multi_family_blackbox_agent_curriculum_v1.json")
    assert policy["required_channels"] == ["GET", "POST"]
    assert policy["independent_target_implementation_required_for_ood_claim"] is True
    assert policy["strict_ood_abstain"] is True
    assert policy["capability_gate_before_training_promotion"] is True
    assert policy["checkpoint_selection_must_not_use_accuracy_only"] is True
    assert policy["surface_discriminator_positive_authority"] is False
    assert rules["project_purpose_and_dataset_utility_policy"]["schema_smoke_test_is_not_training_data"] is True
    protocol = _load("pg34_multi_family_blackbox_agent_curriculum_v1.json")
    assert protocol["independent_source_holdout"]["weights_updated"] is False
    assert protocol["independent_source_holdout"]["capability_claim_allowed"] is False
    assert protocol["surface_discriminator_diagnostic"]["checkpoint_selection"] == "minimum_train_cross_entropy"
    assert protocol["surface_discriminator_diagnostic"]["promotion_allowed"] is False
