import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg35_protocol_requires_pairing_and_three_source_blind_gate():
    protocol = _load("pg35_encoding_pair_protocol_v1.json")
    assert protocol["scope"]["targets"] == ["127.0.0.1", "localhost", "workspace_authorized_docker_only"]
    assert protocol["scope"]["methods"] == ["GET", "POST"]
    assert protocol["scope"]["encodings"] == ["identity", "url_percent"]
    assert protocol["fixture"]["sample_count"] == 648
    assert protocol["fixture"]["typed_positive_count"] == 288
    assert protocol["fixture"]["encoding_pair_count"] == 324
    assert protocol["capability_gate"]["minimum_independent_source_hashes"] == 3
    assert protocol["capability_gate"]["zero_negative_false_accept"] is True
    assert protocol["capability_gate"]["encoding_pair_agreement_required"] is True
    assert protocol["training_policy"]["pair_consistency_loss_required"] is True
    assert protocol["training_policy"]["long_term_memory_write"] is False
    strict = protocol["strict_family_holdout_run"]
    assert strict["family_holdout_typed_recall"] == 0.0
    assert strict["family_holdout_false_positive_rate"] == 0.5
    assert strict["capability_gate_status"] == "no_proven_gain"
    assert strict["capability_claim_allowed"] is False


def test_pg35_improvement_rule_points_to_gate_first_artifacts():
    rules = _load("improvement_rules.json")
    policy = rules["pg35_encoding_pair_policy"]
    assert policy["required_channels"] == ["GET", "POST"]
    assert policy["required_encoding_pairs"] == ["identity", "url_percent"]
    assert policy["minimum_independent_source_hashes"] == 3
    assert policy["strict_ood_abstain"] is True
    assert policy["pair_consistency_required"] is True
    assert policy["capability_gate_before_training_promotion"] is True
    assert policy["memory_promotion_on_failure"] is False
