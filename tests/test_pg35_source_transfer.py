import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg35_source_transfer_is_blind_on_route_implementation_and_not_a_family_holdout_claim():
    report = _load("pg35_source_transfer_diagnostic_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["model"]["visible_projection_labels"] is False
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["model"]["positive_authority"] is False
    for split in ("beta_source_holdout", "gamma_source_holdout"):
        assert report["splits"][split]["typed_recall"] == 1.0
        assert report["splits"][split]["precision"] == 1.0
        assert report["splits"][split]["false_positive_rate"] == 0.0
        assert report["pair_consistency"][split]["agreement_rate"] == 1.0
    assert report["splits"]["negative_control"]["false_positive_rate"] == 0.0
    assert report["capability_claim_allowed"] is False
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False


def test_pg35_protocol_records_source_transfer_without_promoting_it():
    protocol = _load("pg35_encoding_pair_protocol_v1.json")
    result = protocol["source_transfer_diagnostic"]
    assert result["blind_variants"] == ["beta", "gamma"]
    assert result["beta_typed_recall"] == 1.0
    assert result["gamma_typed_recall"] == 1.0
    assert result["beta_false_positive_rate"] == 0.0
    assert result["gamma_false_positive_rate"] == 0.0
    assert result["capability_claim_allowed"] is False
    rules = _load("improvement_rules.json")
    assert rules["pg35_source_transfer_diagnostic"]["source_transfer_is_not_family_holdout"] is True
