import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_surface_discriminator_is_diagnostic_and_uses_strict_ood_abstain():
    report = _load("pg34_surface_discriminator_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["visible_projection_labels"] is False
    assert report["typed_oracle_consumed_by_model"] is False
    assert report["positive_authority"] is False
    assert report["training_allowed"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["checkpoint_selection"] == "minimum_train_cross_entropy"
    assert report["splits"]["train"]["route_accuracy"] == 1.0
    assert report["splits"]["train"]["accepted_accuracy"] == 1.0
    assert report["splits"]["negative_control"]["false_route_rate"] == 0.0
    assert report["splits"]["family_holdout"]["abstain_rate"] == 1.0
    assert report["splits"]["ood_source"]["abstain_rate"] == 1.0


def test_surface_discriminator_checkpoint_hash_matches_report():
    report = _load("pg34_surface_discriminator_report_v1.json")
    checkpoint = ROOT / report["checkpoint"]
    assert checkpoint.exists()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == report["checkpoint_sha256"]
