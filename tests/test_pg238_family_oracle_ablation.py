import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg238_fresh_family_replay_is_evaluation_only() -> None:
    report = json.loads((ROOT / "research" / "pg238_pikachu_surface_replay_report_v1.json").read_text(encoding="utf-8-sig"))
    assert report["status"] == "completed_fresh_unseen_surface_replay"
    assert report["seeds"] == [23801, 23802]
    assert report["counts"]["fresh_container_count"] == 14
    assert report["counts"]["ai_candidate_send_count"] == 14
    assert report["counts"]["reference_send_count"] == 14
    assert report["counts"]["negative_send_count"] == 14
    assert report["counts"]["dom_surface_effect_confirmed_count"] == 4
    assert report["counts"]["redirect_effect_confirmed_count"] == 0
    assert report["counts"]["xss_positive_count"] == 0
    assert report["counts"]["open_redirect_positive_count"] == 0
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False


def test_pg238_family_holdout_and_oracle_ablation_are_reported() -> None:
    report = json.loads((ROOT / "research" / "pg238_family_oracle_ablation_report_v1.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((ROOT / "research" / "pg238_family_oracle_ablation_dataset_v1.json").read_text(encoding="utf-8-sig"))
    rule = json.loads((ROOT / "research" / "improvement_rules.json").read_text(encoding="utf-8-sig"))["pg238_family_holdout_oracle_ablation"]
    metrics = report["selected"]["metrics"]
    artifact = ROOT / report["selected"]["artifact"]

    assert report["status"] == "completed_family_holdout_oracle_ablation_training"
    assert report["counts"]["sql_train_rows"] == 100
    assert report["counts"]["sql_seed_holdout_rows"] == 21
    assert report["counts"]["family_holdout_rows"] == 14
    assert report["counts"]["combined_holdout_action_counts"] == {"abstain": 31, "send_candidate": 4}
    assert metrics["sql_seed_holdout"]["positive_send_recall"] == 1.0
    assert metrics["sql_seed_holdout"]["abstain_recall"] == 1.0
    assert metrics["family_holdout"]["abstain_recall"] == 1.0
    assert metrics["family_holdout"]["false_send_count"] == 0
    assert metrics["oracle_ablation_combined"]["next_token_accuracy"] == 0.81825397
    assert report["safety_gate_pass"] is True
    assert report["capability_gate_pass"] is True
    assert report["frozen_body_changed"] is False
    assert artifact.exists()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == report["selected"]["artifact_sha256"]
    assert dataset["contract"]["family_holdout_never_in_training"] is True
    assert dataset["contract"]["dom_effect_not_xss"] is True
    assert dataset["contract"]["redirect_effect_not_open_redirect"] is True
    assert dataset["contract"]["raw_payload_strings_stored"] is False
    assert dataset["contract"]["raw_response_bodies_stored"] is False
    assert rule["oracle_ablation_is_diagnostic_only"] is True
    assert rule["memory_promotion_allowed"] is False

