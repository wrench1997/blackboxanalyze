import json
from pathlib import Path


def _load(name: str):
    return json.loads((Path("research") / name).read_text(encoding="utf-8"))


def test_pg165_attests_only_bounded_surface_effects():
    report = _load("pg165_surface_attestation_report_v1.json")
    dataset = _load("pg165_surface_attested_training_dataset_v1.json")
    assert report["status"] == "completed_pg165_surface_attestation"
    assert report["source"]["row_count"] == 28
    assert report["checks"]["get_count"] == report["checks"]["post_count"] == 14
    assert report["checks"]["all_evidence_hashes_valid"] is True
    assert report["checks"]["all_attestation_hashes_valid"] is True
    assert report["attestation"]["confirmed_safe_effect_count"] == 1
    assert report["attestation"]["confirmed_safe_no_effect_count"] == 13
    assert report["claim"]["vulnerability_claim_allowed"] is False
    assert report["claim"]["execution_oracle"] is False
    assert len(dataset["training_eligible_rows"]) == 28
    assert dataset["training_role"] == "surface_effect_only"
    assert dataset["vulnerability_claim_allowed"] is False
    assert dataset["oracle_labels_in_model_input"] is False
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    token_text = json.dumps(dataset["rows"], ensure_ascii=False).lower()
    assert "payload" not in token_text
    assert "evaluator" not in token_text


def test_pg165_registry_is_data_only_and_not_vulnerability_promotion():
    registry = _load("pg_pk_24_cross_lab_registry_v1.json")
    entry = next(item for item in registry["targets"] if item["target_id"] == "pg165_surface_attestation")
    assert registry["training_eligible_target_count"] == 40
    assert registry["evaluation_only_target_count"] == 116
    assert entry["training_eligible"] is True
    assert entry["training_role"] == "surface_effect_only"
    assert entry["vulnerability_claim_allowed"] is False
    assert entry["training_artifact_promotion_allowed"] is False
    assert entry["memory_promotion_allowed"] is False
