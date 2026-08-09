from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _hash_without(value: dict, field: str) -> str:
    copy = dict(value)
    declared = copy.pop(field)
    actual = hashlib.sha256(json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    return actual


def test_pg140_catalog_is_bounded_and_information_complete_as_a_catalog():
    report = _load("pg140_information_complete_catalog_report_v1.json")
    quality = report["quality"]
    assert report["status"] == "completed_pg140_information_complete_catalog"
    assert report["hard_gates_passed"] is False
    assert report["training_eligible"] is False
    assert report["memory_promotion_allowed"] is False
    assert quality["catalog_row_count"] == 840
    assert quality["method_counts"] == {"GET": 420, "POST": 420}
    assert quality["fresh_reset_count"] == 840
    assert quality["matched_negative_control_count"] == 468
    assert quality["typed_oracle_count"] == 630
    assert quality["unknown_oracle_count"] == 210
    assert quality["complete_evidence_count"] == 840
    assert quality["unique_catalog_row_count"] == 840
    assert quality["hash_invalid_count"] == 0
    assert quality["context_binding_missing_count"] == 0
    assert quality["all_catalog_row_hashes_valid"] is True
    assert quality["all_required_information_explicit"] is True
    assert quality["capability_train_candidate_count"] == 384
    assert report["learning_policy"]["incomplete_rows"] == "schema_repair_or_representation_pretrain_only"


def test_pg140_manifest_and_rows_are_recomputable_without_raw_content():
    catalog = _load("pg140_information_complete_catalog_v1.json")
    model = _load("pg140_information_complete_model_dataset_v1.json")
    report = _load("pg140_information_complete_catalog_report_v1.json")
    trace = _load("pg140_information_complete_catalog_trace_v1.json")
    _hash_without(catalog, "catalog_sha256")
    _hash_without(model, "dataset_sha256")
    _hash_without(report, "report_sha256")
    _hash_without(trace, "trace_sha256")
    _hash_without(catalog["manifest"], "manifest_sha256")
    assert catalog["raw_probe_strings_stored"] is False
    assert catalog["raw_response_bodies_stored"] is False
    assert catalog["model_labels_stored"] is False
    assert model["labels_in_model_rows"] is False
    assert len(catalog["rows"]) == len(model["rows"]) == 840
    assert {row["catalog_row_id"] for row in catalog["rows"]} == {row["model_row_id"] for row in model["rows"]}
    assert all("[OBS]" in row["tokens"] for row in model["rows"])
    assert max(row["token_count"] for row in model["rows"]) <= 384
    assert any("obs.baseline.shape_class=unknown" in row["tokens"] for row in model["rows"])

    for row in catalog["rows"]:
        assert row["information_quality"]["raw_probe_stored"] is False
        assert row["information_quality"]["raw_response_body_stored"] is False
        assert row["information_quality"]["model_input_excludes_evaluator_label"] is True
        assert row["information_quality"]["replayable_complete"] == row["information_quality"]["capability_train_candidate"]
        assert row["evidence"]["evidence_sha256"] != "unknown"
        assert row["evidence"]["trace_sha256"] != "unknown"
        assert row["replay"]["fresh_reset"]["completed"] is True
        assert row["replay"]["fresh_reset"]["fresh_target"] is True
        for value in row["observation_projection"]["baseline"].values():
            assert value not in (None, "")
        for value in row["observation_projection"]["response"].values():
            assert value not in (None, "")

    text = json.dumps({"catalog": catalog, "model": model, "report": report}, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text


def test_pg140_protocol_keeps_catalog_repair_separate_from_capability_and_memory():
    protocol = _load("pg140_information_complete_catalog_protocol_v1.json")
    proposal = _load("pg140_information_complete_catalog_proposal_v1.json")
    assert protocol["local_only"] is True
    assert protocol["raw_content_storage"] is False
    assert protocol["capability_train_requires_original_missing_zero"] is True
    assert protocol["memory_promotion_requires_cross_seed_implementation_ood_review"] is True
    assert proposal["status"] == "evaluation_only_catalog_repair"
    assert proposal["training_eligible"] is False
    assert proposal["memory_promotion_allowed"] is False
    rules = _load("improvement_rules.json")
    policy = rules["pg140_information_complete_catalog"]
    assert policy["catalog_row_count"] == 840
    assert policy["explicit_unknown_projection"] is True
    assert policy["capability_train_candidate_count"] == 384
    assert policy["training_eligible"] is False
    assert policy["memory_promotion_allowed"] is False
