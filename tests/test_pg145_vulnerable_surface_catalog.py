from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg145_has_150_local_targets_and_three_families():
    report = _load("pg145_local_multisurface_report_v1.json")
    assert report["status"] == "completed_pg145_local_multisurface_vulnerability_catalog"
    assert report["hard_gates_passed"] is True
    assert report["training_eligible"] is False
    assert report["memory_promotion_allowed"] is False
    assert report["summary"]["target_instance_count"] == 150
    assert report["summary"]["row_count"] == 900
    assert report["summary"]["family_counts"] == {
        "sql_ast_boundary": 200,
        "xss_reflection": 200,
        "xxe_entity_parser": 200,
    }
    assert report["summary"]["method_counts"] == {"GET": 300, "POST": 300}
    assert report["summary"]["positive_count"] == 300
    assert report["summary"]["matched_negative_count"] == 300
    assert report["summary"]["unknown_oracle_count"] == 300
    assert report["summary"]["fresh_reset_count"] == 900
    assert report["summary"]["implementation_holdout_count"] == 180
    assert report["summary"]["source_hash_count"] == 12
    assert report["required_gates"]["external_bypass_payloads_forbidden"] is True
    assert report["model_input_contract"]["raw_source_included"] is False
    assert report["model_input_contract"]["oracle_authority_included"] is False
    assert report["model_input_contract"]["waf_scope"] == "local_mock_only"


def test_pg145_catalog_and_dataset_hashes_and_safe_model_projection():
    catalog = _load("pg145_local_multisurface_vulnerability_catalog_v1.json")
    declared = catalog.pop("catalog_sha256")
    actual = hashlib.sha256(json.dumps(catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert catalog["scope"]["base_url"] == "http://127.0.0.1:14500"
    assert catalog["raw_source_retained"] is False
    assert catalog["raw_payload_retained"] is False
    manifest = _load("pg145_style_reference_manifest_v1.json")
    declared_manifest = manifest.pop("manifest_sha256")
    actual_manifest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared_manifest == actual_manifest
    assert catalog["style_reference_manifest_sha256"] == declared_manifest
    assert manifest["collection_policy"] == "public_design_metadata_only_no_raw_crawl"
    assert manifest["raw_html_stored"] is False
    assert manifest["raw_javascript_stored"] is False
    assert manifest["probe_sent"] is False
    assert len({row["target_instance_id"] for row in catalog["rows"]}) == 150
    assert all(row["target_url"].startswith("http://127.0.0.1:14500/") for row in catalog["rows"])
    assert all(row["fresh_reset"]["fresh"] is True for row in catalog["rows"])
    assert all(row["oracle"]["evidence_hash"] for row in catalog["rows"])
    assert all(row["waf"]["waf_training_scope"] == "local_mock_only" for row in catalog["rows"])
    assert all(row["waf"]["external_bypass_payloads"] is False for row in catalog["rows"])
    dataset = _load("pg145_local_multisurface_model_dataset_v1.json")
    declared = dataset.pop("dataset_sha256")
    actual = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert all(row["labels_in_model_row"] is False for row in dataset["rows"])
    assert all(row["oracle_authority_in_model_row"] is False for row in dataset["rows"])
    assert all(row["target_identity_in_model_row"] is False for row in dataset["rows"])
    assert all(row["action_supervision_allowed"] is False for row in dataset["rows"])
    assert dataset["model_input_contract"]["external_bypass_payloads"] is False
    text = json.dumps(dataset, ensure_ascii=False).casefold()
    assert "<script" not in text
    assert "onerror" not in text
    assert "union select" not in text
    assert "file:///" not in text


def test_pg145_protocol_trace_and_proposal_are_hashed_and_evaluation_only():
    protocol = _load("pg145_local_multisurface_protocol_v1.json")
    declared = protocol.pop("protocol_sha256")
    actual = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert protocol["target_instance_minimum"] == 150
    assert protocol["required_gates"]["target_instance_count_ge_150"] is True
    assert protocol["required_gates"]["external_bypass_payloads_forbidden"] is True
    proposal = _load("pg145_local_multisurface_proposal_v1.json")
    declared = proposal.pop("proposal_sha256")
    actual = hashlib.sha256(json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert proposal["training_eligible"] is False
    trace = _load("pg145_local_multisurface_trace_v1.json")
    declared = trace.pop("trace_sha256")
    actual = hashlib.sha256(json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert declared == actual
    assert trace["raw_model_input_absent"] is True
    assert trace["waf_training_scope"] == "local_mock_only"
    assert trace["external_bypass_payloads"] is False
