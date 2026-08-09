import json
import re
from pathlib import Path

from app.payload_catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg68_real_typed_oracle_is_separate_from_training_promotion():
    report = _read("pg68_real_local_docker_typed_oracle_adapter_report_v1.json")
    metrics = report["metrics"]
    assert report["status"] == "real_local_adapter_audit_completed"
    assert report["source"]["source_status"] == "real_local_oracle_completed"
    assert report["scope"]["loopback_only"] is True
    assert metrics["typed_positive_count"] == 7
    assert metrics["matched_negative_control_count"] == 7
    assert metrics["negative_control_oracle_false_count"] == 7
    assert metrics["evidence_hash_valid_count"] == 7
    assert metrics["get_post_covered"] == {"GET": 6, "POST": 1}
    assert set(metrics["oracle_modalities"]) == {"browser_dom_execution", "sql_ast_differential", "redirect_destination_controlled"}
    assert report["hard_gate"]["status"] == "blocked"
    assert report["hard_gate"]["checks"]["typed_oracle_contract_complete"] is True
    assert report["hard_gate"]["checks"]["fresh_reset_per_action"] is False
    assert report["hard_gate"]["checks"]["family_heldout_replay"] is False
    assert report["promotion"]["evaluation_catalog_generated"] is True
    assert report["promotion"]["training_catalog_generated"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg68_evaluation_catalog_has_provenance_but_no_raw_attack_strings():
    catalog = load_catalog(ROOT / "research" / "pg68_real_local_docker_typed_oracle_adapter_catalog_v1.json")
    assert catalog["catalog_id"] == "pg68-real-local-docker-evaluation-only"
    assert len(catalog["sources"]) == 2
    assert sum(len(source["samples"]) for source in catalog["sources"]) == 7
    assert catalog["safety"]["local_only"] is True
    assert catalog["safety"]["real_exploit_strings"] is False
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onload", "onerror", "union select", "password", "123456"):
        assert forbidden not in serialized
    for source in catalog["sources"]:
        assert source["provenance"]["source_type"] == "authorized_local_container"
        assert source["provenance"]["container_image_digest"].startswith("sha256:")
        assert source["provenance"]["external_network"] is False
        for sample in source["samples"]:
            assert sample["evaluator_state_visible"] is False
            assert sample["replay"]["fresh_reset"]["completed"] is True
            assert sample["replay"]["fresh_reset"]["fresh_target"] is False
            assert re.fullmatch(r"[0-9a-f]{64}", sample["evidence"]["adapter_evidence_sha256"])
            assert sample["payload"]["safety"]["no_external_network"] is True


def test_pg68_rejected_trace_keeps_family_out_of_opaque_model_fields():
    trace = _read("pg68_real_local_docker_typed_oracle_adapter_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_pg68"] is False
    assert trace["episode_count"] == 0
    assert trace["candidate_step_count"] == 7
    assert trace["replay_status"] == "rejected_before_training"
    assert all("fresh_reset" in reason for reason in trace["rejection_reasons"])
    for step in trace["steps"]:
        assert re.fullmatch(r"pg68-route-\d{2}", step["route_template_id"])
        assert re.fullmatch(r"pg68-probe-\d{2}", step["probe_ref"])
        assert step["fresh_reset_audit"]["fresh_target"] is False
        assert step["raw_probe_stored"] is False
        assert step["raw_response_stored"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_sha256"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    for forbidden in ("xss", "injection", "url_redirect", "<script", "union select", "onload"):
        assert forbidden not in serialized
