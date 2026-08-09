import json
import re
from pathlib import Path

from app.payload_catalog import load_catalog


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg69_hard_gate_proves_real_reset_and_unknown_abstention_without_promotion():
    report = _read("pg69_per_action_reset_unseen_family_report_v1.json")
    metrics = report["metrics"]
    assert report["status"] == "completed_evaluation"
    assert report["source"]["docker_case_count"] == 4
    assert report["source"]["workflow_case_count"] == 8
    assert report["source"]["independent_implementation_count"] == 2
    assert metrics["typed_positive_count"] == 12
    assert metrics["negative_control_pass_count"] == 12
    assert metrics["evidence_hash_valid_count"] == 12
    assert metrics["get_post_covered"] == {"GET": 7, "POST": 5}
    assert metrics["unique_candidate_target_instance_count"] == 12
    assert metrics["fresh_reset_per_action"] is True
    assert metrics["unknown_misname_count"] == 0
    assert metrics["unknown_strict_abstain"] is True
    assert metrics["family_holdout_candidate_count"] == 1
    assert report["hard_gate"]["status"] == "passed"
    assert report["hard_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_catalog_generated"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg69_catalog_is_quarantined_but_valid_and_raw_free():
    catalog = load_catalog(ROOT / "research/pg69_per_action_reset_unseen_family_catalog_v1.json")
    assert catalog["catalog_id"] == "pg69-per-action-reset-evaluation-only"
    assert len(catalog["sources"]) == 4
    assert sum(len(source["samples"]) for source in catalog["sources"]) == 12
    families = {sample["semantic"]["family"] for source in catalog["sources"] for sample in source["samples"]}
    assert "workflow_invariant" in families
    assert catalog["safety"]["real_exploit_strings"] is False
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    for forbidden in ("<script", "onload", "onerror", "union select", "password", "123456"):
        assert forbidden not in serialized
    for source in catalog["sources"]:
        assert source["provenance"]["external_network"] is False
        assert source["provenance"]["evaluator_state_visible"] is False
        for sample in source["samples"]:
            assert sample["evaluator_state_visible"] is False
            assert sample["payload"]["safety"]["no_external_network"] is True
            assert re.fullmatch(r"[0-9a-f]{64}", sample["evidence"]["adapter_evidence_sha256"])


def test_pg69_trace_accepts_only_opaque_family_free_steps():
    trace = _read("pg69_per_action_reset_unseen_family_trace_v1.json")
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert trace["model_retrained_on_pg69"] is False
    assert trace["episode_count"] == 3
    assert trace["accepted_episode_count"] == 3
    assert trace["validation_failures"] == []
    assert trace["raw_probe_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert trace["online_weight_update"] is False
    assert trace["long_term_memory_write"] is False
    for step in trace["steps"]:
        assert re.fullmatch(r"pg69-route-\d{2}", step["action_manifest"]["route_template_id"])
        assert re.fullmatch(r"pg69-probe-\d{2}", step["action_manifest"]["probe_ref"])
        assert step["fresh_reset"]["fresh_target"] is True
        assert step["fresh_reset"]["evaluator_state_hidden"] is True
        assert re.fullmatch(r"[0-9a-f]{64}", step["evidence_sha256"])
    serialized = json.dumps(trace, ensure_ascii=False).casefold()
    for forbidden in ("workflow_invariant", "xss", "injection", "url_redirect", "<script", "union select", "onload"):
        assert forbidden not in serialized
