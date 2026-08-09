import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg40_catalog_semantic_refs_are_bounded_and_not_family_names():
    catalog = _load("pg40_semantic_router_catalog_v1.json")
    rows = catalog["samples"]
    assert len(rows) == 960
    assert catalog["typed_positive_count"] == 96
    assert catalog["negative_control_count"] == 864
    assert catalog["trace_episode_count"] == 60
    assert catalog["accepted_evaluation_episode_count"] == 48
    assert catalog["source_count"] == 2
    refs = {row["payload_manifest"]["probe_ref"] for row in rows}
    assert len(refs) == 9
    family_names = {"xss", "injection", "authentication", "access_control", "logic", "url_redirect", "input_validation", "command_injection"}
    assert all(not any(f"-{family}-" in ref or ref.endswith(f"-{family}") for family in family_names) for ref in refs)
    assert all(row["reset"]["fresh_target"] and row["reset"]["completed"] for row in rows)
    assert len({row["evidence"]["evidence_hash"] for row in rows}) == len(rows)
    serialized = json.dumps(catalog, ensure_ascii=False).casefold()
    assert "<script" not in serialized
    assert "onerror" not in serialized
    assert "union select" not in serialized


def test_pg40_source_transfer_is_successful_but_not_a_family_ood_claim():
    report = _load("pg40_semantic_router_report_v1.json")
    assert report["status"] == "diagnostic_only"
    assert report["model"]["semantic_reference_contains_family_name"] is False
    assert report["model"]["semantic_reference_contains_raw_probe"] is False
    assert report["model"]["typed_oracle_consumed_by_model"] is False
    assert report["splits"]["source_holdout"]["typed_recall"] == 1.0
    assert report["splits"]["source_holdout"]["effect_recall_any_family"] == 1.0
    assert report["splits"]["source_holdout"]["false_positive_rate"] == 0.0
    assert report["splits"]["negative_control"]["false_positive_rate"] == 0.0
    assert report["source_transfer_diagnostic"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False


def test_pg40_protocol_requires_strict_family_ood_before_promotion():
    protocol = _load("pg40_semantic_router_protocol_v1.json")
    assert protocol["semantic_ir_contract"]["reference_contains_family_name"] is False
    assert protocol["semantic_ir_contract"]["reference_contains_raw_probe"] is False
    assert protocol["split_plan"]["strict_family_holdout"].startswith("not claimed")
    assert protocol["run_result"]["strict_family_holdout_claim"] is False
    assert protocol["run_result"]["capability_claim_allowed"] is False
    assert protocol["status"] == "source_transfer_completed_family_ood_claim_blocked_no_promotion"
