import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg106_cross_implementation_and_decoy_gate_passes_without_promotion():
    report = _load("pg106_decoy_projection_holdout_report_v1.json")
    assert report["status"] == "passed_cross_implementation_decoy_diagnostic"
    assert report["capability_gate"]["claim_allowed"] is False
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    guarded = report["metrics"]["guarded_proposal"]["all_evaluation"]
    assert guarded["known_confirm_recall"] == 1.0
    assert guarded["false_accept_count"] == 0
    assert report["metrics"]["guarded_proposal"]["pg69"]["unknown_family_strict_abstain"] is True
    assert report["metrics"]["guarded_proposal"]["pg106_independent"]["unknown_family_strict_abstain"] is True
    decoy = report["metrics"]["decoy"]
    assert decoy["decoy_count"] == 4
    assert decoy["decoy_anomaly_count"] == 4
    assert decoy["decoy_false_confirm_count"] == 0
    assert decoy["decoy_abstain_count"] == 4
    assert decoy["opaque_positive_count"] == 4
    assert decoy["opaque_positive_anomaly_count"] == 4
    composition = report["metrics"]["compositional_rule_ir_ablation"]
    assert composition["candidate_promotion_eligible_count"] == 0
    assert composition["cross_sample_recombination_executable"] is False


def test_pg106_visible_dataset_is_fresh_get_post_and_bounded():
    report = _load("pg106_decoy_projection_holdout_report_v1.json")
    dataset = _load("pg106_decoy_projection_holdout_visible_dataset_v1.json")
    trace = _load("pg106_decoy_projection_holdout_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert trace["evaluator_labels_in_trace"] is False
    assert len(dataset["rows"]) == 578
    assert len(trace["steps"]) == 578
    assert sorted({row["method"] for row in dataset["rows"]}) == ["GET", "POST"]
    assert len({row["fresh_reset"]["target_instance_id"] for row in dataset["rows"]}) == 578
    assert all(row["negative_control_matched"] for row in dataset["rows"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]) for row in dataset["rows"])
    assert all(not row["raw_probe_strings_stored"] and not row["raw_response_body_stored"] for row in dataset["rows"])
    assert all(row["guarded_proposal"]["promotion_eligible"] is False for row in dataset["rows"])
    assert all(
        isinstance(row["model_input"].get("causal_extension"), dict)
        for row in dataset["rows"]
        if row["source"] in {"pg69", "pg106"}
    )
    text = json.dumps({"report": report, "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("workflow_invariant", "xss", "union select", "<script", "onerror", '"subject":', '"shape":'):
        assert forbidden not in text
    for name, digest in report["source"]["source_hashes"].items():
        path = {
            "pg101_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
            "pg103_dataset": "research/pg103_auto_goal_label_active_probe_visible_dataset_v1.json",
            "pg103_trace": "research/pg103_auto_goal_label_active_probe_trace_v1.json",
            "pg69_fixture": "app/pg69_workflow_fixture.py",
            "pg106_fixture": "app/pg106_decoy_projection_fixture.py",
            "causal_module": "app/pg105_observable_projection.py",
            "inducer_module": "app/active_goal_label_inducer.py",
            "runner": "scripts/run_pg106_decoy_projection_holdout.py",
        }[name]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
