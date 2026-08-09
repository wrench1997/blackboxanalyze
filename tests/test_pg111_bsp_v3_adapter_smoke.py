import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg111_structure_adapter_smoke_passes_without_training_or_old_weights():
    report = _load("pg111_bsp_v3_adapter_smoke_report_v1.json")
    assert report["status"] == "passed_bsp_v3_structure_adapter_smoke"
    assert report["capability_gate"]["claim_allowed"] is False
    assert all(report["capability_gate"]["checks"].values())
    assert report["scope"]["dog_project_imported"] is False
    assert report["scope"]["dog_weights_loaded"] is False
    assert report["scope"]["mandarin_foundation_used"] is False
    assert report["scope"]["full_training_started"] is False
    assert report["scope"]["cuda_execution_started"] is False
    assert report["contract"]["python_reference_core"] is True
    assert report["lineage"]["architecture_transfer_mode"] == "bsp_v3_structure_contract_only"
    assert report["lineage"]["parent_checkpoint_reused"] is False
    assert report["lineage"]["weight_load_performed"] is False
    assert report["lineage"]["dataset_merge"] is False
    assert report["replay"]["executable"] is False
    assert report["replay"]["typed_oracle_required"] is True
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    core = report["python_reference_core"]
    assert core["split_forward_max_abs_error"] <= 1.0e-12
    assert core["merge_roundtrip_max_abs_error"] <= 1.0e-12
    assert core["page_mass_max_abs_error"] <= 1.0e-12
    assert core["old_weights_loaded"] is False
    assert core["training_started"] is False


def test_pg111_trace_and_visible_dataset_are_bounded():
    dataset = _load("pg111_bsp_v3_adapter_smoke_visible_dataset_v1.json")
    trace = _load("pg111_bsp_v3_adapter_smoke_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["evaluation_only"] is True
    assert trace["training_eligible"] is False
    assert len(trace["steps"]) == 1
    step = trace["steps"][0]
    assert step["decision"] == "await_typed_oracle"
    assert step["typed_oracle_called"] is False
    assert step["confirmed_positive"] is False
    assert step["mutates_weights"] is False
    assert step["mutates_topology"] is False
    text = json.dumps({"dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("<script", "union select", "raw_payload_value", "raw_response_body_value"):
        assert forbidden not in text


def test_pg111_source_hashes_match_current_files():
    report = _load("pg111_bsp_v3_adapter_smoke_report_v1.json")
    for key, relative_path in {
        "adapter": "app/bsp_v3_rule_ir_adapter.py",
        "python_core": "app/bsp_v3_research_core.py",
        "runner": "scripts/run_pg111_bsp_v3_adapter_smoke.py",
    }.items():
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == report["source"]["source_hashes"][key]


def test_pg111_rule_policy_preserves_architecture_only_transfer_and_mandarin_isolation():
    rules = _load("improvement_rules.json")
    policy = rules["pg111_bsp_v3_structure_adapter_policy"]
    assert policy["current_project_owns_contract"] is True
    assert policy["dog_project_imported"] is False
    assert policy["previous_checkpoint_reuse_forbidden"] is True
    assert policy["fresh_checkpoint_required"] is True
    assert policy["mandarin_foundation_training_isolated"] is True
    assert policy["immutable_replay_package"] is True
    assert policy["cpu_cuda_scope"].startswith("backend_neutral_structure_signature_only")
    assert policy["numerical_forward_parity_claimed"] is False
    assert policy["training_eligible"] is False
    assert policy["python_reference_core"] is True
    assert policy["python_core_split_forward_invariance"] is True
    assert policy["python_core_merge_roundtrip_invariance"] is True
