from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from scripts.plan_pg369_multitask_moe_candidate import (
    MOE_SOURCE,
    SLOTS,
    _audit_dataset,
    _model_source_audit,
    build_plan,
    derive_multitask_labels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pg369_real_inputs_pass_static_audit_without_running_training() -> None:
    plan = build_plan()
    assert plan["status"] == "ready_static_candidate_design"
    assert plan["execution"] == {
        "plan_only": True,
        "trainer_invoked": False,
        "checkpoint_written": False,
        "gpu_touched": False,
        "docker_started": False,
        "network_used": False,
        "raw_material_loaded": False,
    }
    assert plan["inputs"]["combined"]["train_rows"] == 1475
    assert plan["inputs"]["combined"]["implementation_holdout_rows"] == 1477
    assert plan["inputs"]["combined"]["vocabulary_union_size"] == 875
    assert plan["inputs"]["combined"]["max_context_target_length"] == 621
    assert plan["model"]["config"]["max_length"] == 768
    assert plan["model"]["context_policy"]["all_abstract_context_kept"] is True
    assert plan["model"]["context_policy"]["target_tokens_as_labels_only"] is True
    assert [task["name"] for task in plan["multi_task_objective"]["tasks"]] == [
        "next_token", "slot_query", "ask", "repair", "negative",
    ]
    assert plan["multi_task_objective"]["weights"] == {
        "next_token": 1.0,
        "slot_query": 1.0,
        "ask": 1.5,
        "repair": 1.5,
        "negative": 2.0,
    }
    assert plan["expected_metrics"]["status"] == "not_run_static_plan"
    assert all(value is False for value in plan["promotion"].values())


def test_pg369_plan_contains_no_raw_wire_or_payload_material() -> None:
    plan = build_plan()
    text = json.dumps(plan, ensure_ascii=False, sort_keys=True).casefold()
    for marker in (
        "pg367-runtime-canary",
        "http://",
        "https://",
        "raw_payload=",
        "response_body=",
        "oracle_answer",
        "evaluator_answer",
    ):
        assert marker not in text


def _minimal_dataset() -> dict[str, object]:
    target = ["[TARGET_BOS]"] + [
        f"{key}=" + ("ask_failure" if key == "question" else "failure_feedback" if key == "ask_reason" else "repair" if key == "next_action" else "method" if key == "repair_action" else "get_query" if key == "transport_ref" else "query_term" if key == "field_role_ref" else "identity" if key == "encoding_ref" else "marker" if key == "syntax_category_ref" else "none" if key == "probe_variant_ref" else "0" if key == "safe_to_send" else "query_marker" if key == "payload_shape_ref" else "unknown" if key == "oracle_ref" else "matched_triplet")
        for key in SLOTS
    ] + ["[TARGET_EOS]"]
    return {
        "status": "diagnostic_candidate_only",
        "records": [{
            "record_id": "row-1",
            "split": "train",
            "context_tokens": ["document_presence=observed", "request_method=get"],
            "target_tokens": target,
            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        }, {
            "record_id": "row-2",
            "split": "implementation_holdout",
            "context_tokens": ["document_presence=observed", "request_method=post"],
            "target_tokens": target,
            "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
            "raw_payload_stored": False,
            "raw_response_body_stored": False,
            "oracle_answer_in_context": False,
            "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        }],
        "vocabulary": {"context_tokens": ["document_presence=observed", "request_method=get", "request_method=post"], "target_tokens": target},
    }


def test_pg369_input_audit_is_fail_closed_for_raw_token_and_promotion() -> None:
    dataset = _minimal_dataset()
    clean = _audit_dataset(dataset, source_name="synthetic")
    assert clean["status"] == "passed_candidate_input_audit"
    mutated = copy.deepcopy(dataset)
    mutated["records"][0]["context_tokens"].append("payload=<literal>")
    mutated["records"][1]["promotion"]["memory_promotion_allowed"] = True
    result = _audit_dataset(mutated, source_name="synthetic")
    assert result["status"] == "blocked_input_audit"
    assert "row_promotion_not_closed" in result["failures"]
    assert result["counts"]["raw_hits"] == 1


def test_pg369_derives_abstract_auxiliary_labels_without_wire_material() -> None:
    dataset = _minimal_dataset()
    target = dataset["records"][0]["target_tokens"]
    labels = derive_multitask_labels(target)
    assert labels is not None
    assert set(labels["next_token"]) == set(SLOTS)
    assert labels["ask"] == {"is_ask": True, "question": "ask_failure", "ask_reason": "failure_feedback"}
    assert labels["repair"]["is_repair"] is True
    assert labels["negative"] == {
        "is_negative_or_abstain": True,
        "safe_to_send": False,
        "negative_control_presence_ref": "matched_triplet",
    }
    assert derive_multitask_labels(["[TARGET_BOS]", "payload=literal", "[TARGET_EOS]"]) is None


def test_pg369_reuses_existing_moe_symbols_without_importing_or_running_torch() -> None:
    audit = _model_source_audit(MOE_SOURCE)
    assert audit["status"] == "passed_model_source_contract"
    assert set(audit["reusable_symbols_present"]) == {"CausalMoEConfig", "CausalMoELanguageModel", "train_causal_moe"}
    source = Path("scripts/plan_pg369_multitask_moe_candidate.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names) for node in tree.body)
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "torch" for node in tree.body)
