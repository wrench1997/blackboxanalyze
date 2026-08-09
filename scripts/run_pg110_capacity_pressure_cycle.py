"""PG-110: typed BSP capacity pressure, merge and speed-constrained ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_capacity_controller import evaluate_ablation_gate, plan_capacity_action  # noqa: E402


PROTOCOL_ID = "pg-pk-110-capacity-pressure-cycle-v1"
PG108_REPORT_PATH = ROOT / "research" / "pg108_belief_stress_report_v1.json"
PG109_REPORT_PATH = ROOT / "research" / "pg109_fragment_composition_report_v1.json"
CONTROLLER_PATH = ROOT / "app" / "bsp_capacity_controller.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg110_capacity_pressure_cycle.py"
REPORT_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _base_observation(target_id: str, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "target_id": target_id,
        "unit_kind": "bsp_node",
        "capacity_units": 8,
        "typed_bottleneck": False,
        "fresh_holdout_gap": False,
        "cross_dataset_evidence": True,
        "cross_seed_evidence": True,
        "cross_implementation_evidence": True,
        "known_recall": 1.0,
        "false_accept_count": 0,
        "unknown_abstain_rate": 1.0,
        "all_abstain": False,
        "latency_ms": 100.0,
        "latency_budget_ms": 150.0,
        "memory_ratio": 1.0,
        "redundancy_evidence": False,
        "low_contribution_units": [],
    }
    value.update(overrides)
    return value


def _scenarios() -> list[dict[str, Any]]:
    redundant = [_unit("pg110-redundant-node-0"), _unit("pg110-redundant-node-1")]
    return [
        {
            "scenario_id": "stable_hold",
            "observation": _base_observation("pg109-stable-hold"),
            "expected_action": "hold_capacity",
        },
        {
            "scenario_id": "typed_gap_growth",
            "observation": _base_observation(
                "pg109-typed-gap",
                typed_bottleneck=True,
                fresh_holdout_gap=True,
                known_recall=0.55,
                all_abstain=False,
            ),
            "expected_action": "wake_target_unit",
        },
        {
            "scenario_id": "all_abstain_typed_gap",
            "observation": _base_observation(
                "pg109-all-abstain-gap",
                typed_bottleneck=True,
                fresh_holdout_gap=True,
                known_recall=0.0,
                all_abstain=True,
            ),
            "expected_action": "wake_target_unit",
        },
        {
            "scenario_id": "speed_redundancy",
            "observation": _base_observation(
                "pg109-speed-redundancy",
                latency_ms=240.0,
                latency_budget_ms=150.0,
                redundancy_evidence=True,
                low_contribution_units=redundant,
            ),
            "expected_action": "merge_then_ablate_low_contribution_units",
        },
        {
            "scenario_id": "speed_without_redundancy",
            "observation": _base_observation(
                "pg109-speed-no-proof",
                latency_ms=240.0,
                latency_budget_ms=150.0,
            ),
            "expected_action": "measure_speed_without_ablation",
        },
        {
            "scenario_id": "speed_and_capability_gap",
            "observation": _base_observation(
                "pg109-speed-capability-tradeoff",
                typed_bottleneck=True,
                fresh_holdout_gap=True,
                known_recall=0.70,
                latency_ms=240.0,
                latency_budget_ms=150.0,
            ),
            "expected_action": "hold_and_measure_tradeoff",
        },
        {
            "scenario_id": "false_accept_safety",
            "observation": _base_observation(
                "pg109-false-accept",
                typed_bottleneck=True,
                fresh_holdout_gap=True,
                known_recall=0.80,
                false_accept_count=1,
            ),
            "expected_action": "hold_and_repair_evidence",
        },
        {
            "scenario_id": "missing_cross_evidence",
            "observation": _base_observation(
                "pg109-missing-cross-seed",
                typed_bottleneck=True,
                fresh_holdout_gap=True,
                known_recall=0.60,
                cross_seed_evidence=False,
            ),
            "expected_action": "hold_and_collect_cross_evidence",
        },
    ]


def run() -> dict[str, Any]:
    pg108 = json.loads(PG108_REPORT_PATH.read_text(encoding="utf-8"))
    pg109 = json.loads(PG109_REPORT_PATH.read_text(encoding="utf-8"))
    scenarios = _scenarios()
    decisions: list[dict[str, Any]] = []
    for item in scenarios:
        plan = plan_capacity_action(item["observation"])
        decisions.append({
            "scenario_id": item["scenario_id"],
            "observation": plan.pop("observation", item["observation"]),
            "plan": plan,
            "expected_action": item["expected_action"],
            "action_matches_fixture": plan["action"] == item["expected_action"],
        })

    speed_plan = next(item["plan"] for item in decisions if item["scenario_id"] == "speed_redundancy")
    baseline = {
        "known_recall": 1.0,
        "false_accept_count": 0,
        "unknown_abstain_rate": 1.0,
        "latency_ms": 240.0,
        "memory_ratio": 1.0,
        "fresh_holdout_tested": True,
        "capacity_units": speed_plan["capacity_before"],
    }
    candidate_pass = {
        "known_recall": 1.0,
        "false_accept_count": 0,
        "unknown_abstain_rate": 1.0,
        "latency_ms": 140.0,
        "memory_ratio": 0.92,
        "fresh_holdout_tested": True,
        "capacity_units": speed_plan["capacity_after_proposed"],
    }
    candidate_fail = dict(candidate_pass)
    candidate_fail["known_recall"] = 0.95
    ablation_pass = evaluate_ablation_gate(baseline, candidate_pass, action=speed_plan["action"])
    ablation_fail = evaluate_ablation_gate(baseline, candidate_fail, action=speed_plan["action"])

    checks = {
        "source_pg108_stress_passed": pg108.get("status") == "passed_belief_order_seed_stress" and pg108.get("capability_gate", {}).get("claim_allowed") is False,
        "source_pg109_composition_passed": pg109.get("status") == "passed_fragment_composition_diagnostic" and pg109.get("capability_gate", {}).get("claim_allowed") is False,
        "source_metrics_stable": pg109.get("metrics", {}).get("known_effect_assembly_recall") == 1.0 and pg109.get("metrics", {}).get("unknown_or_decoy_abstain_rate") == 1.0,
        "all_fixture_actions_match": all(item["action_matches_fixture"] for item in decisions),
        "typed_gap_wakes_target": next(item for item in decisions if item["scenario_id"] == "typed_gap_growth")["plan"]["capacity_after_proposed"] == 9,
        "all_abstain_is_not_success": next(item for item in decisions if item["scenario_id"] == "all_abstain_typed_gap")["plan"]["action"] == "wake_target_unit",
        "false_accept_blocks_growth": next(item for item in decisions if item["scenario_id"] == "false_accept_safety")["plan"]["action"] == "hold_and_repair_evidence",
        "missing_cross_evidence_blocks_growth": next(item for item in decisions if item["scenario_id"] == "missing_cross_evidence")["plan"]["action"] == "hold_and_collect_cross_evidence",
        "speed_without_redundancy_does_not_ablate": next(item for item in decisions if item["scenario_id"] == "speed_without_redundancy")["plan"]["action"] == "measure_speed_without_ablation",
        "speed_and_gap_requires_tradeoff": next(item for item in decisions if item["scenario_id"] == "speed_and_capability_gap")["plan"]["action"] == "hold_and_measure_tradeoff",
        "good_ablation_passes": ablation_pass["status"] == "passed_ablation_gate" and ablation_pass["rollback"] is False,
        "bad_ablation_rolls_back": ablation_fail["status"] == "rollback" and ablation_fail["rollback"] is True and "known_recall_regressed" in ablation_fail["reasons"],
        "capacity_plans_non_executable": all(item["plan"]["executable"] is False and item["plan"]["promotion_eligible"] is False for item in decisions),
        "no_training_or_memory_promotion": True,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_capacity_pressure_diagnostic" if not blocked else "blocked"
    metrics = {
        "scenario_count": len(scenarios),
        "growth_action_count": sum(int(item["plan"]["action"] == "wake_target_unit") for item in decisions),
        "merge_ablate_action_count": sum(int(item["plan"]["action"] == "merge_then_ablate_low_contribution_units") for item in decisions),
        "hold_or_measure_action_count": sum(int(item["plan"]["action"] in {"hold_capacity", "hold_and_repair_evidence", "hold_and_collect_cross_evidence", "measure_speed_without_ablation", "hold_and_measure_tradeoff"}) for item in decisions),
        "ablation_pass": ablation_pass["status"] == "passed_ablation_gate",
        "ablation_rollback": ablation_fail["status"] == "rollback",
        "source_known_recall": pg109.get("metrics", {}).get("known_effect_assembly_recall"),
        "source_unknown_abstain_rate": pg109.get("metrics", {}).get("unknown_or_decoy_abstain_rate"),
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg110-capacity-pressure-cycle-report-v1",
        "status": status,
        "source": {
            "pg108_report": "research/pg108_belief_stress_report_v1.json",
            "pg109_report": "research/pg109_fragment_composition_report_v1.json",
            "source_hashes": {
                "pg108_report": _sha256_file(PG108_REPORT_PATH),
                "pg109_report": _sha256_file(PG109_REPORT_PATH),
                "controller": _sha256_file(CONTROLLER_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "controller": {
            "architecture": "typed pressure planner for BSP Page/Node/Expert targets",
            "actions": ["hold_capacity", "wake_target_unit", "merge_then_ablate_low_contribution_units", "measure_speed_without_ablation", "hold_and_measure_tradeoff", "hold_and_repair_evidence", "hold_and_collect_cross_evidence"],
            "architecture_transfer_mode": "bsp_v3_structure_contract_only",
            "previous_checkpoint_reuse_forbidden": True,
            "fresh_checkpoint_required": True,
            "mandarin_foundation_training_isolated": True,
            "neural_weight_mutation_performed": False,
            "full_training_started": False,
            "family_labels_in_controller": False,
            "oracle_labels_in_controller": False,
            "promotion_allowed": False,
        },
        "metrics": metrics,
        "checks": checks,
        "capacity_decisions": decisions,
        "ablation": {"baseline": baseline, "candidate_pass": candidate_pass, "candidate_fail": candidate_fail, "pass_result": ablation_pass, "fail_result": ablation_fail},
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "capacity_policy_evaluation_only", "reason": "a planner decision is not evidence that a neural BSP mutation improved a fresh target"},
        "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_labels_in_model_input": False, "family_labels_in_controller": False, "full_training_started": False, "weight_mutation_performed": False, "architecture_only_transfer": True, "previous_weights_reuse_forbidden": True, "fresh_checkpoint_required": True, "mandarin_foundation_training_isolated": True, "rollback_on_ablation_regression": True, "long_term_memory_write": False},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {
        "schema_version": "pg110-capacity-pressure-cycle-visible-dataset-v1",
        "dataset_id": "pg110-capacity-pressure-cycle-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "decisions": decisions,
        "ablation": report["ablation"],
        "long_term_memory_write": False,
    }
    DATASET_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {
        "schema_version": "pg110-capacity-pressure-cycle-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "steps": [{"scenario_id": item["scenario_id"], "action": item["plan"]["action"], "reason": item["plan"]["reason"], "capacity_before": item["plan"]["capacity_before"], "capacity_after_proposed": item["plan"]["capacity_after_proposed"], "executable": False, "promotion_eligible": False} for item in decisions],
        "evaluator_labels_in_trace": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL_PATH.write_text(json.dumps({"schema_version": "pg110-capacity-pressure-cycle-proposal-v1", "required_evidence": ["typed_bottleneck", "fresh_holdout_gap", "cross_dataset_evidence", "cross_seed_evidence", "cross_implementation_evidence"], "capacity_actions_are_non_executable": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg110-capacity-pressure-cycle-protocol-v1", "purpose": "test typed BSP capacity growth, merge/ablation and fail-closed evidence collection", "source_reports": ["PG108", "PG109"], "growth_gate": {"typed_bottleneck": True, "fresh_holdout_gap": True, "cross_dataset_seed_implementation": True}, "ablation_gate": {"known_recall_no_drop": True, "false_accept_no_increase": True, "unknown_abstain_no_drop": True, "latency_or_memory_improvement": True, "rollback_on_regression": True}, "result": {"status": status, "blocking_reasons": blocked}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(f"# PG-110 BSP capacity pressure cycle\n\n状态：`{status}`；场景：`{metrics['scenario_count']}`；wake：`{metrics['growth_action_count']}`；merge/ablate：`{metrics['merge_ablate_action_count']}`；hold/measure：`{metrics['hold_or_measure_action_count']}`。\n\n合格消融：`{metrics['ablation_pass']}`；回归消融回滚：`{metrics['ablation_rollback']}`。本轮未修改神经权重、未启动完整训练、未写长期记忆。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "scenario_count": result["metrics"]["scenario_count"], "growth_action_count": result["metrics"]["growth_action_count"], "merge_ablate_action_count": result["metrics"]["merge_ablate_action_count"], "ablation_pass": result["metrics"]["ablation_pass"], "ablation_rollback": result["metrics"]["ablation_rollback"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
