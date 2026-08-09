"""PG-108: stress generic belief under order, seed, duplicate and budget changes."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg107_multistep_generic_belief as pg107  # noqa: E402
from app.active_goal_label_inducer import ActiveGoalLabelInducer  # noqa: E402
from app.generic_belief_state import GenericBeliefState, likelihood_from_projection, schedule_next_action  # noqa: E402
from app.probe_binding_attestation import CANONICAL_BINDING_SHA256, binding_attestation_valid, add_binding_attestation  # noqa: E402


PROTOCOL_ID = "pg-pk-108-belief-stress-v1"
TRAIN_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG105_DATASET_PATH = ROOT / "research" / "pg105_observable_projection_visible_dataset_v1.json"
PG105_TRACE_PATH = ROOT / "research" / "pg105_observable_projection_trace_v1.json"
PG106_DATASET_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_visible_dataset_v1.json"
PG106_TRACE_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_trace_v1.json"
BELIEF_PATH = ROOT / "app" / "generic_belief_state.py"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg108_belief_stress.py"
REPORT_PATH = ROOT / "research" / "pg108_belief_stress_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg108_belief_stress_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg108_belief_stress_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg108_belief_stress_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg108_belief_stress_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg108_belief_stress_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training() -> list[dict[str, Any]]:
    data = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    rows = [
        {"model_input": add_binding_attestation(row["model_input"])}
        for row in data.get("rows", [])
        if row.get("role") == "train"
    ]
    if len(rows) != 32:
        raise ValueError("PG-108 requires the frozen 32-row training role")
    return rows


def _load_rows() -> list[dict[str, Any]]:
    rows = pg107._load_rows(PG105_DATASET_PATH, PG105_TRACE_PATH, sources={"pg42", "pg35", "pg76", "pg69"})
    rows.extend(pg107._load_rows(PG106_DATASET_PATH, PG106_TRACE_PATH, sources={"pg106"}))
    return rows


def _likelihood_for(row: dict[str, Any], inducer: ActiveGoalLabelInducer) -> tuple[dict[str, Any], dict[str, float]]:
    output = inducer.predict(row["model_input"], guarded=True)
    return output, likelihood_from_projection(output)


def _run_order(rows: list[dict[str, Any]], inducer: ActiveGoalLabelInducer, *, order_name: str, max_steps: int, seed: int | None = None, duplicate_mode: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = list(rows)
    if order_name == "reverse":
        ordered.reverse()
    elif order_name == "seed_shuffle":
        random.Random(int(seed or 0)).shuffle(ordered)
    belief = GenericBeliefState()
    methods: set[str] = set()
    steps: list[dict[str, Any]] = []
    for index, row in enumerate(ordered[:max_steps]):
        method = str(row.get("method", ""))
        methods.add(method)
        output, likelihood = _likelihood_for(row, inducer)
        evidence = str(row["evidence_sha256"])
        belief_step = belief.observe(f"{row['episode_group']}|{method}|{order_name}", likelihood, evidence_hash=evidence)
        action = schedule_next_action(output, observed_methods=methods, max_steps=max_steps, step_count=index + 1)
        steps.append({
            "episode_group": str(row["episode_group"]),
            "scenario": order_name,
            "seed": int(seed) if seed is not None else None,
            "method": method,
            "decision": str(output.get("decision", "abstain")),
            "composition_decision": str(output.get("composition_decision", "abstain")),
            "next_action": action,
            "belief_step": belief_step,
            "evidence_sha256": evidence,
            "fresh_reset": dict(row["fresh_reset"]),
            "negative_control_matched": bool(row["negative_control_matched"]),
            "typed_oracle_called": False,
            "confirmed_positive": False,
        })
        if action in {"await_typed_oracle", "await_typed_oracle_then_abstain", "abstain_invalid_binding", "abstain_budget_exhausted", "abstain_no_repeated_effect"}:
            break
    if duplicate_mode and steps:
        first = steps[0]
        duplicate_likelihood = {"effect": 0.01, "input_only": 0.01, "no_effect": 0.01, "unknown": 0.97} if duplicate_mode == "conflicting" else {"effect": 0.80, "input_only": 0.05, "no_effect": 0.05, "unknown": 0.10}
        duplicate_step = belief.observe(
            f"{first['episode_group']}|{first['method']}|duplicate",
            duplicate_likelihood,
            evidence_hash=str(first["evidence_sha256"]),
        )
        steps.append({
            "episode_group": str(first["episode_group"]),
            "scenario": duplicate_mode,
            "seed": None,
            "method": str(first["method"]),
            "decision": str(first["decision"]),
            "composition_decision": str(first["composition_decision"]),
            "next_action": "duplicate_evidence_no_action",
            "belief_step": duplicate_step,
            "evidence_sha256": str(first["evidence_sha256"]),
            "fresh_reset": dict(first["fresh_reset"]),
            "negative_control_matched": bool(first["negative_control_matched"]),
            "typed_oracle_called": False,
            "confirmed_positive": False,
        })
    episode = {
        "episode_group": str(rows[0]["episode_group"]) if rows else "",
        "source": str(rows[0].get("source", "")) if rows else "",
        "implementation": str(rows[0].get("implementation", "")) if rows else "",
        "scenario": duplicate_mode or order_name,
        "seed": int(seed) if seed is not None else None,
        "step_count": len(steps),
        "methods": sorted(methods),
        "final_action": str(steps[-1]["next_action"]) if steps else "abstain_no_steps",
        "belief": belief.snapshot(),
        "typed_oracle_called": False,
        "confirmed_positive": False,
        "promotion_eligible": False,
    }
    return episode, steps


def run() -> dict[str, Any]:
    train = _load_training()
    rows = _load_rows()
    inducer = ActiveGoalLabelInducer(minimum_support=2, require_get_post=True, require_binding_attestation=True, expected_binding_sha256=CANONICAL_BINDING_SHA256).fit(train)
    proposal = inducer.proposal()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["episode_group"])].append(row)
    canonical: dict[str, dict[str, Any]] = {}
    reverse: dict[str, dict[str, Any]] = {}
    seed_a: dict[str, dict[str, Any]] = {}
    seed_b: dict[str, dict[str, Any]] = {}
    budget_one: dict[str, dict[str, Any]] = {}
    duplicate: dict[str, dict[str, Any]] = {}
    conflicting: dict[str, dict[str, Any]] = {}
    all_steps: list[dict[str, Any]] = []
    scenario_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group, group_rows in groups.items():
        for name, kwargs, target in (
            ("canonical", {"order_name": "canonical", "max_steps": 2}, canonical),
            ("reverse", {"order_name": "reverse", "max_steps": 2}, reverse),
            ("seed_a", {"order_name": "seed_shuffle", "max_steps": 2, "seed": 10801}, seed_a),
            ("seed_b", {"order_name": "seed_shuffle", "max_steps": 2, "seed": 10802}, seed_b),
            ("budget_one", {"order_name": "budget_one", "max_steps": 1}, budget_one),
        ):
            episode, steps = _run_order(group_rows, inducer, **kwargs)
            target[group] = episode
            scenario_steps[name].extend(steps)
        episode, steps = _run_order(group_rows, inducer, order_name="canonical", max_steps=2, duplicate_mode="duplicate")
        duplicate[group] = episode
        scenario_steps["duplicate"].extend(steps)
        episode, steps = _run_order(group_rows, inducer, order_name="canonical", max_steps=2, duplicate_mode="conflicting")
        conflicting[group] = episode
        scenario_steps["conflicting"].extend(steps)
    def signature(item: dict[str, Any]) -> tuple[Any, ...]:
        return (item["final_action"], item["step_count"], tuple(item["methods"]))
    common_groups = set(canonical) & set(reverse) & set(seed_a) & set(seed_b)
    order_invariant = all(signature(canonical[group]) == signature(reverse[group]) for group in common_groups)
    seed_invariant = all(signature(seed_a[group]) == signature(seed_b[group]) for group in common_groups)
    # The duplicate scenarios intentionally retain the two ordinary replay
    # steps in the trace.  Only the synthetic third step carries the repeated
    # evidence hash; counting the first two steps would turn a correct guard
    # into a false 1/3 failure rate.
    duplicate_steps = [step for step in scenario_steps["duplicate"] if step["scenario"] == "duplicate"]
    conflict_steps = [step for step in scenario_steps["conflicting"] if step["scenario"] == "conflicting"]
    checks = {
        "training_row_count": len(train) == 32,
        "evaluation_row_count": len(rows) == 578,
        "binding_valid": all(binding_attestation_valid(row["model_input"], expected_sha256=CANONICAL_BINDING_SHA256) for row in rows),
        "order_invariant": order_invariant,
        "seed_invariant": seed_invariant,
        "duplicate_evidence_rejected": all(step["belief_step"]["duplicate_evidence"] for step in duplicate_steps),
        "conflicting_duplicate_rejected": all(step["belief_step"]["duplicate_evidence"] for step in conflict_steps),
        "duplicate_posterior_unchanged": all(step["belief_step"]["information_gain"] == 0.0 for step in duplicate_steps),
        "conflicting_posterior_unchanged": all(step["belief_step"]["information_gain"] == 0.0 for step in conflict_steps),
        "budget_one_fail_closed": all(episode["final_action"] in {"abstain_budget_exhausted", "abstain_no_repeated_effect"} for episode in budget_one.values()),
        "unknown_and_decoy_never_confirm": all(not episode["confirmed_positive"] for episode in list(canonical.values()) if episode["source"] in {"pg69", "pg106"}),
        "no_typed_oracle_called": all(not episode["typed_oracle_called"] for episode in canonical.values()),
        "no_confirm_without_oracle": all(not episode["confirmed_positive"] for episode in canonical.values()),
        "fresh_reset_and_negative_preserved": all(step["negative_control_matched"] and step["fresh_reset"]["fresh_target"] for step in scenario_steps["canonical"]),
        "posterior_family_free": all(set(episode["belief"]["posterior"]) == {"effect", "input_only", "no_effect", "unknown"} for episode in canonical.values()),
        "promotion_disabled": all(not episode["promotion_eligible"] for episode in canonical.values()),
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_belief_order_seed_stress" if not blocked else "blocked"
    metrics = {
        "episode_count": len(groups),
        "scenario_count": 7,
        "order_invariant": order_invariant,
        "seed_invariant": seed_invariant,
        "duplicate_step_count": len(duplicate_steps),
        "conflicting_duplicate_step_count": len(conflict_steps),
        "duplicate_posterior_unchanged_rate": round(sum(int(step["belief_step"]["information_gain"] == 0.0) for step in duplicate_steps) / len(duplicate_steps), 6) if duplicate_steps else 0.0,
        "conflicting_posterior_unchanged_rate": round(sum(int(step["belief_step"]["information_gain"] == 0.0) for step in conflict_steps) / len(conflict_steps), 6) if conflict_steps else 0.0,
        "budget_one_fail_closed_rate": round(sum(int(episode["final_action"] in {"abstain_budget_exhausted", "abstain_no_repeated_effect"}) for episode in budget_one.values()) / len(budget_one), 6) if budget_one else 0.0,
        "canonical_confirmed_positive_count": sum(int(episode["confirmed_positive"]) for episode in canonical.values()),
        "canonical_typed_oracle_called_count": sum(int(episode["typed_oracle_called"]) for episode in canonical.values()),
        "posterior_states": ["effect", "input_only", "no_effect", "unknown"],
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg108-belief-stress-report-v1",
        "status": status,
        "source": {"training_source": "PG101 train", "evaluation_source": "PG105/PG106 fresh traces", "training_row_count": len(train), "evaluation_row_count": len(rows), "episode_count": len(groups), "source_hashes": {"train": _sha256_file(TRAIN_PATH), "pg105_dataset": _sha256_file(PG105_DATASET_PATH), "pg105_trace": _sha256_file(PG105_TRACE_PATH), "pg106_dataset": _sha256_file(PG106_DATASET_PATH), "pg106_trace": _sha256_file(PG106_TRACE_PATH), "belief": _sha256_file(BELIEF_PATH), "inducer": _sha256_file(INDUCER_PATH), "runner": _sha256_file(RUNNER_PATH)}},
        "controller": {"scenario_set": ["canonical", "reverse", "seed_a", "seed_b", "budget_one", "duplicate", "conflicting"], "posterior_family_free": True, "typed_oracle_called": False, "duplicate_evidence_guard": True, "conflicting_duplicate_fail_closed": True, "promotion_allowed": False},
        "metrics": metrics,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "belief_stress_evaluation_only", "reason": "stress results do not add training memory without cross-dataset and human review"},
        "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_labels_in_model_input": False, "fresh_reset_replayed": True, "negative_controls_preserved": True, "long_term_memory_write": False},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_steps = [step for scenario in scenario_steps.values() for step in scenario]
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg108-belief-stress-visible-dataset-v1", "dataset_id": "pg108-belief-stress-visible", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "scenario_count": 7, "checks": checks, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg108-belief-stress-trace-v1", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "steps": all_steps, "evaluator_labels_in_trace": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg108-belief-stress-protocol-v1", "purpose": "stress generic belief against action order, sampling seed, duplicate/conflicting evidence and budget perturbations", "scenarios": ["canonical", "reverse", "seed_a", "seed_b", "budget_one", "duplicate", "conflicting"], "gate": {"order_invariant": True, "seed_invariant": True, "duplicate_posterior_unchanged": True, "conflicting_duplicate_posterior_unchanged": True, "budget_fail_closed": True, "typed_oracle_called": False, "promotion_on_pass": False}, "result": {"status": status, "blocking_reasons": blocked}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(f"# PG-108 belief stress / belief 压力审计\n\n状态：`{status}`；episode：`{len(groups)}`；顺序不变：`{order_invariant}`；seed 不变：`{seed_invariant}`。\n\n重复证据 posterior 不变率：`{metrics['duplicate_posterior_unchanged_rate']}`；冲突重复证据不变率：`{metrics['conflicting_posterior_unchanged_rate']}`；budget=1 fail-closed：`{metrics['budget_one_fail_closed_rate']}`。\n\n所有场景 typed oracle 调用和确认均为 0；训练及长期记忆关闭。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "episode_count": result["metrics"]["episode_count"], "order_invariant": result["metrics"]["order_invariant"], "seed_invariant": result["metrics"]["seed_invariant"], "duplicate_unchanged_rate": result["metrics"]["duplicate_posterior_unchanged_rate"], "conflicting_duplicate_unchanged_rate": result["metrics"]["conflicting_posterior_unchanged_rate"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
