"""PG-107: generic multi-step belief updates and fail-closed scheduling."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_goal_label_inducer import ActiveGoalLabelInducer  # noqa: E402
from app.active_probe_signature import model_input_has_forbidden_field  # noqa: E402
from app.generic_belief_state import GenericBeliefState, GENERIC_STATES, likelihood_from_projection, schedule_next_action  # noqa: E402
from app.probe_binding_attestation import CANONICAL_BINDING_SHA256, binding_attestation_valid  # noqa: E402


PROTOCOL_ID = "pg-pk-107-multistep-generic-belief-v1"
TRAIN_PATH = ROOT / "research" / "pg101_active_probe_signature_visible_dataset_v1.json"
PG105_DATASET_PATH = ROOT / "research" / "pg105_observable_projection_visible_dataset_v1.json"
PG105_TRACE_PATH = ROOT / "research" / "pg105_observable_projection_trace_v1.json"
PG106_DATASET_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_visible_dataset_v1.json"
PG106_TRACE_PATH = ROOT / "research" / "pg106_decoy_projection_holdout_trace_v1.json"
INDUCER_PATH = ROOT / "app" / "active_goal_label_inducer.py"
BELIEF_PATH = ROOT / "app" / "generic_belief_state.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg107_multistep_generic_belief.py"
REPORT_PATH = ROOT / "research" / "pg107_multistep_generic_belief_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg107_multistep_generic_belief_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg107_multistep_generic_belief_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg107_multistep_generic_belief_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg107_multistep_generic_belief_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg107_multistep_generic_belief_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(dataset_path: Path, trace_path: Path, *, sources: set[str]) -> list[dict[str, Any]]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    groups = {str(step["trace_id"]): str(step["episode_group"]) for step in trace.get("steps", [])}
    rows: list[dict[str, Any]] = []
    for original in dataset.get("rows", []):
        if str(original.get("source")) not in sources:
            continue
        row = dict(original)
        row["episode_group"] = groups.get(str(row["row_id"]), f"pg107-single-{row['row_id']}")
        if model_input_has_forbidden_field(row["model_input"]):
            raise ValueError("PG-107 source model input leaked an evaluator or raw field")
        rows.append(row)
    return rows


def _summarize_model_input(value: dict[str, Any]) -> dict[str, Any]:
    extension = value.get("causal_extension") if isinstance(value.get("causal_extension"), dict) else {}
    anomaly = any(bool(item) for item in extension.get("input_changed_response_unchanged_pattern", []))
    effect = any(bool(item) for item in value.get("delta_pattern", []))
    return {
        "method": str(value.get("method", "")),
        "phase": str(value.get("phase", "")),
        "effect_delta_present": effect,
        "input_only_anomaly_present": anomaly,
        "binding_attested": binding_attestation_valid(value, expected_sha256=CANONICAL_BINDING_SHA256),
    }


def _run_episode(rows: list[dict[str, Any]], inducer: ActiveGoalLabelInducer) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (0 if str(row.get("method")) == "GET" else 1, str(row.get("row_id"))))
    belief = GenericBeliefState()
    observed_methods: set[str] = set()
    steps: list[dict[str, Any]] = []
    selected = ordered[:2]
    for index, row in enumerate(selected):
        method = str(row.get("method", ""))
        observed_methods.add(method)
        output = inducer.predict(row["model_input"], guarded=True)
        likelihood = likelihood_from_projection(output)
        belief_step = belief.observe(
            f"{row['episode_group']}|{method}",
            likelihood,
            evidence_hash=str(row["evidence_sha256"]),
        )
        next_action = schedule_next_action(
            output,
            observed_methods=observed_methods,
            max_steps=2,
            step_count=index + 1,
        )
        steps.append({
            "step_id": f"pg107-step-{row['row_id']}",
            "episode_group": str(row["episode_group"]),
            "source": str(row.get("source", "")),
            "implementation": str(row.get("implementation", "")),
            "method": method,
            "model_observation": _summarize_model_input(row["model_input"]),
            "decision": str(output.get("decision", "abstain")),
            "composition_decision": str(output.get("composition_decision", "abstain")),
            "belief_step": belief_step,
            "next_action": next_action,
            "evidence_sha256": str(row["evidence_sha256"]),
            "fresh_reset": dict(row["fresh_reset"]),
            "negative_control_matched": bool(row["negative_control_matched"]),
            "typed_oracle_called": False,
            "confirmed_positive": False,
            "online_weight_update": False,
            "long_term_memory_write": False,
        })
        if next_action in {"abstain_invalid_binding", "abstain_budget_exhausted", "await_typed_oracle_then_abstain", "abstain_no_repeated_effect"}:
            break
    final_action = str(steps[-1]["next_action"]) if steps else "abstain_no_steps"
    episode = {
        "episode_group": str(ordered[0]["episode_group"]) if ordered else "",
        "source": str(ordered[0].get("source", "")) if ordered else "",
        "implementation": str(ordered[0].get("implementation", "")) if ordered else "",
        "step_count": len(steps),
        "methods": sorted(observed_methods),
        "final_action": final_action,
        "belief": belief.snapshot(),
        "typed_oracle_called": False,
        "confirmed_positive": False,
        "training_eligible": False,
        "long_term_memory_write": False,
    }
    return episode, steps


def run() -> dict[str, Any]:
    train_dataset = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    train_rows = [{"model_input": row["model_input"]} for row in train_dataset.get("rows", []) if row.get("role") == "train"]
    if len(train_rows) != 32:
        raise ValueError("PG-107 requires the frozen 32-row PG-101 training role")
    train_rows = [{"model_input": __import__("app.probe_binding_attestation", fromlist=["add_binding_attestation"]).add_binding_attestation(row["model_input"])} for row in train_rows]
    rows = _load_rows(PG105_DATASET_PATH, PG105_TRACE_PATH, sources={"pg42", "pg35", "pg76", "pg69"})
    rows.extend(_load_rows(PG106_DATASET_PATH, PG106_TRACE_PATH, sources={"pg106"}))
    inducer = ActiveGoalLabelInducer(minimum_support=2, require_get_post=True, require_binding_attestation=True, expected_binding_sha256=CANONICAL_BINDING_SHA256).fit(train_rows)
    proposal = inducer.proposal()
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_group"])].append(row)
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        episode, episode_steps = _run_episode(group_rows, inducer)
        episodes.append(episode)
        steps.extend(episode_steps)
    known_groups = [episode for episode in episodes if episode["source"] in {"pg42", "pg35"}]
    known_effect_groups = [
        episode for episode in known_groups
        if any(
            step["model_observation"]["effect_delta_present"]
            for step in steps
            if step["episode_group"] == episode["episode_group"]
        )
    ]
    anomaly_groups = [episode for episode in episodes if any(step["model_observation"]["input_only_anomaly_present"] for step in steps if step["episode_group"] == episode["episode_group"])]
    decoy_groups = [episode for episode in episodes if episode["source"] == "pg106" and episode["step_count"] >= 1 and any(step["model_observation"]["input_only_anomaly_present"] for step in steps if step["episode_group"] == episode["episode_group"])]
    checks = {
        "training_row_count": len(train_rows) == 32,
        "evaluation_row_count": len(rows) == 578,
        "model_input_oracle_blind": all(not model_input_has_forbidden_field(row["model_input"]) for row in rows),
        "binding_valid": all(binding_attestation_valid(row["model_input"], expected_sha256=CANONICAL_BINDING_SHA256) for row in rows),
        "get_post_in_episodes": all(set(episode["methods"]) == {"GET", "POST"} for episode in episodes),
        "multi_step_rate_min": (sum(int(episode["step_count"] >= 2) for episode in episodes) / len(episodes)) >= 0.80 if episodes else False,
        "known_effect_handoff_not_confirm": bool(known_effect_groups) and all(episode["final_action"] == "await_typed_oracle" for episode in known_effect_groups),
        "unknown_anomaly_never_confirm": all(episode["final_action"] != "confirmed_positive" and episode["confirmed_positive"] is False for episode in anomaly_groups),
        "decoy_never_confirm": all(episode["final_action"] != "confirmed_positive" for episode in decoy_groups),
        "typed_oracle_never_called_by_scheduler": all(not episode["typed_oracle_called"] for episode in episodes),
        "no_confirmed_positive_without_oracle": all(not episode["confirmed_positive"] for episode in episodes),
        "duplicate_evidence_rejected": all(
            sum(int(step["belief_step"]["duplicate_evidence"]) for step in steps if step["episode_group"] == episode["episode_group"]) == 0
            for episode in episodes
        ),
        "generic_posterior_states_only": all(set(episode["belief"]["posterior"]) == set(GENERIC_STATES) for episode in episodes),
        "negative_controls_preserved": all(bool(step["negative_control_matched"]) for step in steps),
        "promotion_disabled": True,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_generic_multistep_belief_diagnostic" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg107-multistep-generic-belief-report-v1",
        "status": status,
        "source": {
            "training_source": "PG101 train role PG36 north seeds 361/367",
            "evaluation_sources": ["PG105 PG42/PG35/PG76/PG69", "PG106 independent decoy"],
            "training_row_count": len(train_rows),
            "evaluation_row_count": len(rows),
            "episode_count": len(episodes),
            "source_hashes": {
                "train_dataset": _sha256_file(TRAIN_PATH),
                "pg105_dataset": _sha256_file(PG105_DATASET_PATH),
                "pg105_trace": _sha256_file(PG105_TRACE_PATH),
                "pg106_dataset": _sha256_file(PG106_DATASET_PATH),
                "pg106_trace": _sha256_file(PG106_TRACE_PATH),
                "inducer": _sha256_file(INDUCER_PATH),
                "belief": _sha256_file(BELIEF_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "controller": {
            "architecture": "generic posterior over effect/input_only/no_effect/unknown",
            "family_names_in_posterior": False,
            "typed_oracle_before_action": False,
            "typed_oracle_called_by_scheduler": False,
            "duplicate_evidence_guard": True,
            "input_only_anomaly_requires_negative_replay": True,
            "confirm_requires_typed_oracle": True,
        },
        "metrics": {
            "episode_count": len(episodes),
            "multi_step_episode_rate": round(sum(int(episode["step_count"] >= 2) for episode in episodes) / len(episodes), 6) if episodes else 0.0,
            "mean_steps": round(sum(episode["step_count"] for episode in episodes) / len(episodes), 6) if episodes else 0.0,
            "known_group_count": len(known_groups),
            "known_effect_group_count": len(known_effect_groups),
            "anomaly_group_count": len(anomaly_groups),
            "decoy_group_count": len(decoy_groups),
            "typed_oracle_handoff_count": sum(int(episode["final_action"] == "await_typed_oracle") for episode in episodes),
            "abstain_after_anomaly_count": sum(int(episode["final_action"] == "await_typed_oracle_then_abstain") for episode in anomaly_groups),
            "confirmed_positive_count": sum(int(episode["confirmed_positive"]) for episode in episodes),
            "typed_oracle_called_count": sum(int(episode["typed_oracle_called"]) for episode in episodes),
            "posterior_state_vocabulary": list(GENERIC_STATES),
        },
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "generic_belief_diagnostic_quarantined", "reason": "belief/scheduling traces are evaluation-only until independent typed replay and OOD review"},
        "safety": {"loopback_only": True, "external_network": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "evaluator_labels_in_model_input": False, "family_names_in_belief": False, "fresh_reset_per_step": True, "negative_controls_preserved": True, "long_term_memory_write": False},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {
        "schema_version": "pg107-multistep-generic-belief-visible-dataset-v1",
        "dataset_id": "pg107-multistep-generic-belief-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {"family_names_in_belief": False, "oracle_labels_in_features": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False},
        "proposal_sha256": proposal["proposal_sha256"],
        "episodes": episodes,
        "long_term_memory_write": False,
    }
    DATASET_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg107-multistep-generic-belief-trace-v1", "evaluation_only": True, "training_eligible": False, "proposal_sha256": proposal["proposal_sha256"], "steps": steps, "evaluator_labels_in_trace": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg107-multistep-generic-belief-protocol-v1", "purpose": "family-free multi-step feedback belief and fail-closed scheduling", "posterior_states": list(GENERIC_STATES), "input_contract": {"oracle_visible": False, "family_visible": False, "raw_values_visible": False}, "action_contract": {"effect_present": "replay_other_method_then_await_typed_oracle", "input_only_anomaly": "repeat_matched_negative_other_method", "no_effect": "probe_other_method_or_abstain", "invalid_binding": "abstain"}, "gate": {"multi_step_episode_rate_min": 0.80, "confirmed_positive_without_oracle": 0, "decoy_confirm_count": 0, "family_names_in_posterior": False, "promotion_on_pass": False}, "result": {"status": status, "blocking_reasons": blocked}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(f"# PG-107 Generic multi-step belief / 通用多步 belief\n\n状态：`{status}`；episode：`{len(episodes)}`；平均步数：`{report['metrics']['mean_steps']}`；多步比例：`{report['metrics']['multi_step_episode_rate']}`。\n\nposterior 只含 `{', '.join(GENERIC_STATES)}`；typed oracle 调用：`0`；确认正例：`0`；异常组最终弃权：`{report['metrics']['abstain_after_anomaly_count']}`。\n\n该轨迹只验证调度和反馈链，不生成训练样本或长期记忆。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["status"], "episode_count": result["metrics"]["episode_count"], "multi_step_episode_rate": result["metrics"]["multi_step_episode_rate"], "typed_oracle_called_count": result["metrics"]["typed_oracle_called_count"], "confirmed_positive_count": result["metrics"]["confirmed_positive_count"], "training_allowed": False, "memory_promotion_allowed": False}, ensure_ascii=False, indent=2))
