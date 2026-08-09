"""PG-96: automatically propose a bounded goal and intermediate labels.

The proposer receives only matched control/candidate observation deltas from
the PG-94 local maze.  Family names, route names, typed oracle results, and
raw probe/response material are removed before proposal.  The proposal is
then evaluated against the hidden typed oracle on a seed holdout.  This is an
evaluation-only experiment; neither the proposal nor its labels can update a
checkpoint or long-term memory.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.auto_goal_label import (  # noqa: E402
    SCHEMA_VERSION,
    apply_proposal,
    make_visible_pair,
    proposal_digest,
    propose_goal_and_labels,
)


TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg96_auto_goal_label_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg96_auto_goal_label_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg96_auto_goal_label_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg96_auto_goal_label_visible_dataset_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg96_auto_goal_label_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg96_auto_goal_label_report_v1.md"
PROTOCOL_ID = "pg-pk-96-auto-goal-label-design-v1"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _layout(step: dict[str, Any]) -> str:
    route = str((step.get("action_manifest") or {}).get("route_template_id", ""))
    match = re.match(r"pg36-(north|south)-", route)
    if not match:
        raise ValueError("PG-96 saw an unexpected route implementation")
    return match.group(1)


def _phase(step: dict[str, Any]) -> str:
    return str((step.get("action_manifest") or {}).get("probe_ref", "")).rsplit("-", 1)[-1]


def _pairs(trace: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(step["step_id"]): step for step in trace.get("steps", [])}
    result: list[dict[str, Any]] = []
    for candidate in trace.get("steps", []):
        step_id = str(candidate.get("step_id", ""))
        if "-candidate" not in step_id:
            continue
        control = by_id.get(step_id.replace("-candidate", "-control", 1))
        if control is None:
            raise ValueError(f"missing matched control for {step_id}")
        visible = make_visible_pair(control, candidate)
        seed = int(candidate.get("sampling_seed", -1))
        method = str(visible["method"])
        phase = str(visible["phase"])
        layout = _layout(candidate)
        # The proposer sees only opaque row/context ids.  Pairing and split
        # metadata remain outside the model-visible feature map.
        row_id = _digest(step_id)[:24]
        context_group = _digest(f"{layout}|{seed}|{method}|{phase}")[:24]
        split = "seed_holdout" if seed == 373 else "design"
        visible_row = {
            "row_id": row_id,
            "context_group": context_group,
            **visible,
        }
        result.append(
            {
                "visible": visible_row,
                "split": split,
                "layout": layout,
                "seed": seed,
                "method": method,
                "phase": phase,
                "family": str(candidate.get("hypothesis", "")),
                "oracle": dict(candidate.get("oracle_projection") or {}),
                "episode_id": str(candidate.get("episode_id", "")),
                "step_id_digest": row_id,
                "fresh_reset": bool((candidate.get("fresh_reset") or {}).get("completed"))
                and bool((candidate.get("fresh_reset") or {}).get("fresh_target"))
                and not bool((candidate.get("fresh_reset") or {}).get("external_network"))
                and str((candidate.get("fresh_reset") or {}).get("transport", "")) == "httpx_loopback",
            }
        )
    return result


def _metric(rows: Iterable[dict[str, Any]], proposal: dict[str, Any]) -> dict[str, Any]:
    rows = list(rows)
    positive = negative = confirm = false_accept = abstain = 0
    per_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for item in rows:
        oracle = item["oracle"]
        expected_positive = bool(oracle.get("positive_authority")) and bool(oracle.get("positive"))
        prediction = apply_proposal(item["visible"], proposal)
        decision = prediction["decision"]
        if expected_positive:
            positive += 1
            per_family[item["family"]][0] += 1
            if decision == "confirm_candidate":
                confirm += 1
                per_family[item["family"]][1] += 1
        else:
            negative += 1
            if decision == "confirm_candidate":
                false_accept += 1
        if decision == "abstain":
            abstain += 1
    family_recalls = {
        family: round(values[1] / values[0], 6) if values[0] else 0.0
        for family, values in sorted(per_family.items())
    }
    unknown = [item for item in rows if item["family"] == "unknown_surface"]
    unknown_abstain = sum(apply_proposal(item["visible"], proposal)["decision"] == "abstain" for item in unknown)
    return {
        "count": len(rows),
        "typed_positive_count": positive,
        "typed_negative_count": negative,
        "confirm_recall": round(confirm / positive, 6) if positive else 0.0,
        "false_accept_count": false_accept,
        "abstain_count": abstain,
        "not_all_abstain": bool(rows) and abstain < len(rows),
        "family_confirm_recall": family_recalls,
        "family_min_confirm_recall": min(family_recalls.values()) if family_recalls else 0.0,
        "unknown_family_count": len(unknown),
        "unknown_family_strict_abstain": bool(unknown) and unknown_abstain == len(unknown),
    }


def _goal_metric(rows: list[dict[str, Any]], proposal: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the proposed two-step goal, still using oracle only outside it."""

    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        episodes[item["episode_id"]].append(item)
    completed_positive = completed_negative = 0
    positive_episode_count = negative_episode_count = 0
    for episode_rows in episodes.values():
        confirm_rows = [item for item in episode_rows if item["phase"] == "confirm"]
        predicted = [apply_proposal(item["visible"], proposal)["decision"] for item in confirm_rows]
        oracle_positive = [
            bool(item["oracle"].get("positive_authority")) and bool(item["oracle"].get("positive"))
            for item in confirm_rows
        ]
        episode_positive = any(oracle_positive)
        if episode_positive:
            positive_episode_count += 1
            # A repeatable goal requires both channel probes to confirm.  This
            # prevents one lucky response from being treated as a goal hit.
            if len(predicted) >= 2 and all(value == "confirm_candidate" for value in predicted) and all(oracle_positive):
                completed_positive += 1
        else:
            negative_episode_count += 1
            if any(value == "confirm_candidate" for value in predicted):
                completed_negative += 1
    return {
        "episode_count": len(episodes),
        "positive_episode_count": positive_episode_count,
        "negative_episode_count": negative_episode_count,
        "repeat_goal_positive_completion_rate": round(completed_positive / positive_episode_count, 6) if positive_episode_count else 0.0,
        "repeat_goal_false_completion_count": completed_negative,
        "max_steps": int((proposal.get("goal") or {}).get("budget", {}).get("max_steps", 0)),
    }


def run() -> dict[str, Any]:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    pairs = _pairs(trace)
    if not pairs:
        raise RuntimeError("PG-96 did not find candidate/control pairs")
    design_visible = [item["visible"] for item in pairs if item["split"] == "design"]
    proposal = propose_goal_and_labels(design_visible)
    proposal["proposal_sha256"] = proposal_digest(proposal)
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Separate layout-blind check: the proposal sees only the north route and
    # is replayed on the south route.  PG36 uses one Python fixture with two
    # layouts, so this is intentionally reported as layout holdout rather
    # than an independent-implementation claim.
    layout_proposal = propose_goal_and_labels([item["visible"] for item in pairs if item["layout"] == "north"])

    metrics = {
        "design": _metric([item for item in pairs if item["split"] == "design"], proposal),
        "seed_holdout": _metric([item for item in pairs if item["split"] == "seed_holdout"], proposal),
    }
    goal = {
        "design": _goal_metric([item for item in pairs if item["split"] == "design"], proposal),
        "seed_holdout": _goal_metric([item for item in pairs if item["split"] == "seed_holdout"], proposal),
    }
    metrics["all"] = _metric(pairs, proposal)
    metrics["layout_holdout"] = _metric([item for item in pairs if item["layout"] == "south"], layout_proposal)
    goal["all"] = _goal_metric(pairs, proposal)
    goal["layout_holdout"] = _goal_metric([item for item in pairs if item["layout"] == "south"], layout_proposal)
    safety_failures = sum(not item["visible"]["safe_probe"] or not item["fresh_reset"] for item in pairs)
    checks = {
        "proposal_did_not_see_oracle": proposal["proposal_inputs"]["oracle_visible"] is False,
        "proposal_did_not_see_family": proposal["proposal_inputs"]["family_visible"] is False,
        "proposal_did_not_see_raw": proposal["proposal_inputs"]["raw_probe_visible"] is False and proposal["proposal_inputs"]["raw_response_visible"] is False,
        "get_post_covered": sorted({item["method"] for item in pairs}) == ["GET", "POST"],
        "fresh_reset_and_safe": safety_failures == 0,
        "seed_holdout_not_all_abstain": metrics["seed_holdout"]["not_all_abstain"],
        "seed_holdout_recall_min": metrics["seed_holdout"]["confirm_recall"] >= 0.80,
        "seed_holdout_false_accept_zero": metrics["seed_holdout"]["false_accept_count"] == 0,
        "layout_holdout_recall_min": metrics["layout_holdout"]["confirm_recall"] >= 0.80,
        "layout_holdout_false_accept_zero": metrics["layout_holdout"]["false_accept_count"] == 0,
        "unknown_family_strict_abstain": metrics["seed_holdout"]["unknown_family_strict_abstain"],
        "repeat_goal_not_all_failed": goal["seed_holdout"]["repeat_goal_positive_completion_rate"] > 0.0,
    }
    blocked = [name for name, passed in checks.items() if not passed]
    status = "passed" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg96-auto-goal-label-report-v1",
        "status": status,
        "source": {
            "trace": str(TRACE_PATH.relative_to(ROOT)),
            "design_split": "seeds 361/367",
            "seed_holdout": [373],
            "device": "cpu_deterministic_inductive_synthesis",
            "oracle_after_proposal": True,
            "training": False,
            "memory_write": False,
        },
        "proposal": {
            "proposal_file": str(PROPOSAL_PATH.relative_to(ROOT)),
            "proposal_sha256": proposal["proposal_sha256"],
            "selected_predicate": proposal["selected_predicate"],
            "layout_blind_selected_predicate": layout_proposal["selected_predicate"],
            "layout_blind_proposal_sha256": proposal_digest(layout_proposal),
            "candidate_label_count": len(proposal["labels"]),
        },
        "metrics": metrics,
        "goal_metrics": goal,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "proposal_quarantined",
            "reason": "automatic goals/labels require independent typed-oracle, cross-source, unknown-family, and human/Codex review gates",
        },
        "safety": {
            "loopback_source_only": True,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "evaluator_labels_in_proposal_input": False,
            "typed_oracle_labels_used_only_for_evaluation": True,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    visible_rows = []
    trace_rows = []
    for item in pairs:
        prediction = apply_proposal(item["visible"], proposal)
        visible = dict(item["visible"])
        visible["split"] = item["split"]
        visible["auto_label"] = prediction["label_id"]
        visible_rows.append(visible)
        trace_rows.append(
            {
                "trace_id": item["step_id_digest"],
                "split": item["split"],
                "method": item["method"],
                "phase": item["phase"],
                "auto_label": prediction["label_id"],
                "decision": prediction["decision"],
                "observed_prefixes": prediction["observed_prefixes"],
                "fresh_reset": item["fresh_reset"],
                "safety": item["visible"]["safe_probe"],
            }
        )
    dataset = {
        "schema_version": "pg96-auto-goal-label-visible-dataset-v1",
        "dataset_id": "pg96-auto-goal-label-visible",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_contract": {
            "oracle_is_label_not_feature": True,
            "family_label_in_features": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "visible_fields": ["method", "encoding_class", "phase", "delta_tokens", "delta_count", "has_observed_change", "safe_probe"],
        },
        "proposal_sha256": proposal["proposal_sha256"],
        "rows": visible_rows,
        "long_term_memory_write": False,
    }
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace_out = {
        "schema_version": "pg96-auto-goal-label-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "steps": trace_rows,
        "long_term_memory_write": False,
    }
    TRACE_OUT_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg96-auto-goal-label-protocol-v1",
        "purpose": "test whether AI-like inductive synthesis can invent useful generic goals/labels from safe observation deltas",
        "input_contract": {
            "source": "pg94_pg36_surface_trace_v1",
            "visible": ["GET/POST", "encoding class", "phase", "bounded response difference tokens", "safety flags"],
            "forbidden": ["family", "hypothesis", "typed oracle", "decision", "belief", "route identity", "raw probe", "raw response", "hash identifiers"],
        },
        "split_contract": {
            "design_seeds": [361, 367],
            "seed_holdout": [373],
            "layout_blind_design": "north",
            "layout_blind_holdout": "south",
            "layout_blind_is_not_independent_implementation": True,
            "fresh_target_required": True,
        },
        "proposal_contract": {
            "goal_must_include": ["success_condition", "failure_condition", "abstain_condition", "safety_budget"],
            "labels_must_be_bounded": True,
            "proposal_is_not_a_vulnerability_finding": True,
        },
        "oracle_contract": {"typed_oracle_after_proposal": True, "negative_controls_required": True, "evidence_hash_required": True},
        "gate": {"minimum_seed_recall": 0.80, "false_accept_count": 0, "unknown_family_strict_abstain": True, "not_all_abstain": True, "promotion_on_pass": False},
        "result": {"status": status, "blocking_reasons": blocked},
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-96 自动目标/标签设计\n\n"
        f"状态：`{status}`；选择的无监督谓词：`{proposal['selected_predicate']['predicate_id']}`。\n\n"
        "提案只看安全观测差分，typed oracle 仅在提案之后用于盲测；不训练、不改 checkpoint、不写长期记忆。\n\n"
        f"seed holdout 召回：`{metrics['seed_holdout']['confirm_recall']}`；误报：`{metrics['seed_holdout']['false_accept_count']}`；未知族严格弃权：`{metrics['seed_holdout']['unknown_family_strict_abstain']}`。\n\n"
        f"阻塞项：{', '.join(blocked) if blocked else '无'}。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "selected_predicate": result["proposal"]["selected_predicate"]["predicate_id"],
        "seed_holdout_recall": result["metrics"]["seed_holdout"]["confirm_recall"],
        "seed_holdout_false_accept": result["metrics"]["seed_holdout"]["false_accept_count"],
        "unknown_family_strict_abstain": result["metrics"]["seed_holdout"]["unknown_family_strict_abstain"],
        "repeat_goal_completion": result["goal_metrics"]["seed_holdout"]["repeat_goal_positive_completion_rate"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
