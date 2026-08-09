"""PG-97: neural oracle-blind automatic goal/label proposal.

This run trains a tiny masked token-presence autoencoder only on visible
PG-94 design rows, clusters its latent states without typed labels, and emits
an abstract goal/label proposal.  The held-out typed oracle is consulted only
after decoding.  A deterministic token-shuffle ablation is included to catch
the common failure mode where a plausible report is produced from an
unvalidated feature association.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.goal_label_decoder import NeuralGoalLabelDecoder, SCHEMA_VERSION  # noqa: E402
from app.auto_goal_label import make_visible_pair  # noqa: E402


TRACE_PATH = ROOT / "research" / "pg94_pg36_surface_trace_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg97_neural_goal_label_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg97_neural_goal_label_protocol_v1.json"
REPORT_PATH = ROOT / "research" / "pg97_neural_goal_label_report_v1.json"
DATASET_PATH = ROOT / "research" / "pg97_neural_goal_label_visible_dataset_v1.json"
TRACE_OUT_PATH = ROOT / "research" / "pg97_neural_goal_label_trace_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg97-neural-auto-goal-label" / "model.pt"
MARKDOWN_PATH = ROOT / "research" / "pg97_neural_goal_label_report_v1.md"
PROTOCOL_ID = "pg-pk-97-neural-auto-goal-label-v1"
SEED = 20260803


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _layout(step: dict[str, Any]) -> str:
    route = str((step.get("action_manifest") or {}).get("route_template_id", ""))
    match = re.match(r"pg36-(north|south)-", route)
    if not match:
        raise ValueError("PG-97 saw an unexpected route implementation")
    return match.group(1)


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
        layout = _layout(candidate)
        seed = int(candidate.get("sampling_seed", -1))
        visible.update({
            "row_id": _digest(step_id)[:24],
            "context_group": _digest(f"{layout}|{seed}|{visible['method']}|{visible['phase']}")[:24],
        })
        reset = candidate.get("fresh_reset") or {}
        control_reset = control.get("fresh_reset") or {}
        fresh = (
            bool(reset.get("completed"))
            and bool(reset.get("fresh_target"))
            and not bool(reset.get("external_network"))
            and str(reset.get("transport", "")) == "httpx_loopback"
            and bool(control_reset.get("completed"))
            and bool(control_reset.get("fresh_target"))
            and not bool(control_reset.get("external_network"))
            and str(control_reset.get("transport", "")) == "httpx_loopback"
        )
        result.append({
            "visible": visible,
            "split": "seed_holdout" if seed == 373 else "design",
            "layout": layout,
            "seed": seed,
            "family": str(candidate.get("hypothesis", "")),
            "phase": str(visible["phase"]),
            "method": str(visible["method"]),
            "oracle": dict(candidate.get("oracle_projection") or {}),
            "episode_id": str(candidate.get("episode_id", "")),
            "fresh_reset": fresh,
        })
    return result


def _metric(items: Iterable[dict[str, Any]], decoder: NeuralGoalLabelDecoder) -> dict[str, Any]:
    rows = list(items)
    positive = negative = confirm = false_accept = abstain = 0
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unknown_count = unknown_abstain = 0
    for item in rows:
        output = decoder.predict(item["visible"])
        decision = output["decision"]
        positive_authority = bool(item["oracle"].get("positive_authority")) and bool(item["oracle"].get("positive"))
        if positive_authority:
            positive += 1
            family_counts[item["family"]][0] += 1
            if decision == "confirm_candidate":
                confirm += 1
                family_counts[item["family"]][1] += 1
        else:
            negative += 1
            if decision == "confirm_candidate":
                false_accept += 1
        if decision == "abstain":
            abstain += 1
        if item["family"] == "unknown_surface":
            unknown_count += 1
            unknown_abstain += int(decision == "abstain")
    family_recall = {
        family: round(values[1] / values[0], 6) if values[0] else 0.0
        for family, values in sorted(family_counts.items())
    }
    return {
        "count": len(rows),
        "typed_positive_count": positive,
        "typed_negative_count": negative,
        "confirm_recall": round(confirm / positive, 6) if positive else 0.0,
        "false_accept_count": false_accept,
        "abstain_count": abstain,
        "not_all_abstain": bool(rows) and abstain < len(rows),
        "family_confirm_recall": family_recall,
        "family_min_confirm_recall": min(family_recall.values()) if family_recall else 0.0,
        "unknown_family_count": unknown_count,
        "unknown_family_strict_abstain": bool(unknown_count) and unknown_abstain == unknown_count,
    }


def _goal_metric(items: Iterable[dict[str, Any]], decoder: NeuralGoalLabelDecoder) -> dict[str, Any]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        episodes[item["episode_id"]].append(item)
    positive_episodes = negative_episodes = completed_positive = completed_negative = 0
    for episode in episodes.values():
        confirm = [item for item in episode if item["phase"] == "confirm"]
        outputs = [decoder.predict(item["visible"])["decision"] for item in confirm]
        oracle_positive = [bool(item["oracle"].get("positive_authority")) and bool(item["oracle"].get("positive")) for item in confirm]
        if any(oracle_positive):
            positive_episodes += 1
            if len(outputs) >= 2 and all(value == "confirm_candidate" for value in outputs) and all(oracle_positive):
                completed_positive += 1
        else:
            negative_episodes += 1
            if any(value == "confirm_candidate" for value in outputs):
                completed_negative += 1
    return {
        "episode_count": len(episodes),
        "positive_episode_count": positive_episodes,
        "negative_episode_count": negative_episodes,
        "repeat_goal_positive_completion_rate": round(completed_positive / positive_episodes, 6) if positive_episodes else 0.0,
        "repeat_goal_false_completion_count": completed_negative,
        "max_steps": 2,
    }


def _fit(rows: list[dict[str, Any]], *, epochs: int = 80) -> NeuralGoalLabelDecoder:
    decoder = NeuralGoalLabelDecoder(seed=SEED, epochs=epochs)
    return decoder.fit(rows)


def _shuffled_design(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    token_lists = [list(row.get("delta_tokens", [])) for row in rows]
    rng.shuffle(token_lists)
    result = []
    for row, tokens in zip(rows, token_lists):
        clone = dict(row)
        clone["delta_tokens"] = tokens
        result.append(clone)
    return result


def _tokenless_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove every observation-delta token for a signal-ablation control."""
    result = []
    for row in rows:
        clone = dict(row)
        clone["delta_tokens"] = []
        clone["delta_count"] = 0
        clone["has_observed_change"] = False
        result.append(clone)
    return result


def _tokenless_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item, visible=visible) for item, visible in zip(items, _tokenless_rows([item["visible"] for item in items]))]


def run() -> dict[str, Any]:
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    pairs = _pairs(trace)
    design = [item for item in pairs if item["split"] == "design"]
    seed_holdout = [item for item in pairs if item["split"] == "seed_holdout"]
    if not design or not seed_holdout:
        raise RuntimeError("PG-97 requires non-empty design and seed holdout")

    decoder = _fit([item["visible"] for item in design])
    proposal = decoder.proposal(design_row_count=len(design))
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.checkpoint(), CHECKPOINT_PATH)

    north_decoder = _fit([item["visible"] for item in pairs if item["layout"] == "north"])
    layout_holdout = [item for item in pairs if item["layout"] == "south"]
    shuffled_decoder = _fit(_shuffled_design([item["visible"] for item in design]))
    tokenless_decoder = _fit(_tokenless_rows([item["visible"] for item in design]))

    metrics = {
        "design": _metric(design, decoder),
        "seed_holdout": _metric(seed_holdout, decoder),
        "layout_holdout": _metric(layout_holdout, north_decoder),
        "shuffled_ablation_seed_holdout": _metric(seed_holdout, shuffled_decoder),
        "tokenless_signal_ablation_seed_holdout": _metric(_tokenless_items(seed_holdout), tokenless_decoder),
    }
    goal_metrics = {
        "design": _goal_metric(design, decoder),
        "seed_holdout": _goal_metric(seed_holdout, decoder),
        "layout_holdout": _goal_metric(layout_holdout, north_decoder),
        "shuffled_ablation_seed_holdout": _goal_metric(seed_holdout, shuffled_decoder),
        "tokenless_signal_ablation_seed_holdout": _goal_metric(_tokenless_items(seed_holdout), tokenless_decoder),
    }
    safety_failures = sum(not item["fresh_reset"] or not item["visible"]["safe_probe"] for item in pairs)
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
        "tokenless_signal_ablation_degrades": metrics["tokenless_signal_ablation_seed_holdout"]["confirm_recall"] < metrics["seed_holdout"]["confirm_recall"] and metrics["tokenless_signal_ablation_seed_holdout"]["abstain_count"] == metrics["tokenless_signal_ablation_seed_holdout"]["count"],
        "repeat_goal_not_all_failed": goal_metrics["seed_holdout"]["repeat_goal_positive_completion_rate"] > 0.0,
    }
    blocked = [name for name, value in checks.items() if not value]
    status = "passed" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg97-neural-auto-goal-label-report-v1",
        "status": status,
        "source": {
            "trace": str(TRACE_PATH.relative_to(ROOT)),
            "design_split": "seeds 361/367",
            "seed_holdout": [373],
            "layout_blind_design": "north",
            "layout_blind_holdout": "south",
            "layout_blind_is_not_independent_implementation": True,
            "device": str(decoder.device),
            "oracle_after_proposal": True,
            "training": "self_supervised_visible_delta_only",
            "memory_write": False,
        },
        "proposal": {
            "proposal_file": str(PROPOSAL_PATH.relative_to(ROOT)),
            "proposal_sha256": proposal["proposal_sha256"],
            "architecture": proposal["model"]["architecture"],
            "vocabulary_size": proposal["model"]["vocabulary_size"],
            "layout_blind_high_effect_cluster": north_decoder.cluster_stats.get("high_effect_cluster"),
        },
        "metrics": metrics,
        "goal_metrics": goal_metrics,
        "ablation": {
            "shuffled_design_input": True,
            "shuffled_decoder_training_oracle_visible": False,
            "row_shuffle_is_diagnostic_only": True,
            "tokenless_signal_ablation": True,
            "tokenless_signal_ablation_oracle_visible": False,
        },
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": blocked, "claim_allowed": False},
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "neural_proposal_quarantined",
            "reason": "a neural proposal is still a hypothesis; typed oracle, unknown-family abstention, cross-source replay, and review are mandatory",
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
        output = decoder.predict(item["visible"])
        row = dict(item["visible"])
        row["split"] = item["split"]
        row["neural_auto_label"] = output["label_id"]
        visible_rows.append(row)
        trace_rows.append({
            "trace_id": _digest(f"{item['episode_id']}|{item['method']}|{item['phase']}")[:24],
            "split": item["split"],
            "method": item["method"],
            "phase": item["phase"],
            "neural_auto_label": output["label_id"],
            "decision": output["decision"],
            "unknown_tokens": output.get("unknown_tokens", []),
            "fresh_reset": item["fresh_reset"],
            "safety": item["visible"]["safe_probe"],
        })
    dataset = {
        "schema_version": "pg97-neural-goal-label-visible-dataset-v1",
        "dataset_id": "pg97-neural-goal-label-visible",
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
    TRACE_OUT_PATH.write_text(json.dumps({
        "schema_version": "pg97-neural-goal-label-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "steps": trace_rows,
        "long_term_memory_write": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg97-neural-goal-label-protocol-v1",
        "purpose": "test a self-supervised neural decoder that invents generic goals/labels without evaluator labels",
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
        "model_contract": {
            "architecture": "token_presence_autoencoder_plus_two_means_kmeans",
            "training_signal": "masked_visible_token_reconstruction_only",
            "typed_oracle_before_decoder_forbidden": True,
            "family_name_before_decoder_forbidden": True,
        },
        "oracle_contract": {"typed_oracle_after_proposal": True, "negative_controls_required": True, "evidence_hash_required": True},
        "ablation_contract": {
            "shuffled_design_input_required": True,
            "row_shuffle_is_diagnostic_only": True,
            "tokenless_signal_ablation_required": True,
            "tokenless_should_degrade_to_abstain": True,
        },
        "gate": {"minimum_seed_recall": 0.80, "false_accept_count": 0, "unknown_family_strict_abstain": True, "not_all_abstain": True, "promotion_on_pass": False},
        "result": {"status": status, "blocking_reasons": blocked},
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-97 神经自动目标/标签解码\n\n"
        f"状态：`{status}`；架构：`{proposal['model']['architecture']}`；设备：`{decoder.device}`。\n\n"
        f"seed holdout 召回：`{metrics['seed_holdout']['confirm_recall']}`；误报：`{metrics['seed_holdout']['false_accept_count']}`；未知族严格弃权：`{metrics['seed_holdout']['unknown_family_strict_abstain']}`。\n\n"
        f"打乱对照召回：`{metrics['shuffled_ablation_seed_holdout']['confirm_recall']}`；阻塞项：{', '.join(blocked) if blocked else '无'}。\n\n"
        "该模型只提出通用的观测目标/标签，不输出漏洞结论，不更新 active checkpoint 或长期记忆。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
        "architecture": result["proposal"]["architecture"],
        "device": result["source"]["device"],
        "seed_holdout_recall": result["metrics"]["seed_holdout"]["confirm_recall"],
        "seed_holdout_false_accept": result["metrics"]["seed_holdout"]["false_accept_count"],
        "layout_holdout_recall": result["metrics"]["layout_holdout"]["confirm_recall"],
        "shuffled_recall": result["metrics"]["shuffled_ablation_seed_holdout"]["confirm_recall"],
        "tokenless_recall": result["metrics"]["tokenless_signal_ablation_seed_holdout"]["confirm_recall"],
        "unknown_family_strict_abstain": result["metrics"]["seed_holdout"]["unknown_family_strict_abstain"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
