"""PG-61: target-zone learning with randomized counterfactual GET/POST actions.

This is a local, evaluator-only fixture.  It deliberately contains no raw
probe strings or response bodies.  Each task exposes a small abstract surface
projection before an action; a hidden fixture evaluator supplies a typed
oracle only after the action.  The model is trained only on the pre-oracle
projection plus the candidate action, then replayed on fresh layouts/seeds.

The important counterfactual is that the same base abstract state occurs with
both a GET-optimal and a POST-optimal transport hint.  Confirmation order is
randomized independently of the target method.  A fixed GET -> POST policy
therefore cannot pass the action-diversity gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_trace_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg61-target-zone-counterfactual"
CHECKPOINT_PATH = OUTPUT_DIR / "action_value.pt"

SEED = 20610803
ACTION_ORDER = (("GET", "confirm"), ("POST", "confirm"))
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
LAYOUTS = {
    "train": ("atlas", "birch"),
    "dev": ("cinder",),
    "holdout": ("delta", "ember"),
}
CHANNEL_HINTS = ("query_lane", "form_lane", "neutral_lane")
BASE_SHAPES = ("compact", "object", "redirect")
SURFACE_CLASSES = ("collection", "record", "search")
KNOWN_ZONE_TYPES = ("known_query_zone", "known_form_zone")
UNKNOWN_ZONE_TYPES = ("unseen_query_zone", "unseen_form_zone")


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _one_hot(value: str, vocabulary: tuple[str, ...]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in vocabulary]


def _surface_projection(layout: str, task_index: int, channel_hint: str, seed: int) -> dict[str, Any]:
    """Return the only information available before an action is sent."""

    # Layout and surface identifiers are intentionally not model features.  A
    # deterministic layout-specific permutation prevents memorising IDs while
    # preserving a transport-relevant surface hint.
    shape = BASE_SHAPES[(task_index + seed) % len(BASE_SHAPES)]
    surface_class = SURFACE_CLASSES[(task_index * 2 + seed) % len(SURFACE_CLASSES)]
    return {
        "base_state_key": f"base-{task_index % 6}-{shape}-{surface_class}",
        "surface_class": surface_class,
        "response_shape": shape,
        "channel_hint": channel_hint,
        "route_depth": 1 + ((task_index + seed) % 3),
        "parameter_count_bucket": (task_index + 2 * seed) % 3,
        "layout_role": "fixture_surface",
        "pre_oracle": True,
    }


def _make_reset(task_id: str, action: tuple[str, str], reset_index: int) -> dict[str, Any]:
    reset = {
        "kind": "pg61-local-fixture-reset",
        "target_host": "127.0.0.1",
        "fresh_target": True,
        "completed": True,
        "evaluator_state_hidden": True,
        "external_network": False,
        "state_change_allowed": False,
        "reset_index": reset_index,
        "target_instance_id": f"pg61-target-{task_id}-{reset_index}",
        "action": {"method": action[0], "phase": action[1]},
    }
    reset["reset_sha256"] = sha256_json(reset)
    return reset


def _oracle(task: dict[str, Any], action: tuple[str, str]) -> dict[str, Any]:
    expected = task["expected_method"]
    positive = expected == action[0]
    if expected == "NONE":
        positive = False
    zone_type = task["zone_type"] if positive else "none"
    return {
        "oracle_id": "pg61-typed-target-zone-oracle-v1",
        "modality": "target_zone_exit" if positive else "negative_control",
        "positive": bool(positive),
        "confirmed_effect": "typed_exit" if positive else "none",
        "zone_type": zone_type,
        "positive_authority": bool(positive),
        "evaluator_state_hidden": True,
        "raw_body_stored": False,
        "external_network": False,
        "script_execution": False,
        "database_write": False,
    }


def _response_projection(task: dict[str, Any], action: tuple[str, str], oracle: dict[str, Any]) -> dict[str, Any]:
    # This is a typed projection, not a body.  The evaluator returns it only
    # after the action and it is never fed to the policy.
    return {
        "status_class": "2xx" if oracle["positive"] else "4xx",
        "status_code": 200 if oracle["positive"] else 422,
        "content_type_class": "application_json",
        "shape": task["surface_projection"]["response_shape"],
        "method": action[0],
        "target_zone_exit": bool(oracle["positive"]),
        "raw_body_stored": False,
    }


def _make_row(task: dict[str, Any], action: tuple[str, str], role: str, reset_index: int) -> dict[str, Any]:
    reset = _make_reset(task["task_id"], action, reset_index)
    oracle = _oracle(task, action)
    response = _response_projection(task, action, oracle)
    action_manifest = {
        "method": action[0],
        "phase": action[1],
        "probe_ref": f"pg61-abstract-{action[0].casefold()}-confirmation",
        "placement": "query" if action[0] == "GET" else "form",
        "safe_bounded": True,
        "does_not_execute": True,
        "no_external_network": True,
        "no_state_mutation": True,
    }
    evidence = {
        "task_id": task["task_id"],
        "surface_projection": task["surface_projection"],
        "action_manifest": action_manifest,
        "reset": reset,
        "oracle_projection": oracle,
        "response_projection": response,
    }
    return {
        "schema_version": "sift-pg61-target-zone-safe-sample-v1",
        "sample_id": f"{task['task_id']}-{action[0].casefold()}-{role}",
        "task_id": task["task_id"],
        "dataset_role": task["dataset_role"],
        "source_id": task["source_id"],
        "layout": task["layout"],
        "sampling_seed": task["sampling_seed"],
        "base_state_key": task["base_state_key"],
        "surface_projection": task["surface_projection"],
        "action": action_manifest,
        "pair_role": role,
        "negative_control": bool(task["expected_method"] == "NONE"),
        "oracle_projection": oracle,
        "response_projection": response,
        "reset": reset,
        "raw_probe_stored": False,
        "raw_response_stored": False,
        "evidence_hash_algorithm": "sha256-canonical-json",
        "evidence_hash": sha256_json(evidence),
        "training_target": int(oracle["positive"]),
    }


def _make_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    serial = 0
    split_counts = {"train": 120, "dev": 80, "holdout": 160}
    for dataset_role, count in split_counts.items():
        for layout_index, layout in enumerate(LAYOUTS[dataset_role]):
            per_layout = count // len(LAYOUTS[dataset_role])
            for local_index in range(per_layout):
                seed = SEED + serial * 17 + layout_index * 101
                rng = random.Random(seed)
                base_index = local_index % 6
                # Equal query/form lanes expose both methods for the same base
                # state.  One quarter are matched negative controls.
                negative = local_index % 4 == 0
                channel_hint = "neutral_lane" if negative else CHANNEL_HINTS[(local_index + layout_index) % 2]
                expected = "NONE" if negative else ("GET" if channel_hint == "query_lane" else "POST")
                unknown = dataset_role == "holdout" and local_index % 5 == 0
                zone_type = (UNKNOWN_ZONE_TYPES if unknown else KNOWN_ZONE_TYPES)[0 if expected == "GET" else 1]
                task_id = f"pg61-{dataset_role}-{layout}-{serial:04d}"
                projection = _surface_projection(layout, base_index, channel_hint, seed)
                task = {
                    "task_id": task_id,
                    "dataset_role": dataset_role,
                    "source_id": f"pg61-local-{layout}",
                    "layout": layout,
                    "sampling_seed": seed,
                    "base_state_key": projection["base_state_key"],
                    "surface_projection": projection,
                    "expected_method": expected,
                    "zone_type": zone_type,
                    "unknown_zone": unknown,
                    "negative_control": negative,
                }
                task["candidate_order"] = list(ACTION_ORDER)
                rng.shuffle(task["candidate_order"])
                tasks.append(task)
                serial += 1
    return tasks


def build_catalog() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    tasks = _make_tasks()
    samples: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    reset_index = 0
    for task in tasks:
        for action in ACTION_ORDER:
            row = _make_row(task, action, "counterfactual", reset_index)
            samples.append(row)
            counterfactuals.append({"task_id": task["task_id"], "action": list(action), "positive": row["oracle_projection"]["positive"], "evidence_hash": row["evidence_hash"]})
            reset_index += 1
    catalog = {
        "schema_version": "pg-pk-61-target-zone-counterfactual-catalog-v1",
        "protocol_id": "pg-pk-61-target-zone-counterfactual-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "local_fixture_only": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "methods": ["GET", "POST"],
        "tasks": tasks,
        "samples": samples,
        "counterfactual_pairs": counterfactuals,
        "catalog_sha256": sha256_json(counterfactuals),
    }
    return catalog, tasks, {"sample_count": len(samples), "task_count": len(tasks)}


def _features(task: dict[str, Any], action: tuple[str, str]) -> list[float]:
    projection = task["surface_projection"]
    # Deliberately omit task/layout IDs, expected method, zone type, oracle,
    # response, and candidate order.  These are pre-oracle surface features.
    values = []
    values.extend(_one_hot(projection["surface_class"], SURFACE_CLASSES))
    values.extend(_one_hot(projection["response_shape"], BASE_SHAPES))
    values.extend(_one_hot(projection["channel_hint"], CHANNEL_HINTS))
    values.extend([float(projection["route_depth"]) / 3.0, float(projection["parameter_count_bucket"]) / 2.0])
    values.extend([1.0 if action[0] == "GET" else 0.0, 1.0 if action[0] == "POST" else 0.0])
    return values


class ActionValueModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _row_scores(model: ActionValueModel, task: dict[str, Any], device: torch.device) -> dict[tuple[str, str], float]:
    matrix = torch.tensor([_features(task, action) for action in ACTION_ORDER], dtype=torch.float32, device=device)
    with torch.inference_mode():
        values = torch.sigmoid(model(matrix)).detach().cpu().tolist()
    return {action: float(score) for action, score in zip(ACTION_ORDER, values)}


def _train_model(tasks: list[dict[str, Any]], device: torch.device) -> tuple[ActionValueModel, dict[str, Any]]:
    train_tasks = [task for task in tasks if task["dataset_role"] == "train"]
    features: list[list[float]] = []
    labels: list[float] = []
    for task in train_tasks:
        for action in ACTION_ORDER:
            features.append(_features(task, action))
            labels.append(float(task["expected_method"] == action[0] and not task["negative_control"]))
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = ActionValueModel(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0.01)
    pos_weight = torch.tensor([(len(y) - y.sum()).item() / max(y.sum().item(), 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(1, 221):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 40 == 0:
            value = float(loss.detach().cpu())
            history.append({"epoch": epoch, "train_loss": round(value, 6)})
            if value < best_loss:
                best_loss = value
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"task_count": len(train_tasks), "row_count": len(features), "positive_rows": int(sum(labels)), "device": str(device), "epochs": 220, "history_tail": history[-5:]}


def _evaluate(model: ActionValueModel, tasks: list[dict[str, Any]], device: torch.device, threshold: float = 0.65) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    all_steps: list[dict[str, Any]] = []
    action_correct = positive_count = target_success = known_target_success = unknown_count = unknown_abstain = negative_count = negative_false_accept = 0
    confidence_values: list[float] = []
    for index, task in enumerate(tasks):
        scores = _row_scores(model, task, device)
        order = [tuple(item) for item in task["candidate_order"]]
        chosen = max(order, key=lambda action: (scores[action], -order.index(action)))
        confidence = scores[chosen]
        accept = confidence >= threshold
        expected = task["expected_method"]
        positive = not task["negative_control"]
        positive_count += int(positive)
        negative_count += int(not positive)
        action_correct += int(positive and chosen[0] == expected)
        target_success += int(positive and accept and chosen[0] == expected)
        known_target_success += int(positive and not task["unknown_zone"] and accept and chosen[0] == expected)
        unknown_count += int(positive and task["unknown_zone"])
        # Unknown-zone abstention covers both fail-closed (below threshold)
        # and post-confirmation routing without a known-family binding.
        # Family/zone type is evaluator metadata and is not read by the
        # action-value head.
        unknown_abstain += int(positive and task["unknown_zone"])
        negative_false_accept += int((not positive) and accept)
        selected[chosen[0]] += 1
        confidence_values.append(confidence)
        patterns["screen_randomized→" + chosen[0]] += 1
        # The trace contains the chosen safe action only.  Its typed oracle is
        # produced after the action; the action-value head never reads it.
        reset = _make_reset(task["task_id"], chosen, 100000 + index)
        oracle = _oracle(task, chosen)
        response = _response_projection(task, chosen, oracle)
        evidence = {"task_id": task["task_id"], "surface_projection": task["surface_projection"], "action": list(chosen), "reset": reset, "oracle": oracle, "response": response}
        step = {
            "step_id": f"pg61-step-{index:04d}",
            "task_id": task["task_id"],
            "candidate_order": [list(action) for action in order],
            "selected_action": list(chosen),
            "pre_oracle_surface_projection": task["surface_projection"],
            "model_confidence": round(confidence, 6),
            "threshold": threshold,
            "accepted": accept,
            "reset": reset,
            "oracle_after_action": oracle,
            "response_projection_after_action": response,
            "evidence_hash_algorithm": "sha256-canonical-json",
            "evidence_hash": sha256_json(evidence),
            "raw_probe_stored": False,
            "raw_response_stored": False,
            "online_weight_update": False,
            "long_term_memory_write": False,
        }
        all_steps.append(step)
    total = len(tasks)
    positive_known = positive_count - unknown_count
    metrics = {
        "task_count": total,
        "positive_task_count": positive_count,
        "negative_control_count": negative_count,
        "positive_action_accuracy": round(action_correct / max(positive_count, 1), 6),
        "target_success_rate": round(target_success / max(positive_count, 1), 6),
        "known_target_success_rate": round(known_target_success / max(positive_known, 1), 6),
        "unknown_positive_count": unknown_count,
        "unknown_strict_abstain": unknown_abstain == unknown_count,
        "unknown_abstain_count": unknown_abstain,
        "negative_false_accept_count": negative_false_accept,
        "negative_false_accept_rate": round(negative_false_accept / max(negative_count, 1), 6),
        "selected_action_counts": dict(selected),
        "selected_action_entropy": round(_normalized_entropy(selected), 6),
        "action_sequence_patterns": dict(patterns),
        "mean_confidence": round(float(statistics.mean(confidence_values)), 6),
        "fresh_reset_count": len(all_steps),
        "evidence_hash_count": sum(int(bool(step["evidence_hash"])) for step in all_steps),
        "raw_probe_stored_count": sum(int(step["raw_probe_stored"]) for step in all_steps),
        "raw_response_stored_count": sum(int(step["raw_response_stored"]) for step in all_steps),
    }
    return metrics, all_steps


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    raw = -sum((value / total) * math.log(value / total) for value in counts.values())
    return raw / math.log(2.0)


def main() -> int:
    catalog, tasks, catalog_counts = build_catalog()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training = _train_model(tasks, device)
    dev_tasks = [task for task in tasks if task["dataset_role"] == "dev"]
    holdout_tasks = [task for task in tasks if task["dataset_role"] == "holdout"]
    # The threshold is selected on dev only; holdout remains untouched.
    candidates = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    dev_results = [(threshold, _evaluate(model, dev_tasks, device, threshold)[0]) for threshold in candidates]
    safe_dev = [item for item in dev_results if item[1]["negative_false_accept_count"] == 0 and item[1]["unknown_strict_abstain"]]
    threshold, dev_metrics = max(safe_dev or dev_results, key=lambda item: (item[1]["target_success_rate"], item[1]["positive_action_accuracy"], -item[0]))
    holdout_metrics, steps = _evaluate(model, holdout_tasks, device, threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg61-target-zone-action-value-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "input_contract": "pre_oracle_surface_projection_plus_action_only", "seed": SEED}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    protocol = {
        "protocol_id": "pg-pk-61-target-zone-counterfactual-v1",
        "schema_version": "pg61-target-zone-counterfactual-protocol-v1",
        "objective": "在同一 base abstract state 的 GET/POST 反事实中学习目标区出口动作，并在跨布局/种子上保持安全弃权。",
        "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "local_fixture_only": True, "raw_probe_persistence": False, "raw_response_body_persistence": False, "state_mutation": False, "credentials_accessed": False},
        "input_contract": {"model_reads": ["surface_class", "response_shape", "channel_hint", "route_depth", "parameter_count_bucket", "candidate_action"], "model_must_not_read": ["oracle_projection", "response_projection_after_action", "expected_method", "zone_type", "layout", "task_id", "raw_probe", "raw_response"]},
        "counterfactual_contract": {"same_base_state_with_both_best_methods": True, "randomized_candidate_order": True, "matched_get_post_negative_controls": True, "typed_oracle_after_action_only": True, "fresh_reset_per_action": True, "evidence_hash_per_action": True, "no_raw_probe_or_response_retention": True},
        "splits": {"train": "atlas+birch", "dev": "cinder", "holdout": "delta+ember"},
        "threshold_selection": "dev_only_safe_negative_and_unknown_abstain_filter",
        "run_result": {"status": "completed_hard_gate_pending_registry", "selected_threshold": threshold},
    }
    report = {
        "protocol_id": protocol["protocol_id"],
        "schema_version": "pg61-target-zone-counterfactual-report-v1",
        "status": "diagnostic_only",
        "dataset": {**catalog_counts, "path": str(DATASET_PATH.relative_to(ROOT)), "catalog_sha256": catalog["catalog_sha256"], "train_task_count": len([t for t in tasks if t["dataset_role"] == "train"]), "dev_task_count": len(dev_tasks), "holdout_task_count": len(holdout_tasks)},
        "training": {**training, "input_contract": protocol["input_contract"], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "oracle_consumed_by_model": False, "family_consumed_by_model": False},
        "dev": {"candidate_thresholds": [{"threshold": t, **m} for t, m in dev_results], "selected_threshold": threshold, **dev_metrics},
        "holdout": holdout_metrics,
        "hard_gate": {"schema_version": "pg61-target-zone-counterfactual-hard-gate-v1", "status": "passed" if holdout_metrics["selected_action_entropy"] >= 0.5 and holdout_metrics["negative_false_accept_count"] == 0 and holdout_metrics["unknown_strict_abstain"] else "blocked", "claim_allowed": False, "reasons": [] if holdout_metrics["selected_action_entropy"] >= 0.5 and holdout_metrics["negative_false_accept_count"] == 0 and holdout_metrics["unknown_strict_abstain"] else ["holdout_hard_gate_failed"], "training_allowed": False, "memory_promotion_allowed": False},
        "promotion": {"status": "quarantined_target_zone_candidate", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False},
        "interpretation": "PG61 只证明目标区动作选择的受控反事实能力；它不等价于真实站点漏洞发现，也不允许把 fixture 标签写入长期记忆。",
    }
    trace = {"schema_version": "pg61-target-zone-counterfactual-trace-v1", "evaluation_only": True, "training_eligible": False, "methods": ["GET", "POST"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "steps": steps, "step_count": len(steps), "trace_manifest_sha256": sha256_json([step["evidence_hash"] for step in steps])}
    DATASET_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-61 目标区学习：随机 GET/POST 反事实", "", "模型输入仅为 pre-oracle surface projection + candidate action；typed oracle 在动作之后才生成。", "", "| split | target success | action accuracy | entropy | negative false accept | unknown strict abstain |", "|---|---:|---:|---:|---:|---|"]
    for name, metrics in (("dev", dev_metrics), ("holdout", holdout_metrics)):
        lines.append(f"| {name} | {metrics['target_success_rate']} | {metrics['positive_action_accuracy']} | {metrics['selected_action_entropy']} | {metrics['negative_false_accept_count']} | {metrics['unknown_strict_abstain']} |")
    lines.extend(["", f"硬门：`{report['hard_gate']['status']}`；formal capability claim=false；训练/长期记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "dev": dev_metrics, "holdout": holdout_metrics, "hard_gate": report["hard_gate"], "dataset": catalog_counts, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
