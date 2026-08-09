"""Run PG-36's maze-style active probing and belief update diagnostic.

No network call is made here: the model replays the already captured
projection-only Catalog.  The source-transfer checkpoint ranks actions from
the control projection; typed oracle feedback is consulted only after the
candidate step to decide whether the episode may stop.  This is an upper-bound
controller diagnostic, never a training or memory-promotion event.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "research" / "pg36_independent_maze_catalog_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg35-source-transfer" / "pair_rule_ir.pt"
TRAINER_PATH = ROOT / "scripts" / "train_pg35_pair_rule_ir.py"
REPORT_PATH = ROOT / "research" / "pg36_active_belief_diagnostic_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg36_active_belief_diagnostic_v1.md"
SEED = 20360802


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("pg35_pair_trainer_for_pg36", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-transfer model definition")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalise(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 1e-8) for key, value in values.items()}
    total = sum(clipped.values()) or 1.0
    return {key: value / total for key, value in clipped.items()}


def _entropy(values: dict[str, float]) -> float:
    return -sum(value * math.log(value) for value in values.values() if value > 0)


class Belief:
    def __init__(self, families: tuple[str, ...]) -> None:
        self.families = families
        self.posterior = {family: 1.0 / len(families) for family in families}
        self.steps: list[dict[str, Any]] = []
        self.seen: set[str] = set()

    def observe(self, action: str, likelihood: dict[str, float], evidence_sha256: str) -> dict[str, Any]:
        before = dict(self.posterior)
        duplicate = evidence_sha256 in self.seen
        if duplicate:
            after = before
            gain = 0.0
        else:
            normalized = _normalise({family: likelihood.get(family, 0.0) for family in self.families})
            softened = _normalise({family: normalized[family] ** 0.35 for family in self.families})
            after = _normalise({family: 0.8 * before[family] + 0.2 * softened[family] for family in self.families})
            gain = max(0.0, _entropy(before) - _entropy(after))
            self.posterior = after
            self.seen.add(evidence_sha256)
        step = {
            "step": len(self.steps) + 1,
            "action": action,
            "prior": before,
            "likelihood": _normalise(likelihood),
            "posterior": dict(after),
            "entropy_before": round(_entropy(before), 6),
            "entropy_after": round(_entropy(after), 6),
            "information_gain": round(gain, 6),
            "evidence_sha256": evidence_sha256,
            "duplicate_evidence": duplicate,
            "accepted": not duplicate,
        }
        self.steps.append(step)
        return step

    def snapshot(self) -> dict[str, Any]:
        return {"posterior": dict(self.posterior), "entropy": round(_entropy(self.posterior), 6), "unique_evidence_count": len(self.seen), "steps": list(self.steps)}


def _model_scores(module: Any, feature_module: Any, model: Any, mean: torch.Tensor, std: torch.Tensor, rows: list[dict[str, Any]], device: torch.device) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    with torch.inference_mode():
        features = (module._visible_features(feature_module, rows) - mean) / std
        family_logits, effect_logits = model(features.to(device))
        probabilities = torch.softmax(family_logits, dim=-1).cpu()
        effects = torch.sigmoid(effect_logits).cpu()
    scores: dict[str, dict[str, Any]] = {}
    for row, probability, effect in zip(rows, probabilities, effects):
        distribution = {family: float(value) for family, value in zip(module.FAMILIES, probability)}
        distribution["unknown_surface"] = 0.05
        distribution = _normalise(distribution)
        scores[row["sample_id"]] = {"probabilities": distribution, "effect_probability": round(float(effect), 6), "confidence": round(float(probability.max()), 6)}
    return scores


def _action_order() -> list[tuple[str, str]]:
    return [(method, phase) for method in ("GET", "POST") for phase in ("screen", "confirm", "error", "timeout")]


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[tuple[str, str], dict[str, dict[str, Any]]]]:
    grouped: dict[tuple[str, str, int], dict[tuple[str, str], dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (str(row["implementation"]), str(row["surface_id"]), int(row["sampling_seed"]))
        grouped[key][(str(row["method"]), str(row["phase"]))][str(row["pair_role"])] = row
    return grouped


def _fixed_episode(actions: dict[tuple[str, str], dict[str, dict[str, Any]]], order: list[tuple[str, str]]) -> dict[str, Any]:
    positives: list[dict[str, Any]] = []
    for method, phase in order:
        pair = actions[(method, phase)]
        candidate = pair["candidate"]
        if candidate["oracle_projection"]["positive"] and candidate.get("negative_control"):
            positives.append(candidate)
    return {"query_count": len(order) * 2, "accepted_positive": bool(positives), "false_positive": False, "abstained": not bool(positives), "action_order": [[method, phase] for method, phase in order]}


def _active_episode(module: Any, actions: dict[tuple[str, str], dict[str, dict[str, Any]]], score_by_id: dict[str, dict[str, Any]], families: tuple[str, ...]) -> dict[str, Any]:
    belief = Belief(families)
    remaining = set(actions)
    completed_screen: set[str] = set()
    selected: list[tuple[str, str]] = []
    trace: list[dict[str, Any]] = []
    while remaining:
        ranked: list[tuple[float, tuple[str, str]]] = []
        for method, phase in sorted(remaining):
            pair = actions[(method, phase)]
            control = pair["control"]
            scores = score_by_id[control["sample_id"]]
            entropy = _entropy(scores["probabilities"])
            stage_bonus = 0.0
            if phase == "screen" and method not in completed_screen:
                stage_bonus = 0.55
            elif phase == "confirm" and method in completed_screen:
                stage_bonus = 0.75
            elif phase == "confirm":
                stage_bonus = 0.20
            elif phase == "error":
                stage_bonus = 0.04
            else:
                stage_bonus = 0.02
            score = 0.65 * entropy + 0.25 * stage_bonus + 0.10 * (1.0 - scores["effect_probability"])
            ranked.append((score, (method, phase)))
        _, action = max(ranked, key=lambda item: (item[0], tuple(item[1])))
        method, phase = action
        pair = actions[action]
        selected.append(action)
        control = pair["control"]
        candidate = pair["candidate"]
        control_scores = score_by_id[control["sample_id"]]
        candidate_scores = score_by_id[candidate["sample_id"]]
        belief.observe(f"{method}:{phase}:control", control_scores["probabilities"], control["evidence"]["evidence_hash"])
        belief.observe(f"{method}:{phase}:candidate", candidate_scores["probabilities"], candidate["evidence"]["evidence_hash"])
        trace.append({"method": method, "phase": phase, "control_sample_id": control["sample_id"], "candidate_sample_id": candidate["sample_id"], "typed_oracle_after_step": True, "candidate_positive": bool(candidate["oracle_projection"]["positive"]), "belief": belief.snapshot()})
        remaining.remove(action)
        if phase == "screen":
            completed_screen.add(method)
        if candidate["oracle_projection"]["positive"] and candidate.get("negative_control"):
            return {"query_count": len(selected) * 2, "accepted_positive": True, "false_positive": False, "abstained": False, "action_order": [[item[0], item[1]] for item in selected], "stop_reason": "typed_positive_after_confirm", "trace": trace, "belief": belief.snapshot()}
    return {"query_count": len(selected) * 2, "accepted_positive": False, "false_positive": False, "abstained": True, "action_order": [[item[0], item[1]] for item in selected], "stop_reason": "no_typed_positive_within_budget", "trace": trace, "belief": belief.snapshot()}


def _aggregate(results: list[dict[str, Any]], positive_count: int) -> dict[str, Any]:
    accepted = sum(int(item["accepted_positive"]) for item in results)
    false_positive = sum(int(item["false_positive"]) for item in results)
    return {"episode_count": len(results), "positive_episode_count": positive_count, "accepted_positive_count": accepted, "typed_recall": round(accepted / max(positive_count, 1), 6), "false_positive_count": false_positive, "false_positive_rate": round(false_positive / max(len(results) - positive_count, 1), 6), "abstain_count": sum(int(item["abstained"]) for item in results), "median_queries": float(median(item["query_count"] for item in results)), "mean_queries": round(sum(item["query_count"] for item in results) / max(len(results), 1), 6)}


def main() -> int:
    module = _load_module()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.PairRuleIRModel().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    feature_module = module._load_features()
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32)
    rows = list(catalog["samples"])
    scores = _model_scores(module, feature_module, model, mean, std, rows, device)
    grouped = _group(rows)
    order = _action_order()
    fixed_results: list[dict[str, Any]] = []
    active_results: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    positive_count = 0
    families = tuple(module.FAMILIES) + ("unknown_surface",)
    for key, actions in sorted(grouped.items()):
        known_positive = any(bool(pair["candidate"]["oracle_projection"]["positive"]) for pair in actions.values())
        positive_count += int(known_positive)
        fixed = _fixed_episode(actions, order)
        active = _active_episode(module, actions, scores, families)
        fixed_results.append(fixed)
        active_results.append(active)
        episodes.append({"implementation": key[0], "surface_id": key[1], "sampling_seed": key[2], "fixed": fixed, "active": active})
    fixed_metrics = _aggregate(fixed_results, positive_count)
    active_metrics = _aggregate(active_results, positive_count)
    report = {
        "protocol_id": "sift-pg36-active-belief-diagnostic-v1",
        "schema_version": "pg-pk-36-active-belief-diagnostic-v1",
        "status": "diagnostic_only",
        "source": {"catalog": str(CATALOG_PATH.relative_to(ROOT)), "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(), "device": str(device), "typed_oracle_visible_before_probe": False, "model_input_raw_strings": False},
        "controller": {"class": "source_transfer_model_plus_multistep_belief", "visible_action_fields": ["method", "phase", "response_projection", "model_probabilities"], "typed_oracle_used_after_probe_for_stop_only": True, "belief_duplicate_evidence_guard": True, "positive_authority": False},
        "fixed_policy": fixed_metrics,
        "active_policy": active_metrics,
        "query_reduction": {"mean": round(fixed_metrics["mean_queries"] - active_metrics["mean_queries"], 6), "median": round(fixed_metrics["median_queries"] - active_metrics["median_queries"], 6)},
        "episodes": episodes,
        "promotion": {"status": "diagnostic_only", "training_allowed": False, "memory_promotion_allowed": False, "capability_claim_allowed": False, "reason": "active_controller_replays typed oracle and is not a learned capability proof"},
        "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg36-active-belief-diagnostic-v1", "episodes": episodes}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-36 active belief diagnostic", "", "source-transfer checkpoint 只对 control 投影评分，typed oracle 仅在 candidate replay 后用于停止。", "", "| policy | typed recall | FPR | median queries | mean queries |", "|---|---:|---:|---:|---:|"]
    lines.append(f"| fixed all phases | {fixed_metrics['typed_recall']:.2f} | {fixed_metrics['false_positive_rate']:.2f} | {fixed_metrics['median_queries']:.1f} | {fixed_metrics['mean_queries']:.2f} |")
    lines.append(f"| active belief | {active_metrics['typed_recall']:.2f} | {active_metrics['false_positive_rate']:.2f} | {active_metrics['median_queries']:.1f} | {active_metrics['mean_queries']:.2f} |")
    lines.extend(["", "状态：`diagnostic_only`；不允许训练晋升或长期记忆。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "positive_episode_count": positive_count, "fixed": fixed_metrics, "active": active_metrics, "query_reduction": report["query_reduction"], "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
