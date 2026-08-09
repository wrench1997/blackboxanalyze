"""Train PG-47 action-value head on PG-40 and run a learned active policy on PG-42."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG40_CATALOG_PATH = ROOT / "research" / "pg40_semantic_router_catalog_v1.json"
PG42_CATALOG_PATH = ROOT / "research" / "pg42_independent_semantic_catalog_v1.json"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
PG46_SCRIPT = ROOT / "scripts" / "run_pg46_active_probe_policy.py"
OUTPUT_DIR = ROOT / "artifacts" / "pg47-counterfactual-action-value"
CHECKPOINT_PATH = OUTPUT_DIR / "action_value.pt"
REPORT_PATH = ROOT / "research" / "pg47_counterfactual_action_value_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg47_counterfactual_action_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg47_counterfactual_action_value_report_v1.md"
SEED = 20470802
EPOCHS = 320
EFFECT_THRESHOLD = 0.60
ACTION_ORDER = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
KNOWN_BINDINGS = {"markup-context": "xss", "operator-context": "injection", "auth-boundary": "authentication", "subject-boundary": "access_control", "state-invariant": "logic", "url-target": "url_redirect", "scalar-boundary": "input_validation", "local-canary": "command_injection", "ordinary-surface": "ordinary_response"}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class ActionValueModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(-1)


def _invariant(pg39: Any, pairs: list[dict[str, Any]], indices: tuple[int, ...]) -> torch.Tensor:
    if not pairs:
        return torch.empty((0, len(indices)), dtype=torch.float32)
    return torch.sign(torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, indices])


def _state_features(pg39: Any, pairs: list[dict[str, Any]], actions: list[tuple[str, str]], indices: tuple[int, ...], beliefs: list[tuple[float, float, float]]) -> torch.Tensor:
    invariant = _invariant(pg39, pairs, indices)
    action_one_hot = torch.zeros((len(pairs), len(ACTION_ORDER)), dtype=torch.float32)
    for row, action in enumerate(actions): action_one_hot[row, ACTION_INDEX[action]] = 1.0
    belief_tensor = torch.tensor(beliefs, dtype=torch.float32)
    return torch.cat([invariant, action_one_hot, belief_tensor], dim=1)


def _metrics(model: ActionValueModel, features: torch.Tensor, pairs: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not pairs: return {"pair_count": 0, "positive_count": 0, "negative_count": 0, "effect_recall": 0.0, "false_positive_rate": 0.0, "mean_probability": 0.0}
    with torch.inference_mode(): probability = torch.sigmoid(model(features.to(device))).cpu()
    positive = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in pairs])
    accepted = probability >= EFFECT_THRESHOLD
    return {"pair_count": len(pairs), "positive_count": int(positive.sum()), "negative_count": int((~positive).sum()), "effect_recall": round(float((positive & accepted).sum()) / max(int(positive.sum()), 1), 6), "false_positive_rate": round(float((~positive & accepted).sum()) / max(int((~positive).sum()), 1), 6), "mean_probability": round(float(probability.mean()), 6)}


def _episode_metrics(pg38: Any, pg39: Any, model: ActionValueModel, pairs: list[dict[str, Any]], indices: tuple[int, ...], device: torch.device, trace_module: Any, report_module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for pair in pairs:
        c = pair["candidate"]
        groups[(str(c["implementation"]), str(c["surface_id"]), str(c["surface_variant"]), int(c["sampling_seed"]))][(str(c["method"]), str(c["phase"]))] = pair
    all_steps: list[dict[str, Any]] = []; episode_reports: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []; action_counts: Counter[str] = Counter(); method_counts: Counter[str] = Counter(); query_counts: list[int] = []; positive_queries: list[int] = []; negative_queries: list[int] = []
    effect_success = known_success = unknown_safe = false_accept = pos_count = neg_count = 0; semantic_counts: Counter[str] = Counter()
    for task_key in sorted(groups):
        task = groups[task_key]; first = next(iter(task.values()))["candidate"]; semantic = str(first["payload_manifest"]["probe_ref"]).replace("pg42-semantic-", "")
        positive_episode = any(bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in task.values()); pos_count += int(positive_episode); neg_count += int(not positive_episode); semantic_counts[semantic] += int(positive_episode)
        episode_id = f"pg47-episode-{task_key[0]}-{task_key[1]}-{task_key[2]}-s{task_key[3]}"; steps: list[dict[str, Any]] = []; parent: str | None = None; belief = {"unknown": 1.0}; stopped = False; route = "unknown_surface"
        # Keep the first two orthogonal screens as a required exploration
        # prefix; the learned value head chooses the confirmation action.
        candidate_actions = list(ACTION_ORDER[:2])
        for action in ACTION_ORDER[2:]:
            candidate_actions.append(action)
        for action_index, action in enumerate(candidate_actions):
            if action not in task: continue
            pair = task[action]; candidate = pair["candidate"]; control = pair["control"]
            current_belief = (float(belief.get("unknown", 1.0)), max([value for key, value in belief.items() if key != "unknown"] or [0.0]), 1.0 if len(belief) == 1 else 0.0)
            feature = _state_features(pg39, [pair], [action], indices, [current_belief])
            with torch.inference_mode(): probability = float(torch.sigmoid(model(feature.to(device))).cpu()[0])
            effect = probability >= EFFECT_THRESHOLD; known = semantic in KNOWN_BINDINGS; route = KNOWN_BINDINGS.get(semantic, "unknown_surface") if effect else "unknown_surface"; abstain = bool(effect and not known)
            if effect:
                confidence = max(EFFECT_THRESHOLD, min(0.99, probability)); belief_after = {"unknown": round(1.0 - confidence, 6), **({KNOWN_BINDINGS[semantic]: round(confidence, 6)} if known else {})}
                next_action = "stop_episode"; stopped = True
            else:
                belief_after = {"unknown": 1.0}; next_action = "probe_" + (ACTION_ORDER[action_index + 1][0] + "_" + ACTION_ORDER[action_index + 1][1] if action_index + 1 < len(ACTION_ORDER) else "exhausted")
            # PG-46's trace helper validates fresh reset, bounded action,
            # negative pair, echo hash, and memory/update prohibitions.
            step = report_module._make_trace_step(trace_module, candidate, control, episode_id, f"{episode_id}-{action[0].casefold()}-{action[1]}", parent, next_action, belief, belief_after, probability, route, abstain)
            steps.append(step); all_steps.append(step); parent = step["step_id"]; belief = belief_after; action_counts[f"{action[0]}.{action[1]}"] += 1; method_counts[action[0]] += 1
            if stopped: break
        episode_report = trace_module.evaluate_episode(steps); episode_reports.append(episode_report); query_count = len(steps); query_counts.append(query_count); (positive_queries if positive_episode else negative_queries).append(query_count)
        effect_success += int(positive_episode and stopped); known_success += int(positive_episode and stopped and semantic in KNOWN_BINDINGS); unknown_safe += int(positive_episode and stopped and semantic not in KNOWN_BINDINGS and route == "unknown_surface"); false_accept += int((not positive_episode) and stopped)
        records.append({"episode_id": episode_id, "implementation": task_key[0], "surface_id": task_key[1], "surface_variant": task_key[2], "sampling_seed": task_key[3], "semantic_reference": semantic, "positive": positive_episode, "step_count": query_count, "status": episode_report["status"], "trace_sha256": episode_report["trace_sha256"], "final_route": route, "abstain": bool((not stopped) or route == "unknown_surface")})
    known_pos = sum(count for semantic, count in semantic_counts.items() if semantic in KNOWN_BINDINGS); unknown_pos = sum(count for semantic, count in semantic_counts.items() if semantic not in KNOWN_BINDINGS)
    metrics = {"episode_count": len(records), "positive_episode_count": pos_count, "negative_episode_count": neg_count, "effect_success_rate": round(effect_success / max(pos_count, 1), 6), "known_positive_count": known_pos, "known_family_recall": round(known_success / max(known_pos, 1), 6), "unknown_positive_count": unknown_pos, "unknown_safe_abstain_count": unknown_safe, "unknown_strict_abstain": unknown_safe == unknown_pos, "negative_false_accept_count": false_accept, "negative_false_accept_rate": round(false_accept / max(neg_count, 1), 6), "mean_queries": round(float(statistics.mean(query_counts)), 6), "median_queries": float(statistics.median(query_counts)), "positive_mean_queries": round(float(statistics.mean(positive_queries)), 6), "negative_mean_queries": round(float(statistics.mean(negative_queries)), 6), "fixed_probe_baseline_queries": 4.0, "mean_query_reduction_rate": round((4.0 - float(statistics.mean(query_counts))) / 4.0, 6), "get_post_covered": set(method_counts) == {"GET", "POST"}, "accepted_trace_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "belief_update_count": len(all_steps)}
    trace = {"schema_version": "pg-pk-47-counterfactual-action-trace-v1", "purpose": "learned action-value active probe trace", "evaluation_only": True, "training_eligible": False, "methods": ["GET", "POST"], "action_order": [f"{method}.{phase}" for method, phase in ACTION_ORDER], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "episodes": episode_reports, "episode_records": records, "steps": all_steps, "episode_count": len(episode_reports), "accepted_evaluation_episode_count": metrics["accepted_trace_episode_count"], "trace_manifest_sha256": trace_module.sha256_json([step["trace_sha256"] for step in all_steps])}
    return {"metrics": metrics, "action_counts": dict(action_counts), "method_counts": dict(method_counts), "semantic_counts": dict(semantic_counts), "trace": trace}, trace


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg47"); pg39 = _load(PG39_SCRIPT, "pg39_for_pg47"); pg43 = _load(PG43_SCRIPT, "pg43_for_pg47"); report_module = _load(PG46_SCRIPT, "pg46_helpers_for_pg47"); trace_module = _load(ROOT / "app" / "trace_aligned_dataset.py", "trace_for_pg47")
    pg40 = json.loads(PG40_CATALOG_PATH.read_text(encoding="utf-8")); pg42 = json.loads(PG42_CATALOG_PATH.read_text(encoding="utf-8"))
    pg40_pairs = pg38._pair_rows(list(pg40["samples"])); pg42_pairs = pg38._pair_rows(list(pg42["samples"]))
    effect_checkpoint = torch.load(ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt", map_location="cpu", weights_only=False); indices = tuple(int(item) for item in effect_checkpoint["invariant_indices"])
    train_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) in {361, 367} and (str(pair["candidate"]["method"]), str(pair["candidate"]["phase"])) in ACTION_INDEX]
    seed_pairs = [pair for pair in pg40_pairs if pair["implementation"] == "atlas" and int(pair["sampling_seed"]) == 373 and (str(pair["candidate"]["method"]), str(pair["candidate"]["phase"])) in ACTION_INDEX]
    neutral = (1.0, 0.0, 1.0)
    train_actions = [(str(pair["candidate"]["method"]), str(pair["candidate"]["phase"])) for pair in train_pairs]; seed_actions = [(str(pair["candidate"]["method"]), str(pair["candidate"]["phase"])) for pair in seed_pairs]
    train_features = _state_features(pg39, train_pairs, train_actions, indices, [neutral] * len(train_pairs)); seed_features = _state_features(pg39, seed_pairs, seed_actions, indices, [neutral] * len(seed_pairs))
    train_labels = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in train_pairs], dtype=torch.float32); seed_labels = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in seed_pairs], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = ActionValueModel(train_features.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01); pos_weight = torch.tensor([max(1.0, float((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0)))], device=device); loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state: dict[str, torch.Tensor] | None = None; best_selection = float("inf"); history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); train_loss = loss_fn(model(train_features.to(device)), train_labels.to(device)); train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 60 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode(): seed_loss = loss_fn(model(seed_features.to(device)), seed_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * seed_loss.detach()).cpu()); history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "seed_loss": round(float(seed_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection: best_selection = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    train_metrics = _metrics(model, train_features, train_pairs, device); seed_metrics = _metrics(model, seed_features, seed_pairs, device); heldout, trace = _episode_metrics(pg38, pg39, model, pg42_pairs, indices, device, trace_module, report_module)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); torch.save({"schema_version": "sift-pg47-action-value-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "invariant_indices": list(indices), "action_order": [list(action) for action in ACTION_ORDER], "seed": SEED}, CHECKPOINT_PATH); checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    gate_reasons = []
    if heldout["metrics"]["effect_success_rate"] < 1.0: gate_reasons.append("effect_success_below_1")
    if heldout["metrics"]["known_family_recall"] < 1.0: gate_reasons.append("known_family_recall_below_1")
    if not heldout["metrics"]["unknown_strict_abstain"]: gate_reasons.append("unknown_not_strict_abstain")
    if heldout["metrics"]["negative_false_accept_count"] != 0: gate_reasons.append("negative_false_accept")
    if heldout["metrics"]["accepted_trace_episode_count"] != heldout["metrics"]["episode_count"]: gate_reasons.append("trace_episode_not_accepted")
    safe_gate = {"schema_version": "sift-pg47-action-value-gate-v1", "status": "passed" if not gate_reasons else "blocked", "claim_allowed": not gate_reasons, "reasons": gate_reasons, "training_allowed": False, "memory_promotion_allowed": False}
    pg46_baseline = json.loads((ROOT / "research" / "pg46_active_probe_report_v1.json").read_text(encoding="utf-8"))
    report = {"protocol_id": "sift-pg47-counterfactual-action-replay-v1", "schema_version": "pg-pk-47-counterfactual-action-value-report-v1", "status": "diagnostic_only", "training": {"catalog": str(PG40_CATALOG_PATH.relative_to(ROOT)), "pair_count": len(train_pairs), "positive_count": int(train_labels.sum()), "implementation": "atlas", "seeds": [361, 367], "oracle_is_target_only": True, "pg42_used_for_training": False, "epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:]}, "model": {"class": "ActionValueModel", "input": "invariant delta + action one-hot + bounded belief features", "action_order": [f"{method}.{phase}" for method, phase in ACTION_ORDER], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "typed_oracle_consumed_by_model": False, "family_consumed_by_model": False}, "train_metrics": train_metrics, "seed_holdout_metrics": seed_metrics, "pg42_holdout": heldout["metrics"], "action_counts": heldout["action_counts"], "method_counts": heldout["method_counts"], "positive_semantic_counts": heldout["semantic_counts"], "pg46_baseline": {key: pg46_baseline["metrics"][key] for key in ("mean_queries", "median_queries", "effect_success_rate", "known_family_recall", "negative_false_accept_count")}, "safe_gate": safe_gate, "formal_capability_claim_allowed": False, "promotion": {"status": "quarantined_action_value_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "trace": str(TRACE_PATH.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg47-counterfactual-action-replay-v1", "catalog_sha256": hashlib.sha256(PG42_CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": checkpoint_sha256, "trace_sha256": trace["trace_manifest_sha256"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-47 counterfactual action value", "", "action-value head 只读取 bounded delta、action one-hot 和 belief，不读 typed oracle。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("effect_success_rate", "known_family_recall", "unknown_strict_abstain", "negative_false_accept_count", "mean_queries", "median_queries", "mean_query_reduction_rate", "get_post_covered", "accepted_trace_episode_count"):
        lines.append(f"| {key} | {heldout['metrics'][key]} |")
    lines.extend(["", f"安全门禁：`{safe_gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""]); MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "train_metrics": train_metrics, "seed_holdout_metrics": seed_metrics, "pg42_holdout": heldout["metrics"], "action_counts": heldout["action_counts"], "safe_gate": safe_gate, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
