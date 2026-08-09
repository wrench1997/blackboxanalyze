"""Train a response-free compositional pre-probe policy on ember and test frost."""

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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "research" / "pg48_compositional_preprobe_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg48_compositional_preprobe_active_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg48_compositional_preprobe_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg48_compositional_preprobe_report_v1.md"
EFFECT_CHECKPOINT_PATH = ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
TRACE_MODULE_PATH = ROOT / "app" / "trace_aligned_dataset.py"
SEED = 20480802
EPOCHS = 360
THRESHOLD = 0.60
ACTION_ORDER = (("GET", "screen"), ("POST", "screen"), ("GET", "confirm"), ("POST", "confirm"))
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
CHANNELS = ("query-channel", "form-channel")
KNOWN_BINDINGS = {"markup-context": "xss", "operator-context": "injection", "auth-boundary": "authentication", "subject-boundary": "access_control", "state-boundary": "logic", "destination-context": "url_redirect", "ordinary-surface": "ordinary_response"}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class PreProbeActionValueModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__(); self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU()); self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.head(self.encoder(features)).squeeze(-1)


def _semantic(pair: dict[str, Any]) -> str: return str(pair["candidate"].get("semantic_reference", ""))
def _channel(pair: dict[str, Any]) -> str: return str(pair["candidate"].get("channel_reference", ""))


def _features(pairs: list[dict[str, Any]], actions: list[tuple[str, str]], beliefs: list[tuple[float, float, float]], semantic_index: dict[str, int], channel_index: dict[str, int]) -> torch.Tensor:
    values = torch.zeros((len(pairs), len(semantic_index) + len(channel_index) + len(ACTION_ORDER) + 3), dtype=torch.float32)
    for row, (pair, action, belief) in enumerate(zip(pairs, actions, beliefs)):
        semantic = _semantic(pair); channel = _channel(pair)
        if semantic in semantic_index: values[row, semantic_index[semantic]] = 1.0
        if channel in channel_index: values[row, len(semantic_index) + channel_index[channel]] = 1.0
        values[row, len(semantic_index) + len(channel_index) + ACTION_INDEX[action]] = 1.0
        values[row, -3:] = torch.tensor(belief, dtype=torch.float32)
    return values


def _effect_scores(pg39: Any, pairs: list[dict[str, Any]], indices: tuple[int, ...], effect_model: Any) -> torch.Tensor:
    if not pairs: return torch.empty((0,))
    features = torch.sign(torch.stack([pg39._coarse_pair(pair) for pair in pairs])[:, indices])
    with torch.inference_mode(): return torch.sigmoid(effect_model(features)).cpu()


def _trace_step(trace_module: Any, row: dict[str, Any], control: dict[str, Any], episode_id: str, step_id: str, parent: str | None, next_action: str, belief: dict[str, float], after: dict[str, float], effect_score: float, route: str) -> dict[str, Any]:
    oracle = copy.deepcopy(row["oracle_projection"]); oracle["negative_control_pair_id"] = control["sample_id"]; manifest = row["payload_manifest"]; action = {"method": manifest["method"], "route_template_id": manifest["route_template_id"], "placement": manifest["placement"], "encoding_chain": manifest["encoding_chain"], "probe_ref": manifest["probe_ref"], "probe_sha256": manifest["payload_sha256"], "safety": {"no_external_network": True, "does_not_execute": True, "no_database_write": True, "no_credential_access": True}}; 
    if manifest["method"] == "POST": action["form_field_names"] = manifest["form_field_names"]
    effect_confirmed = effect_score >= THRESHOLD; positive = bool(oracle.get("positive", False)); unknown = route == "unknown_surface"; decision = "abstain" if effect_confirmed and unknown else "confirmed_positive" if effect_confirmed and positive else "confirmed_negative" if next_action == "stop_episode" else "candidate"; echo_body = {"action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_before": belief, "belief_after": after, "decision": decision, "next_action": next_action}; return trace_module.validate_trace_step({"episode_id": episode_id, "step_id": step_id, "parent_step_id": parent, "sampling_seed": int(row["sampling_seed"]), "target_instance_id": row["target_instance_id"], "hypothesis": route, "belief_before": belief, "action_manifest": action, "baseline_projection": control["response_projection"], "response_projection": row["response_projection"], "oracle_projection": oracle, "belief_after": after, "decision": decision, "next_action": next_action, "fresh_reset": row["reset"], "evidence_sha256": row["evidence"]["evidence_hash"], "dataset_stage": "trace_only", "online_weight_update": False, "long_term_memory_write": False, "echo": {"sha256": trace_module.sha256_json(echo_body)}})


def _run_policy(pg38: Any, pg39: Any, pairs: list[dict[str, Any]], model: PreProbeActionValueModel, effect_model: Any, indices: tuple[int, ...], semantic_index: dict[str, int], channel_index: dict[str, int], device: torch.device, trace_module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[tuple[str, str, str, int, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for pair in pairs:
        c = pair["candidate"]; groups[(str(c["implementation"]), str(c["surface_id"]), str(c["surface_variant"]), int(c["sampling_seed"]), str(c["semantic_reference"]))][(str(c["method"]), str(c["phase"]))] = pair
    all_steps: list[dict[str, Any]] = []; episode_reports: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []; action_counts: Counter[str] = Counter(); method_counts: Counter[str] = Counter(); qcounts: list[int] = []; pos_q: list[int] = []; neg_q: list[int] = []; pos_ep = neg_ep = effect_success = known_success = unknown_safe = false_accept = 0
    for task_key in sorted(groups):
        task = groups[task_key]; first = next(iter(task.values()))["candidate"]; semantic = task_key[4]; positive_episode = any(bool(p["candidate"]["oracle_projection"].get("positive", False)) for p in task.values()); pos_ep += int(positive_episode); neg_ep += int(not positive_episode); episode_id = f"pg48-active-{task_key[0]}-{task_key[1]}-{task_key[2]}-s{task_key[3]}"; belief = {"unknown": 1.0}; parent: str | None = None; steps: list[dict[str, Any]] = []; used: set[tuple[str, str]] = set(); stopped = False; route = "unknown_surface"
        # Required exploration gives both channels a screen observation;
        # confirm method is chosen from pre-probe semantic/channel slots.
        neutral_belief = (1.0, 0.0, 1.0)
        confirm_actions = [action for action in ACTION_ORDER[2:] if action in task]
        confirm_scores: dict[tuple[str, str], float] = {}
        for action in confirm_actions:
            preprobe_feature = _features([task[action]], [action], [neutral_belief], semantic_index, channel_index)
            with torch.inference_mode():
                confirm_scores[action] = float(torch.sigmoid(model(preprobe_feature.to(device))).cpu()[0])
        confirm_actions.sort(key=lambda action: (-confirm_scores[action], ACTION_INDEX[action]))
        selected = list(ACTION_ORDER[:2]) + confirm_actions
        for action_index, action in enumerate(selected):
            if action not in task: continue
            pair = task[action]; c = pair["candidate"]; current = (float(belief.get("unknown", 1.0)), max([v for k, v in belief.items() if k != "unknown"] or [0.0]), 1.0 if len(belief) == 1 else 0.0); feature = _features([pair], [action], [current], semantic_index, channel_index)
            with torch.inference_mode(): preprobe_score = float(torch.sigmoid(model(feature.to(device))).cpu()[0])
            # The action has now been sent; only the bounded post-response
            # effect head may decide whether to stop.
            effect_score = float(_effect_scores(pg39, [pair], indices, effect_model)[0]); effect = effect_score >= THRESHOLD; known = semantic in KNOWN_BINDINGS; route = KNOWN_BINDINGS.get(semantic, "unknown_surface") if effect else "unknown_surface"; after = {"unknown": round(1.0 - max(THRESHOLD, min(0.99, effect_score)), 6), **({KNOWN_BINDINGS[semantic]: round(max(THRESHOLD, min(0.99, effect_score)), 6)} if effect and known else {})} if effect else {"unknown": 1.0}; next_action = "stop_episode" if effect else "probe_confirm_other" if action_index >= 1 and action_index + 1 < len(selected) else "probe_next"; step = _trace_step(trace_module, c, pair["control"], episode_id, f"{episode_id}-{action[0].casefold()}-{action[1]}", parent, next_action, belief, after, effect_score, route); steps.append(step); all_steps.append(step); parent = step["step_id"]; belief = after; used.add(action); action_counts[f"{action[0]}.{action[1]}"] += 1; method_counts[action[0]] += 1
            if effect: stopped = True; break
        episode_report = trace_module.evaluate_episode(steps); episode_reports.append(episode_report); q = len(steps); qcounts.append(q); (pos_q if positive_episode else neg_q).append(q); effect_success += int(positive_episode and stopped); known_success += int(positive_episode and stopped and semantic in KNOWN_BINDINGS); unknown_safe += int(positive_episode and stopped and semantic not in KNOWN_BINDINGS and route == "unknown_surface"); false_accept += int((not positive_episode) and stopped); records.append({"episode_id": episode_id, "implementation": task_key[0], "surface_id": task_key[1], "surface_variant": task_key[2], "sampling_seed": task_key[3], "semantic_reference": semantic, "positive": positive_episode, "step_count": q, "status": episode_report["status"], "trace_sha256": episode_report["trace_sha256"], "final_route": route, "abstain": bool((not stopped) or route == "unknown_surface")})
    positive_count = sum(int(item["positive"]) for item in records); known_count = sum(int(item["positive"] and item["semantic_reference"] in KNOWN_BINDINGS) for item in records); unknown_count = positive_count - known_count; metrics = {"episode_count": len(records), "positive_episode_count": positive_count, "negative_episode_count": len(records) - positive_count, "effect_success_rate": round(effect_success / max(positive_count, 1), 6), "known_positive_count": known_count, "known_family_recall": round(known_success / max(known_count, 1), 6), "unknown_positive_count": unknown_count, "unknown_safe_abstain_count": unknown_safe, "unknown_strict_abstain": unknown_safe == unknown_count, "negative_false_accept_count": false_accept, "negative_false_accept_rate": round(false_accept / max(len(records) - positive_count, 1), 6), "mean_queries": round(float(statistics.mean(qcounts)), 6), "median_queries": float(statistics.median(qcounts)), "positive_mean_queries": round(float(statistics.mean(pos_q)), 6), "negative_mean_queries": round(float(statistics.mean(neg_q)), 6), "fixed_probe_baseline_queries": 4.0, "mean_query_reduction_rate": round((4.0 - float(statistics.mean(qcounts))) / 4.0, 6), "get_post_covered": set(method_counts) == {"GET", "POST"}, "accepted_trace_episode_count": sum(int(item["status"] == "accepted_evaluation") for item in episode_reports), "belief_update_count": len(all_steps)}; trace = {"schema_version": "pg-pk-48-compositional-preprobe-active-trace-v1", "purpose": "response-free semantic/channel active policy trace", "evaluation_only": True, "training_eligible": False, "methods": ["GET", "POST"], "steps": all_steps, "episodes": episode_reports, "episode_records": records, "episode_count": len(episode_reports), "accepted_evaluation_episode_count": metrics["accepted_trace_episode_count"], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "trace_manifest_sha256": trace_module.sha256_json([step["trace_sha256"] for step in all_steps])}; return {"metrics": metrics, "action_counts": dict(action_counts), "method_counts": dict(method_counts), "trace": trace}, trace


def main() -> int:
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg48"); pg39 = _load(PG39_SCRIPT, "pg39_for_pg48"); pg43 = _load(PG43_SCRIPT, "pg43_for_pg48"); trace_module = _load(TRACE_MODULE_PATH, "trace_for_pg48")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8")); pairs = pg38._pair_rows(list(catalog["samples"])); training_pairs = [p for p in pairs if p["candidate"]["implementation"] == "ember" and int(p["candidate"]["sampling_seed"]) in {401, 409} and p["candidate"]["surface_id"] != "node-07" and (p["candidate"]["method"], p["candidate"]["phase"]) in ACTION_INDEX]; seed_pairs = [p for p in pairs if p["candidate"]["implementation"] == "ember" and int(p["candidate"]["sampling_seed"]) == 419 and p["candidate"]["surface_id"] != "node-07" and (p["candidate"]["method"], p["candidate"]["phase"]) in ACTION_INDEX]; semantic_index = {semantic: index for index, semantic in enumerate(sorted({str(p["candidate"]["semantic_reference"]) for p in training_pairs}))}; channel_index = {channel: index for index, channel in enumerate(CHANNELS)}; neutral = (1.0, 0.0, 1.0); train_features = _features(training_pairs, [(p["candidate"]["method"], p["candidate"]["phase"]) for p in training_pairs], [neutral] * len(training_pairs), semantic_index, channel_index); seed_features = _features(seed_pairs, [(p["candidate"]["method"], p["candidate"]["phase"]) for p in seed_pairs], [neutral] * len(seed_pairs), semantic_index, channel_index); train_labels = torch.tensor([bool(p["candidate"]["oracle_projection"].get("positive", False)) for p in training_pairs], dtype=torch.float32); seed_labels = torch.tensor([bool(p["candidate"]["oracle_projection"].get("positive", False)) for p in seed_pairs], dtype=torch.float32); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(SEED); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    model = PreProbeActionValueModel(train_features.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01); pos_weight = torch.tensor([max(1.0, float((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0)))], device=device); loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight); best_state: dict[str, torch.Tensor] | None = None; best_selection = float("inf"); history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); train_loss = loss_fn(model(train_features.to(device)), train_labels.to(device)); train_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch % 60 == 0 or epoch == 1:
            model.eval();
            with torch.inference_mode(): seed_loss = loss_fn(model(seed_features.to(device)), seed_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * seed_loss.detach()).cpu()); history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "seed_loss": round(float(seed_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection: best_selection = selection; best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None: model.load_state_dict(best_state)
    effect_checkpoint = torch.load(EFFECT_CHECKPOINT_PATH, map_location="cpu", weights_only=False); effect_model = pg43.InvariantEffectModel(); effect_model.load_state_dict(effect_checkpoint["model_state"]); effect_model.eval(); indices = tuple(int(item) for item in effect_checkpoint["invariant_indices"]); holdout_pairs = [p for p in pairs if p["candidate"]["implementation"] == "frost"]; heldout, trace = _run_policy(pg38, pg39, holdout_pairs, model, effect_model, indices, semantic_index, channel_index, device, trace_module); train_metrics = {"pair_count": len(training_pairs), "positive_count": int(train_labels.sum()), "effect_recall": 0.0}; seed_metrics = {"pair_count": len(seed_pairs), "positive_count": int(seed_labels.sum()), "effect_recall": 0.0};
    with torch.inference_mode(): train_probs = torch.sigmoid(model(train_features.to(device))).cpu(); seed_probs = torch.sigmoid(model(seed_features.to(device))).cpu()
    train_metrics["effect_recall"] = round(float(((train_labels.bool()) & (train_probs >= THRESHOLD)).sum()) / max(int(train_labels.sum()), 1), 6); seed_metrics["effect_recall"] = round(float(((seed_labels.bool()) & (seed_probs >= THRESHOLD)).sum()) / max(int(seed_labels.sum()), 1), 6)
    OUTPUT_DIR = ROOT / "artifacts" / "pg48-compositional-preprobe"; OUTPUT_DIR.mkdir(parents=True, exist_ok=True); checkpoint_path = OUTPUT_DIR / "preprobe_action_value.pt"; torch.save({"schema_version": "sift-pg48-preprobe-action-value-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "semantic_index": semantic_index, "channel_index": channel_index, "action_order": [list(action) for action in ACTION_ORDER], "seed": SEED, "input_contract": "semantic/channel/action/belief only"}, checkpoint_path); checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(); gate_reasons = []
    for key, condition in (("effect_success_rate", heldout["metrics"]["effect_success_rate"] < 1.0), ("known_family_recall", heldout["metrics"]["known_family_recall"] < 1.0), ("unknown_strict_abstain", not heldout["metrics"]["unknown_strict_abstain"]), ("negative_false_accept", heldout["metrics"]["negative_false_accept_count"] != 0), ("trace", heldout["metrics"]["accepted_trace_episode_count"] != heldout["metrics"]["episode_count"])):
        if condition: gate_reasons.append(key)
    safe_gate = {"schema_version": "sift-pg48-preprobe-gate-v1", "status": "passed" if not gate_reasons else "blocked", "claim_allowed": not gate_reasons, "reasons": gate_reasons, "training_allowed": False, "memory_promotion_allowed": False}; report = {"protocol_id": "sift-pg48-compositional-preprobe-v1", "schema_version": "pg-pk-48-compositional-preprobe-report-v1", "status": "diagnostic_only", "training": {"implementation": "ember", "pair_count": len(training_pairs), "positive_count": int(train_labels.sum()), "seeds": [401, 409], "semantic_index": semantic_index, "channel_index": channel_index, "response_projection_consumed_by_policy": False, "typed_oracle_consumed_by_model": False, "pg48_frost_used_for_training": False, "epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:]}, "model": {"class": "PreProbeActionValueModel", "input": "semantic/channel slots + action + belief only", "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "response_projection_consumed_by_policy": False, "executable": False}, "train_metrics": train_metrics, "seed_holdout_metrics": seed_metrics, "frost_holdout": heldout["metrics"], "action_counts": heldout["action_counts"], "method_counts": heldout["method_counts"], "safe_gate": safe_gate, "formal_capability_claim_allowed": False, "formal_claim_blockers": ["template_boundary_is_unknown_and_abstained", "one_independent_implementation_holdout_is_not_full_capability_proof"], "promotion": {"status": "quarantined_preprobe_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "trace": str(TRACE_PATH.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg48-compositional-preprobe-v1", "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": checkpoint_sha256, "trace_sha256": trace["trace_manifest_sha256"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}; TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); lines = ["# PG-48 compositional pre-probe", "", "发送前策略只看 semantic/channel slots、action 和 belief；response projection 只用于发送后 effect 确认。", "", "| 指标 | 值 |", "|---|---:|"]
    for key in ("effect_success_rate", "known_family_recall", "unknown_strict_abstain", "negative_false_accept_count", "mean_queries", "median_queries", "mean_query_reduction_rate", "get_post_covered", "accepted_trace_episode_count"): lines.append(f"| {key} | {heldout['metrics'][key]} |")
    lines.extend(["", f"安全门禁：`{safe_gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""]); MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8"); print(json.dumps({"protocol_id": report["protocol_id"], "train_metrics": train_metrics, "seed_holdout_metrics": seed_metrics, "frost_holdout": heldout["metrics"], "action_counts": heldout["action_counts"], "safe_gate": safe_gate, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
