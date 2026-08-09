"""Train and replay the compositional pre-probe policy on PG-50."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "research" / "pg50_stability_matrix_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg50_stability_matrix_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg50_stability_matrix_active_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg50_stability_matrix_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg50-stability-matrix"
CHECKPOINT_PATH = OUTPUT_DIR / "preprobe_action_value.pt"
PG48_SCRIPT = ROOT / "scripts" / "train_pg48_compositional_preprobe.py"
PG38_SCRIPT = ROOT / "scripts" / "train_pg38_effect_pair_candidate.py"
PG39_SCRIPT = ROOT / "scripts" / "train_pg39_coarse_delta_candidate.py"
PG43_SCRIPT = ROOT / "scripts" / "train_pg43_invariant_effect_candidate.py"
TRACE_MODULE_PATH = ROOT / "app" / "trace_aligned_dataset.py"
SEED = 20500802
EPOCHS = 360
THRESHOLD = 0.60
KNOWN_BINDINGS = {"markup-context": "xss", "operator-context": "injection", "auth-boundary": "authentication", "subject-boundary": "access_control", "state-boundary": "logic", "destination-context": "url_redirect", "validation-boundary": "input_validation", "command-boundary": "command_injection", "ordinary-surface": "ordinary_response"}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(model: Any, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(features.to(device))).cpu()
    positive = labels.bool()
    accepted = probabilities >= THRESHOLD
    return {"pair_count": len(labels), "positive_count": int(positive.sum()), "positive_recall": round(float((positive & accepted).sum()) / max(int(positive.sum()), 1), 6), "negative_false_accept_count": int((~positive & accepted).sum()), "mean_probability": round(float(probabilities.mean()), 6)}


def _rewrite_trace(trace: dict[str, Any], implementation: str) -> dict[str, Any]:
    rewritten = copy.deepcopy(trace)
    rewritten["schema_version"] = "pg-pk-50-stability-matrix-active-trace-v1"
    rewritten["purpose"] = f"PG-50 {implementation} pre-probe replay trace"
    rewritten["evaluation_only"] = True
    rewritten["training_eligible"] = False
    rewritten["implementation"] = implementation
    rewritten["raw_probe_strings_stored"] = False
    rewritten["raw_response_bodies_stored"] = False
    return rewritten


def _surface_matrix(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in trace["episode_records"]:
        rows.append({key: item[key] for key in ("implementation", "surface_id", "surface_variant", "sampling_seed", "positive", "step_count", "status", "final_route", "abstain")})
    return rows


def main() -> int:
    pg48 = _load(PG48_SCRIPT, "pg48_for_pg50")
    pg38 = _load(PG38_SCRIPT, "pg38_for_pg50")
    pg39 = _load(PG39_SCRIPT, "pg39_for_pg50")
    pg43 = _load(PG43_SCRIPT, "pg43_for_pg50")
    trace_module = _load(TRACE_MODULE_PATH, "trace_for_pg50")
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pairs = pg38._pair_rows(list(catalog["samples"]))
    action_index = dict(pg48.ACTION_INDEX)
    training_pairs = [pair for pair in pairs if pair["candidate"].get("implementation") == "ember" and pair["candidate"].get("dataset_role") == "train" and (pair["candidate"].get("method"), pair["candidate"].get("phase")) in action_index]
    dev_pairs = [pair for pair in pairs if pair["candidate"].get("implementation") == "ember" and pair["candidate"].get("dataset_role") == "dev" and (pair["candidate"].get("method"), pair["candidate"].get("phase")) in action_index]
    semantic_index = {semantic: index for index, semantic in enumerate(sorted({str(pair["candidate"].get("semantic_reference", "")) for pair in training_pairs}))}
    channel_index = {channel: index for index, channel in enumerate(pg48.CHANNELS)}
    neutral = (1.0, 0.0, 1.0)
    train_features = pg48._features(training_pairs, [(pair["candidate"]["method"], pair["candidate"]["phase"]) for pair in training_pairs], [neutral] * len(training_pairs), semantic_index, channel_index)
    dev_features = pg48._features(dev_pairs, [(pair["candidate"]["method"], pair["candidate"]["phase"]) for pair in dev_pairs], [neutral] * len(dev_pairs), semantic_index, channel_index)
    train_labels = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in training_pairs], dtype=torch.float32)
    dev_labels = torch.tensor([bool(pair["candidate"]["oracle_projection"].get("positive", False)) for pair in dev_pairs], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = pg48.PreProbeActionValueModel(train_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01)
    pos_weight = torch.tensor([max(1.0, float((len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1.0)))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state: dict[str, torch.Tensor] | None = None
    best_selection = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss_fn(model(train_features.to(device)), train_labels.to(device))
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch % 60 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                dev_loss = loss_fn(model(dev_features.to(device)), dev_labels.to(device))
            selection = float((train_loss.detach() + 0.5 * dev_loss.detach()).cpu())
            history.append({"epoch": epoch, "train_loss": round(float(train_loss.detach()), 6), "dev_loss": round(float(dev_loss.detach()), 6), "selection": round(selection, 6)})
            if selection < best_selection:
                best_selection = selection
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)

    effect_checkpoint = torch.load(ROOT / "artifacts" / "pg43-invariant-effect" / "invariant_effect.pt", map_location="cpu", weights_only=False)
    effect_model = pg43.InvariantEffectModel()
    effect_model.load_state_dict(effect_checkpoint["model_state"])
    effect_model.eval()
    indices = tuple(int(item) for item in effect_checkpoint["invariant_indices"])
    pg48.KNOWN_BINDINGS.clear()
    pg48.KNOWN_BINDINGS.update(KNOWN_BINDINGS)
    implementation_results: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for implementation in ("frost", "quartz"):
        selected = [pair for pair in pairs if pair["candidate"].get("implementation") == implementation]
        result, trace = pg48._run_policy(pg38, pg39, selected, model, effect_model, indices, semantic_index, channel_index, device, trace_module)
        trace = _rewrite_trace(trace, implementation)
        implementation_results[implementation] = result
        traces[implementation] = trace
    holdout_pairs = [pair for pair in pairs if pair["candidate"].get("implementation") in {"frost", "quartz"}]
    combined, combined_trace = pg48._run_policy(pg38, pg39, holdout_pairs, model, effect_model, indices, semantic_index, channel_index, device, trace_module)
    combined_trace = _rewrite_trace(combined_trace, "frost+quartz")
    combined_matrix = _surface_matrix(combined_trace)
    all_safe = all(item["metrics"]["effect_success_rate"] == 1.0 and item["metrics"]["known_family_recall"] == 1.0 and item["metrics"]["unknown_strict_abstain"] and item["metrics"]["negative_false_accept_count"] == 0 and item["metrics"]["accepted_trace_episode_count"] == item["metrics"]["episode_count"] for item in implementation_results.values())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "sift-pg50-preprobe-action-value-checkpoint-v1", "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "semantic_index": semantic_index, "channel_index": channel_index, "action_order": [list(action) for action in pg48.ACTION_ORDER], "seed": SEED, "input_contract": "semantic/channel/action/belief only", "training_implementation": "ember", "holdout_implementations": ["frost", "quartz"]}, CHECKPOINT_PATH)
    checkpoint_sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    report = {"protocol_id": "sift-pg50-stability-matrix-v1", "schema_version": "pg-pk-50-stability-matrix-report-v1", "status": "diagnostic_only", "training": {"implementation": "ember", "pair_count": len(training_pairs), "positive_count": int(train_labels.sum()), "dev_pair_count": len(dev_pairs), "dev_positive_count": int(dev_labels.sum()), "semantic_index": semantic_index, "channel_index": channel_index, "response_projection_consumed_by_policy": False, "typed_oracle_consumed_by_model": False, "holdout_implementations_used_for_training": [], "epochs": EPOCHS, "seed": SEED, "device": str(device), "best_selection": round(best_selection, 6), "history_tail": history[-5:]}, "model": {"class": "PreProbeActionValueModel", "input": "semantic/channel slots + action + belief only", "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "checkpoint_sha256": checkpoint_sha256, "response_projection_consumed_by_policy": False, "executable": False}, "train_metrics": _metrics(model, train_features, train_labels, device), "dev_metrics": _metrics(model, dev_features, dev_labels, device), "implementation_holdouts": {key: value["metrics"] for key, value in implementation_results.items()}, "action_counts": {key: value["action_counts"] for key, value in implementation_results.items()}, "method_counts": {key: value["method_counts"] for key, value in implementation_results.items()}, "stability_matrix": combined_matrix, "safe_gate": {"schema_version": "sift-pg50-stability-gate-v1", "status": "passed" if all_safe else "blocked", "claim_allowed": all_safe, "reasons": [] if all_safe else ["implementation_holdout_gate_failed"], "training_allowed": False, "memory_promotion_allowed": False}, "formal_capability_claim_allowed": False, "formal_claim_blockers": ["template_boundary_is_unknown_and_abstained", "multi_implementation_matrix_is_still_synthetic", "real_authorized_target_validation_is_not_complete"], "promotion": {"status": "quarantined_stability_candidate", "training_allowed": False, "memory_promotion_allowed": False}, "trace": str(TRACE_PATH.relative_to(ROOT)), "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "manifest_sha256": hashlib.sha256(json.dumps({"protocol_id": "sift-pg50-stability-matrix-v1", "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(), "checkpoint_sha256": checkpoint_sha256, "frost_trace_sha256": traces["frost"]["trace_manifest_sha256"], "quartz_trace_sha256": traces["quartz"]["trace_manifest_sha256"]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    combined_trace["implementation_traces"] = {implementation: traces[implementation] for implementation in traces}
    TRACE_PATH.write_text(json.dumps(combined_trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-50 stability matrix", "", "训练只使用 ember train split 的 semantic/channel/action/belief；frost 与 quartz 是实现外 holdout。", "", "| implementation | effect success | known recall | unknown abstain | negative false accept | mean queries |", "|---|---:|---:|---:|---:|---:|"]
    for implementation, result in implementation_results.items():
        metric = result["metrics"]
        lines.append(f"| {implementation} | {metric['effect_success_rate']} | {metric['known_family_recall']} | {metric['unknown_strict_abstain']} | {metric['negative_false_accept_count']} | {metric['mean_queries']} |")
    lines.extend(["", f"稳定性安全门：`{report['safe_gate']['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "train_metrics": report["train_metrics"], "dev_metrics": report["dev_metrics"], "implementation_holdouts": report["implementation_holdouts"], "safe_gate": report["safe_gate"], "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
