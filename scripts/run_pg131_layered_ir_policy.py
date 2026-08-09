"""PG-131 train and audit an action policy over layered Rule-IR tokens."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.failure_guided_scheduler import validate_failure_signature
from app.layered_ir_tokenizer import layered_compress, validate_layered_compression
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.pg122_logic_authorization_replay import collect_target as collect_pg122_target
from app.pg125_scope_logic_replay import collect_target as collect_pg125_target
from app.pg127_resource_visibility_replay import collect_target as collect_pg127_target
from app.pg131_layered_ir_policy import (
    HIDDEN_DIM,
    IR_MODES,
    IR_TOKEN_FEATURE_DIM,
    LayeredIRActionPolicy,
    MAX_IR_TOKENS,
    SCHEMA_VERSION,
    layered_ir_token_matrix,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg131-layered-ir-policy-v1"
FULL_CHECKPOINT = ARTIFACT_DIR / "layered_ir_weighted.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "layered_ir_zero.pt"
TRACE = RESEARCH / "pg131_layered_ir_trace_v1.json"
DATASET = RESEARCH / "pg131_layered_ir_dataset_v1.json"
VISIBLE = RESEARCH / "pg131_layered_ir_visible_dataset_v1.json"
REPORT = RESEARCH / "pg131_layered_ir_report_v1.json"
PROTOCOL = RESEARCH / "pg131_layered_ir_protocol_v1.json"
PROPOSAL = RESEARCH / "pg131_layered_ir_proposal_v1.json"

PG127_TRAIN_SEEDS = (13111, 13113, 13115)
PG127_DEV_SEEDS = (13112, 13114, 13116)
PG127_HOLDOUT_SEEDS = (13101, 13103, 13105)
PG125_TRAIN_SEEDS = (13131, 13133, 13135)
PG125_DEV_SEEDS = (13132, 13134, 13136)
PG125_OOD_SEEDS = (13121, 13123, 13125)
PG122_FAMILY_OOD_SEEDS = (13141, 13143, 13145)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


async def _collect_targets() -> dict[str, list[dict[str, Any]]]:
    pg127_train = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_TRAIN_SEEDS)]
    pg127_dev = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_DEV_SEEDS)]
    pg127_holdout = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_HOLDOUT_SEEDS)]
    pg125_train = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_TRAIN_SEEDS)]
    pg125_dev = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_DEV_SEEDS)]
    pg125_ood = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_OOD_SEEDS)]
    pg122_family_ood = [await collect_pg122_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG122_FAMILY_OOD_SEEDS)]
    return {"pg127_train": pg127_train, "pg127_dev": pg127_dev, "pg127_holdout": pg127_holdout, "pg125_train": pg125_train, "pg125_dev": pg125_dev, "pg125_family_ood": pg125_ood, "pg122_family_ood": pg122_family_ood}


def _local_ir_layer(step: Mapping[str, Any], *, step_index: int, total_steps: int) -> dict[str, Any]:
    method = str(step["action_manifest"]["method"]).upper()
    html = f'<form method="{method}"><input name="abstract_probe"></form><script>fetch("local")</script>'
    javascript = 'if (document.querySelector("form")) { fetch("local"); }'
    result = layered_compress(html_snapshot=html, javascript_snapshot=javascript, action_manifests=[step["action_manifest"]], response_projection=step["response_projection"], failure_signature=step["failure_signature"])
    validate_layered_compression(result)
    ir_layer = dict(result["layers"]["ir_tokens"])
    progress = f"step_{step_index}_of_{total_steps}"
    tokens = [dict(token) for token in ir_layer["tokens"]]
    tokens.append({"layer": "ir", "kind": "slot", "slot_id": "trajectory.progress", "value": progress, "weight": 1.0})
    ir_layer["tokens"] = tokens
    ir_layer["token_count"] = len(tokens)
    ir_layer["ir_sha256"] = _sha256_json(tokens)
    return ir_layer


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            if episode.get("episode_report", {}).get("status") != "accepted_evaluation":
                raise ValueError(f"PG-131 source episode was not accepted: {episode.get('episode_id')}")
            prefix: list[dict[str, Any]] = []
            total_steps = len(episode["steps"])
            for step_index, step in enumerate(episode["steps"], start=1):
                signature = dict(step.get("failure_signature") or {})
                validate_failure_signature(signature)
                prefix.append(_local_ir_layer(step, step_index=step_index, total_steps=total_steps))
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "split": split, "target_seed": target.get("target_seed"), "episode_id": episode["episode_id"], "surface_kind": episode.get("surface_kind"), "step_id": step["step_id"], "ir_layers": [dict(layer) for layer in prefix], "failure_signature": signature, "label": signature["next_action"], "training_eligible": split in {"train", "dev"}, "memory_promotion_allowed": False})
    return rows


def _batch(rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = [layered_ir_token_matrix(row["ir_layers"], mode=mode) for row in rows]
    return torch.tensor(matrices, dtype=torch.float32, device=device), torch.tensor([policy_index(row["label"]) for row in rows], dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}}


def _allowed(signature: dict[str, Any]) -> set[str]:
    kind = str(signature.get("kind"))
    gate = str(signature.get("failed_gate"))
    methods = {str(item).upper() for item in signature.get("methods_seen", [])}
    remaining = int(signature.get("remaining_probe_budget", 0) or 0)
    if kind == "no_surface_delta" and gate == "matched_negative_control":
        return {"repeat_matched_negative_pair"}
    if kind in {"candidate_without_typed_effect", "no_surface_delta"}:
        return {"probe_candidate_other_method"} if len(methods) < 2 or remaining > 0 else {"abstain_candidate_only"}
    if kind == "oracle_unavailable":
        return {"replay_other_method"} if len(methods) < 2 or remaining > 0 else {"abstain_unknown_oracle"}
    if kind == "typed_positive":
        return {"probe_candidate_other_method"} if len(methods) < 2 else {"stop_confirmed_positive"}
    if kind == "method_disagreement":
        return {"repeat_matched_negative_pair", "abstain_candidate_only"}
    if kind == "budget_exhausted":
        return {"abstain_budget_exhausted"}
    return {"abstain_candidate_only"}


def _predict(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> tuple[list[int], list[int], list[float]]:
    x, y = _batch(rows, device, mode=mode)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), y.cpu().tolist(), confidence.cpu().tolist()


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, mode: str) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, device, mode=mode)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = [names[index] in _allowed(row["failure_signature"]) for index, row in enumerate(rows)]
    per_surface: dict[str, dict[str, float]] = {}
    for surface in sorted({row.get("surface_kind") for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.get("surface_kind") == surface]
        per_surface[surface] = {"count": float(len(indices)), "accuracy": round(sum(predictions[index] == labels[index] for index in indices) / len(indices), 6), "compliance": round(sum(compliant[index] for index in indices) / len(indices), 6)}
    budget_indices = [index for index, row in enumerate(rows) if int(row["failure_signature"].get("remaining_probe_budget", 0) or 0) > 0]
    # PG-127 has six steps (so s04 is intermediate), while PG-125/PG-122
    # have four.  Compute the true final row per episode rather than counting
    # both suffixes as final.
    episode_max: dict[str, int] = {}
    for row in rows:
        match = re.search(r"-s(\d+)$", str(row["step_id"]))
        if match:
            episode = str(row.get("episode_id", ""))
            episode_max[episode] = max(episode_max.get(episode, 0), int(match.group(1)))
    blind_final = []
    for index, row in enumerate(rows):
        if row.get("surface_kind") != "blind":
            continue
        match = re.search(r"-s(\d+)$", str(row["step_id"]))
        if match and int(match.group(1)) == episode_max.get(str(row.get("episode_id", "")), -1):
            blind_final.append(index)
    negative_indices = [index for index, row in enumerate(rows) if row.get("surface_kind") in {"decoy", "steady", "blind"}]
    return {"mode": mode, "metrics": _metrics(predictions, labels), "safety_compliance_rate": round(sum(compliant) / len(compliant), 6), "non_abstain_count": sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names), "mean_confidence": round(sum(confidences) / len(confidences), 6), "per_surface": per_surface, "budget_rows": len(budget_indices), "budget_accuracy": round(sum(predictions[index] == labels[index] for index in budget_indices) / len(budget_indices), 6) if budget_indices else 0.0, "blind_final_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in blind_final) / len(blind_final), 6) if blind_final else 0.0, "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative_indices), "predictions": predictions, "labels": labels}


def _train(train: list[dict[str, Any]], dev: list[dict[str, Any]], device: torch.device, *, mode: str, seed: int) -> tuple[nn.Module, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = LayeredIRActionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train, device, mode=mode)
    history: list[dict[str, float]] = []
    for epoch in range(1, 121):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_pred, _, _ = _predict(model, train, device, mode=mode)
        dev_pred, _, _ = _predict(model, dev, device, mode=mode)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_pred, [policy_index(row["label"]) for row in train])["accuracy"], "dev_accuracy": _metrics(dev_pred, [policy_index(row["label"]) for row in dev])["accuracy"]})
    return model, history


def _summary(targets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {key: [{"target_seed": target.get("target_seed"), "decoy_strength": target.get("decoy_strength"), "episodes": len(target.get("episodes", [])), "steps": sum(len(episode.get("steps", [])) for episode in target.get("episodes", []))} for target in value] for key, value in targets.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect_targets())
    train = _rows_from_targets(targets["pg127_train"], split="train", source="pg127_layered_ir_train") + _rows_from_targets(targets["pg125_train"], split="train", source="pg125_layered_ir_train")
    dev = _rows_from_targets(targets["pg127_dev"], split="dev", source="pg127_layered_ir_dev") + _rows_from_targets(targets["pg125_dev"], split="dev", source="pg125_layered_ir_dev")
    pg127_holdout = _rows_from_targets(targets["pg127_holdout"], split="holdout", source="pg127_layered_ir_holdout")
    pg125_ood = _rows_from_targets(targets["pg125_family_ood"], split="family_ood", source="pg125_layered_ir_family_ood")
    pg122_ood = _rows_from_targets(targets["pg122_family_ood"], split="family_ood", source="pg122_layered_ir_family_ood")
    model, history = _train(train, dev, device, mode="weighted", seed=13131)
    zero_model, zero_history = _train(train, dev, device, mode="zero", seed=13131)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "feature_dim": IR_TOKEN_FEATURE_DIM, "max_ir_tokens": MAX_IR_TOKENS, "policy_actions": list(POLICY_ACTIONS), "mode": "weighted", "model_state_dict": model.state_dict()}, FULL_CHECKPOINT)
    torch.save({"schema_version": SCHEMA_VERSION, "feature_dim": IR_TOKEN_FEATURE_DIM, "max_ir_tokens": MAX_IR_TOKENS, "policy_actions": list(POLICY_ACTIONS), "mode": "zero", "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    dev_report = _evaluate(model, dev, device, mode="weighted")
    holdout_report = _evaluate(model, pg127_holdout, device, mode="weighted")
    family_report = _evaluate(model, pg125_ood, device, mode="weighted")
    pg122_family_report = _evaluate(model, pg122_ood, device, mode="weighted")
    uniform_holdout = _evaluate(model, pg127_holdout, device, mode="uniform")
    no_failure_holdout = _evaluate(model, pg127_holdout, device, mode="no_failure_slots")
    zeroed_holdout = _evaluate(model, pg127_holdout, device, mode="zero")
    zero_baseline = _evaluate(zero_model, pg127_holdout, device, mode="zero")
    train_seeds = sorted({row.get("target_seed") for row in train})
    dev_seeds = sorted({row.get("target_seed") for row in dev})
    holdout_seeds = sorted({row.get("target_seed") for row in pg127_holdout})
    family_seeds = sorted({row.get("target_seed") for row in pg125_ood})
    pg122_seeds = sorted({row.get("target_seed") for row in pg122_ood})
    get_count = sum(row["failure_signature"].get("observed_method") == "GET" for row in pg127_holdout + pg125_ood + pg122_ood)
    post_count = sum(row["failure_signature"].get("observed_method") == "POST" for row in pg127_holdout + pg125_ood + pg122_ood)
    checks = {"fresh_checkpoint": True, "same_capacity_ablation": sum(parameter.numel() for parameter in model.parameters()) == sum(parameter.numel() for parameter in zero_model.parameters()), "feature_dim": IR_TOKEN_FEATURE_DIM == 53, "max_ir_tokens": MAX_IR_TOKENS == 48, "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds + family_seeds + pg122_seeds) or set(dev_seeds) & set(holdout_seeds + family_seeds + pg122_seeds)), "get_post_balanced": get_count == post_count == 84, "holdout_not_training": not set(holdout_seeds + family_seeds + pg122_seeds).intersection(train_seeds + dev_seeds), "pg127_full_accuracy_floor": holdout_report["metrics"]["accuracy"] >= 0.95, "pg127_compliance_floor": holdout_report["safety_compliance_rate"] >= 0.95, "pg127_surface_floor": all(value["accuracy"] >= 0.95 for value in holdout_report["per_surface"].values()), "pg127_budget_floor": holdout_report["budget_accuracy"] >= 0.95, "pg127_blind_abstain": holdout_report["blind_final_abstain_rate"] == 1.0, "pg125_family_accuracy_floor": family_report["metrics"]["accuracy"] >= 0.95, "pg125_family_compliance_floor": family_report["safety_compliance_rate"] >= 0.95, "pg125_family_surface_floor": all(value["accuracy"] >= 0.95 for value in family_report["per_surface"].values()), "pg125_family_blind_abstain": family_report["blind_final_abstain_rate"] == 1.0, "pg122_family_accuracy_floor": pg122_family_report["metrics"]["accuracy"] >= 0.95, "pg122_family_compliance_floor": pg122_family_report["safety_compliance_rate"] >= 0.95, "pg122_family_surface_floor": all(value["accuracy"] >= 0.95 for value in pg122_family_report["per_surface"].values()), "pg122_family_blind_abstain": pg122_family_report["blind_final_abstain_rate"] == 1.0, "negative_false_stop_zero": holdout_report["negative_false_stop_count"] == 0 and family_report["negative_false_stop_count"] == 0 and pg122_family_report["negative_false_stop_count"] == 0, "non_abstain_nonzero": holdout_report["non_abstain_count"] > 0, "uniform_ir_ablation_changes": holdout_report["predictions"] != uniform_holdout["predictions"], "failure_ir_ablation_changes": holdout_report["predictions"] != no_failure_holdout["predictions"], "zero_ablation_changes": holdout_report["predictions"] != zeroed_holdout["predictions"], "memory_promotion_forbidden": True}
    report = {"protocol_id": "pg-pk-131-layered-ir-policy-v1", "schema_version": "pg131-layered-ir-report-v1", "status": "completed_pg131_layered_ir_policy", "scope": {"model": "fresh_layered_rule_ir_token_action_policy", "feature_dim": IR_TOKEN_FEATURE_DIM, "max_ir_tokens": MAX_IR_TOKENS, "hidden_dim": HIDDEN_DIM, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train), "dev_count": len(dev), "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "dev": dev_report, "source_summary": _summary({"pg127_train": targets["pg127_train"], "pg127_dev": targets["pg127_dev"], "pg125_train": targets["pg125_train"], "pg125_dev": targets["pg125_dev"]})}, "holdout": {"pg127_seed_holdout": holdout_report, "pg125_family_ood": family_report, "pg122_family_ood": pg122_family_report, "uniform_ir_ablation": uniform_holdout, "failure_ir_slots_zeroed": no_failure_holdout, "weighted_ir_zeroed": zeroed_holdout, "fresh_zero_baseline": zero_baseline}, "checks": checks, "input_contract": {"source_layers_are_local_only": True, "model_receives_ir_tokens_only": True, "source_html_javascript_retained": False, "oracle_authority_in_model_input": False, "raw_probe_response_in_model_input": False, "failure_slots_weighted_only_on_failure": True, "forward_baseline_after_recovery": True, "rule_ir_slots": ["surface.modalities", "transport.methods_seen", "response.transition_delta", "failure.kind", "failure.failed_gate", "failure.recovery_phase", "probe.remaining_budget", "trajectory.progress"]}, "diagnosis": {"representation_change": "把 PG-130 的 source token 压缩结果作为 Rule IR slot token 序列接入策略，不再把 raw surface 直接喂给模型。", "meaning": "layered Rule-IR token action selection with cross-family holdout; not vulnerability confirmation", "training_source": "fresh PG-127 and PG-125 long-horizon targets", "family_holdout_source": ["fresh PG-125 scope targets", "fresh PG-122 authorization targets"], "history_prefix_only": True}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_layered_ir_passed_pending_manual_review_no_memory_promotion" if all(checks.values()) else "blocked_pg131_gate_failure_preserved", "reason": "Rule IR token 只负责安全抽象动作；typed oracle、负对照、fresh reset 和 evidence hash 仍是最终出口门。"}, "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "policy": hashlib.sha256((ROOT / "app/pg131_layered_ir_policy.py").read_bytes()).hexdigest(), "tokenizer": hashlib.sha256((ROOT / "app/layered_ir_tokenizer.py").read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg131-layered-ir-dataset-v1", "training_eligible": all(checks.values()), "memory_promotion_allowed": False, "rows": train + dev + pg127_holdout + pg125_ood + pg122_ood}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg131-layered-ir-visible-v1", "training_eligible": all(checks.values()), "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "split": row["split"], "ir_layers": row["ir_layers"], "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "training_label": row["label"]} for row in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg131-layered-ir-trace-v1", "protocol_id": "pg-pk-131-layered-ir-policy-v1", "status": "completed_pg131_layered_ir_policy", "training_eligible": all(checks.values()), "memory_promotion_allowed": False, "target_summary": _summary(targets), "train_count": len(train), "dev_count": len(dev), "pg127_holdout_count": len(pg127_holdout), "pg125_family_ood_count": len(pg125_ood), "pg122_family_ood_count": len(pg122_ood), "source_html_javascript_saved": False, "model_input_ir_only": True, "raw_probe_response_saved": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-131-layered-ir-policy-v1", "schema_version": "pg131-layered-ir-protocol-v1", "objective": "将 source token 分层压缩为 Rule IR token，并验证其在新 seed 与独立 scope/authorization 实现上的安全动作泛化。", "training_source": "fresh PG-127 and PG-125 targets", "dev_source": "fresh disjoint seeds", "holdout_sources": ["fresh PG-127 seed holdout", "fresh PG-125 family holdout", "fresh PG-122 family holdout"], "model_input": {"layer": "Rule IR tokens only", "feature_dim": IR_TOKEN_FEATURE_DIM, "max_ir_tokens": MAX_IR_TOKENS, "masked_fields": ["raw_html", "raw_javascript", "raw_probe", "raw_response", "positive_authority", "typed_available", "target_id", "family"]}, "required_gates": checks, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-131-layered-ir-policy-v1", "proposal_id": "pg131-layered-ir-proposal-v1", "question": "分层 Rule IR token 是否能在不依赖页面/脚本表面的情况下保持安全动作、预算探索和 unknown abstain？", "prediction": {"pg127_accuracy": ">=0.95", "pg125_family_accuracy": ">=0.95", "pg122_family_accuracy": ">=0.95", "blind_abstain": 1.0, "negative_false_stop": 0, "failure_ir_ablation_changes": True}, "intervention": "fresh Transformer consumes only eight Rule IR slots per prefix step; failure slots get scheduler weights, forward recovery returns baseline slots", "failure_rule": "任一 family/seed/ablation 门失败都保留 trace、禁止训练/记忆晋升，先做实验-工程分诊。", "next": "通过后再接第三独立实现和原生 HTML/JS snapshot source split；失败则按 IR slot 与 source/IR 层级分解。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": IR_TOKEN_FEATURE_DIM, "max_ir_tokens": MAX_IR_TOKENS, "parameter_count": report["scope"]["parameter_count"], "train": len(train), "dev": len(dev), "pg127_holdout": len(pg127_holdout), "pg125_family_ood": len(pg125_ood), "pg122_family_ood": len(pg122_ood), "pg127_accuracy": holdout_report["metrics"]["accuracy"], "pg125_family_accuracy": family_report["metrics"]["accuracy"], "pg122_family_accuracy": pg122_family_report["metrics"]["accuracy"], "pg127_blind_abstain": holdout_report["blind_final_abstain_rate"], "pg125_blind_abstain": family_report["blind_final_abstain_rate"], "pg122_blind_abstain": pg122_family_report["blind_final_abstain_rate"], "uniform_accuracy": uniform_holdout["metrics"]["accuracy"], "no_failure_accuracy": no_failure_holdout["metrics"]["accuracy"], "zeroed_accuracy": zeroed_holdout["metrics"]["accuracy"], "all_gates": all(checks.values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
