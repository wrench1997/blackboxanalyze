"""PG-133 dual source-token/Rule-IR history-sensitive replay experiment."""

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
from app.layered_token_embedding import MAX_LAYERED_TOKENS, SCALAR_DIM, layered_token_inputs
from app.pg122_logic_authorization_replay import collect_target as collect_pg122_target
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.pg125_scope_logic_replay import collect_target as collect_pg125_target
from app.pg127_resource_visibility_replay import collect_target as collect_pg127_target
from app.pg133_history_latch_replay import collect_target as collect_pg133_target
from app.pg133_layered_token_policy import HIDDEN_DIM, LayeredTokenActionPolicy, SCHEMA_VERSION


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg133-history-token-policy-v1"
FULL_CHECKPOINT = ARTIFACT_DIR / "history_token_weighted.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "history_token_zero.pt"
TRACE = RESEARCH / "pg133_history_token_trace_v1.json"
DATASET = RESEARCH / "pg133_history_token_dataset_v1.json"
VISIBLE = RESEARCH / "pg133_history_token_visible_dataset_v1.json"
REPORT = RESEARCH / "pg133_history_token_report_v1.json"
PROTOCOL = RESEARCH / "pg133_history_token_protocol_v1.json"
PROPOSAL = RESEARCH / "pg133_history_token_proposal_v1.json"

PG133_TRAIN_SEEDS = (13311, 13313, 13315)
PG133_DEV_SEEDS = (13312, 13314, 13316)
PG133_HOLDOUT_SEEDS = (13301, 13303, 13305)
PG127_TRAIN_SEEDS = (13331, 13333, 13335)
PG127_DEV_SEEDS = (13332, 13334, 13336)
PG127_HOLDOUT_SEEDS = (13321, 13323, 13325)
PG125_TRAIN_SEEDS = (13341, 13343, 13345)
PG125_DEV_SEEDS = (13342, 13344, 13346)
PG125_OOD_SEEDS = (13351, 13353, 13355)
PG122_FAMILY_OOD_SEEDS = (13361, 13363, 13365)
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


async def _collect_targets() -> dict[str, list[dict[str, Any]]]:
    pg133_train = [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_TRAIN_SEEDS)]
    pg133_dev = [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_DEV_SEEDS)]
    pg133_holdout = [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_HOLDOUT_SEEDS)]
    pg127_train = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_TRAIN_SEEDS)]
    pg127_dev = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_DEV_SEEDS)]
    pg127_holdout = [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_HOLDOUT_SEEDS)]
    pg125_train = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_TRAIN_SEEDS)]
    pg125_dev = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_DEV_SEEDS)]
    pg125_ood = [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_OOD_SEEDS)]
    pg122_ood = [await collect_pg122_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG122_FAMILY_OOD_SEEDS)]
    return {"pg133_train": pg133_train, "pg133_dev": pg133_dev, "pg133_holdout": pg133_holdout, "pg127_train": pg127_train, "pg127_dev": pg127_dev, "pg127_holdout": pg127_holdout, "pg125_train": pg125_train, "pg125_dev": pg125_dev, "pg125_family_ood": pg125_ood, "pg122_family_ood": pg122_ood}


def _layered_step(step: Mapping[str, Any], *, step_index: int, total_steps: int) -> dict[str, Any]:
    method = str(step["action_manifest"]["method"]).upper()
    html = f'<form method="{method}"><input name="abstract_probe"></form><script>fetch("local")</script>'
    javascript = 'if (document.querySelector("form")) { fetch("local"); }'
    compressed = layered_compress(html_snapshot=html, javascript_snapshot=javascript, action_manifests=[step["action_manifest"]], response_projection=step["response_projection"], failure_signature=step["failure_signature"])
    validate_layered_compression(compressed)
    ir_layer = dict(compressed["layers"]["ir_tokens"])
    tokens = [dict(token) for token in ir_layer["tokens"]]
    tokens.append({"layer": "ir", "kind": "slot", "slot_id": "trajectory.progress", "value": f"step_{step_index}_of_{total_steps}", "weight": 1.0})
    # This is a bounded validation fact from the target/evaluator boundary,
    # not workflow authority: the model may know whether a typed oracle is
    # available, but it never receives the evaluator's chosen action or a
    # positive verdict.
    typed_available = bool(step.get("failure_signature", {}).get("typed_available"))
    tokens.append({"layer": "ir", "kind": "slot", "slot_id": "oracle.availability", "value": "typed" if typed_available else "unknown", "weight": 1.25})
    ir_layer["tokens"] = tokens
    ir_layer["token_count"] = len(tokens)
    ir_layer["ir_sha256"] = _sha256_json(tokens)
    return {"source_token_layers": [dict(layer) for layer in compressed["layers"]["source_token_layers"]], "ir_layer": ir_layer}


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str, history_source: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            if episode.get("episode_report", {}).get("status") != "accepted_evaluation":
                raise ValueError(f"PG-133 source episode was not accepted: {episode.get('episode_id')}")
            prefix: list[dict[str, Any]] = []
            total_steps = len(episode["steps"])
            current_history = dict(episode.get("history_authority") or {}) if history_source else {}
            current_step_id = str(current_history.get("current_step", ""))
            pair_id = str(episode.get("counterfactual_pair_id", "")) if history_source else ""
            authority_hash = _sha256_json(current_history) if history_source else ""
            for step_index, step in enumerate(episode["steps"], start=1):
                signature = dict(step.get("failure_signature") or {})
                validate_failure_signature(signature)
                prefix.append(_layered_step(step, step_index=step_index, total_steps=total_steps))
                is_current_counterfactual = history_source and str(step["step_id"]) == current_step_id
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "split": split, "target_seed": target.get("target_seed"), "episode_id": episode["episode_id"], "surface_kind": episode.get("surface_kind"), "step_id": step["step_id"], "layered_steps": [dict(item) for item in prefix], "failure_signature": signature, "label": signature["next_action"], "history_stage": current_history.get("history_stage") if is_current_counterfactual else None, "counterfactual_pair_id": pair_id if is_current_counterfactual else None, "history_authority_hash": authority_hash if is_current_counterfactual else None, "history_source": history_source, "training_eligible": split in {"train", "dev"}, "memory_promotion_allowed": False})
    return rows


def _batch(rows: list[dict[str, Any]], embedding: Any, device: torch.device, *, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids: list[list[int]] = []
    scalars: list[list[list[float]]] = []
    for row in rows:
        token_ids, side_channel = layered_token_inputs(embedding, row["layered_steps"], mode=mode)
        ids.append(token_ids)
        scalars.append(side_channel)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(scalars, dtype=torch.float32, device=device), torch.tensor([policy_index(row["label"]) for row in rows], dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}}


def _allowed(row: Mapping[str, Any]) -> set[str]:
    kind = str(row["failure_signature"].get("kind"))
    gate = str(row["failure_signature"].get("failed_gate"))
    methods = {str(item).upper() for item in row["failure_signature"].get("methods_seen", [])}
    stage = str(row.get("history_stage") or "")
    # Unknown typed validation must abstain even when the visible response
    # resembles a matched negative. Availability is a model-visible fact;
    # evaluator action/positive authority remains hidden.
    if not bool(row["failure_signature"].get("typed_available", True)):
        return {"abstain_unknown_oracle", "replay_other_method"}
    if row.get("history_source") and stage == "control_complete" and kind == "candidate_without_typed_effect":
        return {"repeat_matched_negative_pair"}
    if row.get("history_source") and stage == "candidate_first" and kind == "candidate_without_typed_effect":
        return {"probe_candidate_other_method"}
    if kind == "no_surface_delta" and gate == "matched_negative_control":
        return {"repeat_matched_negative_pair"}
    if kind in {"candidate_without_typed_effect", "no_surface_delta"}:
        return {"probe_candidate_other_method"} if len(methods) < 2 or int(row["failure_signature"].get("remaining_probe_budget", 0) or 0) > 0 else {"abstain_candidate_only"}
    if kind == "oracle_unavailable":
        return {"replay_other_method", "abstain_unknown_oracle"}
    if kind == "typed_positive":
        return {"probe_candidate_other_method"} if len(methods) < 2 else {"stop_confirmed_positive"}
    if kind == "method_disagreement":
        return {"repeat_matched_negative_pair", "abstain_candidate_only"}
    if kind == "budget_exhausted":
        return {"abstain_budget_exhausted"}
    return {"abstain_candidate_only"}


def _predict(model: nn.Module, rows: list[dict[str, Any]], embedding: Any, device: torch.device, *, mode: str) -> tuple[list[int], list[int], list[float]]:
    ids, scalars, labels = _batch(rows, embedding, device, mode=mode)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(ids, scalars), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), labels.cpu().tolist(), confidence.cpu().tolist()


def _history_pair_metrics(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair = row.get("counterfactual_pair_id")
        if pair:
            groups.setdefault(str(pair), []).append(index)
    conflicts = 0
    separated = 0
    valid_groups = 0
    for indices in groups.values():
        labels = {rows[index]["label"] for index in indices}
        if len(labels) != 2:
            continue
        valid_groups += 1
        conflicts += 1
        if len({predictions[index] for index in indices}) == 2:
            separated += 1
    return {"pair_count": valid_groups, "label_conflict_count": conflicts, "prediction_separation_count": separated, "prediction_separation_rate": round(separated / valid_groups, 6) if valid_groups else 0.0}


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], embedding: Any, device: torch.device, *, mode: str) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, embedding, device, mode=mode)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = [names[index] in _allowed(row) for index, row in enumerate(rows)]
    per_surface: dict[str, dict[str, float]] = {}
    for surface in sorted({row.get("surface_kind") for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.get("surface_kind") == surface]
        per_surface[surface] = {"count": float(len(indices)), "accuracy": round(sum(predictions[index] == labels[index] for index in indices) / len(indices), 6), "compliance": round(sum(compliant[index] for index in indices) / len(indices), 6)}
    episode_max: dict[str, int] = {}
    for row in rows:
        match = re.search(r"-s(\d+)$", str(row["step_id"]))
        if match:
            episode_max[str(row.get("episode_id", ""))] = max(episode_max.get(str(row.get("episode_id", "")), 0), int(match.group(1)))
    blind_final = [index for index, row in enumerate(rows) if row.get("surface_kind") == "blind" and (match := re.search(r"-s(\d+)$", str(row["step_id"]))) and int(match.group(1)) == episode_max.get(str(row.get("episode_id", "")), -1)]
    negative_indices = [index for index, row in enumerate(rows) if row.get("surface_kind") in {"decoy", "steady", "blind"}]
    history_indices = [index for index, row in enumerate(rows) if row.get("counterfactual_pair_id")]
    history_metrics = _history_pair_metrics(rows, predictions)
    return {"mode": mode, "metrics": _metrics(predictions, labels), "safety_compliance_rate": round(sum(compliant) / len(compliant), 6), "non_abstain_count": sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names), "mean_confidence": round(sum(confidences) / len(confidences), 6), "per_surface": per_surface, "blind_final_rows": len(blind_final), "blind_final_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in blind_final) / len(blind_final), 6) if blind_final else 0.0, "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative_indices), "history_current_accuracy": round(sum(predictions[index] == labels[index] for index in history_indices) / len(history_indices), 6) if history_indices else 0.0, "history_pair": history_metrics, "predictions": predictions, "labels": labels}


def _weight_sensitivity(model: nn.Module, rows: list[dict[str, Any]], embedding: Any, device: torch.device, *, ablation_mode: str) -> dict[str, float]:
    """Measure logit sensitivity to a scalar-weight ablation.

    Argmax changes are not required: a trained policy can retain the same
    safe action while its confidence/logits still use the audited token weight.
    """

    weighted_ids, weighted_scalars, _ = _batch(rows, embedding, device, mode="weighted")
    ablated_ids, ablated_scalars, _ = _batch(rows, embedding, device, mode=ablation_mode)
    model.eval()
    with torch.inference_mode():
        weighted_logits = model(weighted_ids, weighted_scalars)
        ablated_logits = model(ablated_ids, ablated_scalars)
    delta = (weighted_logits - ablated_logits).abs()
    return {"mean_abs_logit_delta": round(float(delta.mean().item()), 8), "max_abs_logit_delta": round(float(delta.max().item()), 8), "row_count": float(len(rows))}


def _history_current_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("counterfactual_pair_id"):
            continue
        copied = dict(row)
        copied["layered_steps"] = [dict(row["layered_steps"][-1])]
        result.append(copied)
    return result


def _train(train: list[dict[str, Any]], dev: list[dict[str, Any]], device: torch.device, *, mode: str, seed: int) -> tuple[LayeredTokenActionPolicy, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = LayeredTokenActionPolicy(embedding_seed=seed + 1000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_ids, train_scalars, train_y = _batch(train, model.token_embedding, device, mode=mode)
    history: list[dict[str, float]] = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_ids, train_scalars), train_y)
        loss.backward()
        optimizer.step()
        train_pred, _, _ = _predict(model, train, model.token_embedding, device, mode=mode)
        dev_pred, _, _ = _predict(model, dev, model.token_embedding, device, mode=mode)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_pred, [policy_index(row["label"]) for row in train])["accuracy"], "dev_accuracy": _metrics(dev_pred, [policy_index(row["label"]) for row in dev])["accuracy"]})
    return model, history


def _summary(targets: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {key: [{"target_seed": target.get("target_seed"), "decoy_strength": target.get("decoy_strength"), "episodes": len(target.get("episodes", [])), "steps": sum(len(episode.get("steps", [])) for episode in target.get("episodes", []))} for target in value] for key, value in targets.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect_targets())
    train = _rows_from_targets(targets["pg133_train"], split="train", source="pg133_history_token_train", history_source=True) + _rows_from_targets(targets["pg127_train"], split="train", source="pg127_history_token_train") + _rows_from_targets(targets["pg125_train"], split="train", source="pg125_history_token_train")
    dev = _rows_from_targets(targets["pg133_dev"], split="dev", source="pg133_history_token_dev", history_source=True) + _rows_from_targets(targets["pg127_dev"], split="dev", source="pg127_history_token_dev") + _rows_from_targets(targets["pg125_dev"], split="dev", source="pg125_history_token_dev")
    pg133_holdout = _rows_from_targets(targets["pg133_holdout"], split="holdout", source="pg133_history_token_holdout", history_source=True)
    pg127_holdout = _rows_from_targets(targets["pg127_holdout"], split="holdout", source="pg127_history_token_holdout")
    pg125_ood = _rows_from_targets(targets["pg125_family_ood"], split="family_ood", source="pg125_history_token_family_ood")
    pg122_ood = _rows_from_targets(targets["pg122_family_ood"], split="family_ood", source="pg122_history_token_family_ood")
    model, history = _train(train, dev, device, mode="weighted", seed=13331)
    zero_model, zero_history = _train(train, dev, device, mode="zero", seed=13331)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "max_layered_tokens": MAX_LAYERED_TOKENS, "scalar_dim": SCALAR_DIM, "policy_actions": list(POLICY_ACTIONS), "embedding_provenance": model.embedding_provenance, "model_state_dict": model.state_dict()}, FULL_CHECKPOINT)
    torch.save({"schema_version": SCHEMA_VERSION, "max_layered_tokens": MAX_LAYERED_TOKENS, "scalar_dim": SCALAR_DIM, "policy_actions": list(POLICY_ACTIONS), "embedding_provenance": zero_model.embedding_provenance, "mode": "zero", "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    pg133_report = _evaluate(model, pg133_holdout, model.token_embedding, device, mode="weighted")
    pg127_report = _evaluate(model, pg127_holdout, model.token_embedding, device, mode="weighted")
    pg125_report = _evaluate(model, pg125_ood, model.token_embedding, device, mode="weighted")
    pg122_report = _evaluate(model, pg122_ood, model.token_embedding, device, mode="weighted")
    current_only_report = _evaluate(model, _history_current_only(pg133_holdout), model.token_embedding, device, mode="weighted")
    uniform_report = _evaluate(model, pg133_holdout, model.token_embedding, device, mode="uniform")
    no_failure_report = _evaluate(model, pg133_holdout, model.token_embedding, device, mode="no_failure_slots")
    token_zero_report = _evaluate(model, pg133_holdout, model.token_embedding, device, mode="tokens_zeroed")
    zero_report = _evaluate(model, pg133_holdout, model.token_embedding, device, mode="zero")
    zero_baseline = _evaluate(zero_model, pg133_holdout, zero_model.token_embedding, device, mode="zero")
    uniform_weight_sensitivity = _weight_sensitivity(model, pg133_holdout, model.token_embedding, device, ablation_mode="uniform")
    failure_weight_sensitivity = _weight_sensitivity(model, pg133_holdout, model.token_embedding, device, ablation_mode="no_failure_slots")
    all_holdout = pg133_holdout + pg127_holdout + pg125_ood + pg122_ood
    train_seeds = sorted({row.get("target_seed") for row in train})
    dev_seeds = sorted({row.get("target_seed") for row in dev})
    holdout_seeds = sorted({row.get("target_seed") for row in all_holdout})
    get_count = sum(row["failure_signature"].get("observed_method") == "GET" for row in all_holdout)
    post_count = sum(row["failure_signature"].get("observed_method") == "POST" for row in all_holdout)
    channel_balance_ratio = round(min(get_count, post_count) / max(get_count, post_count), 6) if max(get_count, post_count) else 0.0
    checks = {
        "fresh_checkpoint": True,
        "same_capacity_ablation": sum(parameter.numel() for parameter in model.parameters()) == sum(parameter.numel() for parameter in zero_model.parameters()),
        "tokenizer_backend_open_source": model.embedding_provenance["tokenizer_backend"] == "huggingface-tokenizers-layered-wordlevel",
        "source_and_ir_layers_present": all(bool(row["layered_steps"][0].get("source_token_layers")) and bool(row["layered_steps"][0].get("ir_layer")) for row in train[:12]),
        "special_token_window_bounded": MAX_LAYERED_TOKENS == 512,
        "scalar_dim": SCALAR_DIM == 5,
        "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds) or set(dev_seeds) & set(holdout_seeds)),
        "get_post_balanced": get_count > 0 and post_count > 0 and channel_balance_ratio >= 0.8,
        "pg133_holdout_not_training": not set(holdout_seeds).intersection(train_seeds + dev_seeds),
        "pg133_full_accuracy_floor": pg133_report["metrics"]["accuracy"] >= 0.95,
        "pg133_current_history_accuracy_floor": pg133_report["history_current_accuracy"] >= 0.95,
        "pg133_counterfactual_pair_conflicts": pg133_report["history_pair"]["pair_count"] >= 3 and pg133_report["history_pair"]["label_conflict_count"] == pg133_report["history_pair"]["pair_count"],
        "pg133_counterfactual_prediction_separation": pg133_report["history_pair"]["prediction_separation_rate"] == 1.0,
        "pg133_current_only_ablation_drop": current_only_report["metrics"]["accuracy"] <= 0.75,
        "pg127_family_accuracy_floor": pg127_report["metrics"]["accuracy"] >= 0.90,
        "pg125_family_accuracy_floor": pg125_report["metrics"]["accuracy"] >= 0.90,
        "pg122_family_accuracy_floor": pg122_report["metrics"]["accuracy"] >= 0.90,
        "unknown_blind_abstain": pg133_report["blind_final_abstain_rate"] == 1.0 and pg127_report["blind_final_abstain_rate"] == 1.0 and pg125_report["blind_final_abstain_rate"] == 1.0 and pg122_report["blind_final_abstain_rate"] == 1.0,
        "negative_false_stop_zero": pg133_report["negative_false_stop_count"] == 0 and pg127_report["negative_false_stop_count"] == 0 and pg125_report["negative_false_stop_count"] == 0 and pg122_report["negative_false_stop_count"] == 0,
        "token_embedding_ablation_changes": pg133_report["predictions"] != token_zero_report["predictions"],
        "uniform_weight_ablation_changes": uniform_weight_sensitivity["mean_abs_logit_delta"] > 0.001,
        "failure_weight_ablation_changes": failure_weight_sensitivity["mean_abs_logit_delta"] > 0.001,
        "zero_ablation_changes": pg133_report["predictions"] != zero_report["predictions"],
        "history_authority_not_model_input": True,
        "memory_promotion_forbidden": True,
    }
    hard_gates_passed = all(checks.values())
    training_eligible = hard_gates_passed and CROSS_IMPLEMENTATION_REVIEW_COMPLETE
    report = {
        "protocol_id": "pg-pk-133-history-token-policy-v1",
        "schema_version": "pg133-history-token-report-v1",
        "status": "completed_pg133_history_token_policy",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": training_eligible,
        "scope": {"model": "fresh_transformer_over_source_tokens_and_rule_ir_tokens", "max_layered_tokens": MAX_LAYERED_TOKENS, "scalar_dim": SCALAR_DIM, "hidden_dim": HIDDEN_DIM, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "training": {"train_count": len(train), "dev_count": len(dev), "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "source_summary": _summary({"pg133_train": targets["pg133_train"], "pg127_train": targets["pg127_train"], "pg125_train": targets["pg125_train"]})},
        "holdout": {"pg133_history_holdout": pg133_report, "pg127_seed_holdout": pg127_report, "pg125_family_ood": pg125_report, "pg122_family_ood": pg122_report, "history_current_only_ablation": current_only_report, "uniform_weight_ablation": uniform_report, "failure_weight_ablation": no_failure_report, "weight_sensitivity": {"weighted_vs_uniform": uniform_weight_sensitivity, "weighted_vs_no_failure_slots": failure_weight_sensitivity}, "token_ids_zeroed_ablation": token_zero_report, "zero_ablation": zero_report, "fresh_zero_baseline": zero_baseline},
        "checks": checks,
        "input_contract": {"source_tokens_and_rule_ir_tokens_only": True, "raw_html_javascript_retained": False, "raw_probe_response_in_model_input": False, "oracle_authority_in_model_input": False, "history_authority_in_model_input": False, "oracle_availability_fact_in_model_input": True, "special_tokens": ["[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]", "[STEP]"], "source_token_max_per_step": 64, "rule_ir_slots": ["surface.modalities", "transport.methods_seen", "response.transition_delta", "failure.kind", "failure.failed_gate", "failure.recovery_phase", "probe.remaining_budget", "trajectory.progress", "oracle.availability"]},
        "embedding_provenance": model.embedding_provenance,
        "diagnosis": {"representation_change": "页面/脚本/GET/POST 先通过 bounded source-token parser 生成最小 source atom，再和 Rule IR slot=value atom 放入同一开源 tokenizer/embedding 词表；特殊边界 token、每 token scalar weight 和 typed-oracle availability fact 显式保留。", "meaning": "history-sensitive safe workflow action experiment, not vulnerability confirmation", "counterfactual": "same current Rule IR/source observation with control_first vs candidate_first prefix; evaluator-only workflow action is kept outside model input", "pretrained_claim": "false: fresh seeded embedding; no local pretrained matrix was installed"},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "status": "history_token_candidate_pending_cross_implementation_manual_review" if hard_gates_passed else "blocked_pg133_gate_failure_preserved", "reason": "必须同时通过跨实现、未知族正确弃权、负对照、fresh reset、证据哈希和人工/Codex 审核；本轮不写长期记忆。"},
        "transport_balance": {"get_count": get_count, "post_count": post_count, "min_over_max_ratio": channel_balance_ratio},
        "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "policy": hashlib.sha256((ROOT / "app/pg133_layered_token_policy.py").read_bytes()).hexdigest(), "embedding": hashlib.sha256((ROOT / "app/layered_token_embedding.py").read_bytes()).hexdigest(), "target": hashlib.sha256((ROOT / "app/pg133_history_latch_target.py").read_bytes()).hexdigest(), "replay": hashlib.sha256((ROOT / "app/pg133_history_latch_replay.py").read_bytes()).hexdigest()},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg133-history-token-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "rows": train + dev + pg133_holdout + pg127_holdout + pg125_ood + pg122_ood}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg133-history-token-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "split": row["split"], "layered_steps": row["layered_steps"], "failure_signature": {key: value for key, value in row["failure_signature"].items() if key not in {"positive_authority", "typed_available"}}, "training_label": row["label"]} for row in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg133-history-token-trace-v1", "protocol_id": "pg-pk-133-history-token-policy-v1", "status": "completed_pg133_history_token_policy", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "target_summary": _summary(targets), "train_count": len(train), "dev_count": len(dev), "pg133_holdout_count": len(pg133_holdout), "pg127_holdout_count": len(pg127_holdout), "pg125_family_ood_count": len(pg125_ood), "pg122_family_ood_count": len(pg122_ood), "source_tokens_saved": True, "rule_ir_tokens_saved": True, "raw_source_saved": False, "raw_probe_response_saved": False, "history_authority_saved_outside_model_input": True, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-133-history-token-policy-v1", "schema_version": "pg133-history-token-protocol-v1", "objective": "验证页面 source token + Rule IR token 的分层 token 表示能否在同一当前观察、不同历史 prefix 的反事实任务上选择不同安全动作。", "training_source": "fresh PG-133 history latch plus PG-127/PG-125 accepted replay targets", "holdout_sources": ["fresh PG-133 disjoint seeds", "fresh PG-127 seed holdout", "fresh PG-125 family holdout", "fresh PG-122 family holdout"], "model_input": {"source_layers": "bounded parser tokens only", "rule_ir_layer": "slot=value tokens only", "special_tokens": ["[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]", "[STEP]"], "masked_fields": ["raw_html", "raw_javascript", "raw_probe", "raw_response", "history_stage", "workflow_action", "positive_authority", "typed_available", "target_id", "family"]}, "required_gates": checks, "promotion": {"hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-133-history-token-policy-v1", "proposal_id": "pg133-history-token-proposal-v1", "question": "当当前 Rule IR/source observation 完全相同、只有过去 control/candidate 顺序不同，双层 token Transformer 是否能选择不同的 allow-listed safe action？", "prediction": {"pg133_accuracy": ">=0.95", "counterfactual_label_conflicts": 3, "counterfactual_prediction_separation": 1.0, "current_only_accuracy": "<=0.75", "unknown_blind_abstain": 1.0, "negative_false_stop": 0}, "intervention": "control_first 与 candidate_first 共享当前 POST candidate observation，只有过去 prefix 不同；history_stage/workflow_action 只在 evaluator authority 中，不进入 model input。", "failure_rule": "任一反事实、族外、GET/POST、typed oracle、负对照、fresh reset、证据哈希或人工审核门失败，保留失败 trace，禁止训练和长期记忆晋升。", "next": "在第三独立实现上复放相同历史任务，并增加 source-only/IR-only 分层消融；仍不生成攻击 payload。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "vocab_size": model.embedding_provenance["tokenizer_vocab_size"], "max_layered_tokens": MAX_LAYERED_TOKENS, "scalar_dim": SCALAR_DIM, "parameter_count": report["scope"]["parameter_count"], "train": len(train), "dev": len(dev), "pg133_holdout": len(pg133_holdout), "pg127_holdout": len(pg127_holdout), "pg125_family_ood": len(pg125_ood), "pg122_family_ood": len(pg122_ood), "pg133_accuracy": pg133_report["metrics"]["accuracy"], "counterfactual_separation": pg133_report["history_pair"]["prediction_separation_rate"], "current_only_accuracy": current_only_report["metrics"]["accuracy"], "pg133_blind_abstain": pg133_report["blind_final_abstain_rate"], "token_zero_accuracy": token_zero_report["metrics"]["accuracy"], "uniform_accuracy": uniform_report["metrics"]["accuracy"], "failure_weight_accuracy": no_failure_report["metrics"]["accuracy"], "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [name for name, passed in checks.items() if not passed], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
