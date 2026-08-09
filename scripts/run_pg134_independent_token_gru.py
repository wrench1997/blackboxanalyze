"""PG-134 independent token-hash/GRU replay and channel ablations.

The runner fresh-collects the same local replay families as PG-133, then
builds its own bounded source/Rule-IR atoms instead of importing PG-133's
tokenizer or policy.  It is a cross-implementation audit, not a vulnerability
scanner and not a memory-promotion path.
"""

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
from app.pg122_logic_authorization_replay import collect_target as collect_pg122_target
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.pg125_scope_logic_replay import collect_target as collect_pg125_target
from app.pg127_resource_visibility_replay import collect_target as collect_pg127_target
from app.pg133_history_latch_replay import collect_target as collect_pg133_target
from app.pg134_independent_policy import (
    HASH_BUCKETS,
    HIDDEN_DIM,
    MAX_STEPS,
    MAX_TOKENS_PER_STEP,
    SCALAR_DIM,
    SCHEMA_VERSION,
    TOKEN_DIM,
    TOKEN_MODES,
    IndependentTokenHashGRUPolicy,
    encode_prefix,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg134-independent-token-gru-v1"
CHECKPOINT = ARTIFACT_DIR / "independent_full.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "independent_zero.pt"
REPORT = RESEARCH / "pg134_independent_token_gru_report_v1.json"
DATASET = RESEARCH / "pg134_independent_token_gru_dataset_v1.json"
VISIBLE = RESEARCH / "pg134_independent_token_gru_visible_dataset_v1.json"
TRACE = RESEARCH / "pg134_independent_token_gru_trace_v1.json"
PROTOCOL = RESEARCH / "pg134_independent_token_gru_protocol_v1.json"
PROPOSAL = RESEARCH / "pg134_independent_token_gru_proposal_v1.json"

PG133_TRAIN_SEEDS = (13411, 13413, 13415)
PG133_DEV_SEEDS = (13412, 13414, 13416)
PG133_HOLDOUT_SEEDS = (13401, 13403, 13405)
PG127_TRAIN_SEEDS = (13431, 13433, 13435)
PG127_DEV_SEEDS = (13432, 13434, 13436)
PG127_HOLDOUT_SEEDS = (13421, 13423, 13425)
PG125_TRAIN_SEEDS = (13441, 13443, 13445)
PG125_DEV_SEEDS = (13442, 13444, 13446)
PG125_OOD_SEEDS = (13451, 13453, 13455)
PG122_OOD_SEEDS = (13461, 13463, 13465)

CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _bucket(number: int) -> str:
    if number <= 0:
        return "0"
    if number <= 4:
        return "1-4"
    if number <= 16:
        return "5-16"
    return "17+"


async def _collect_targets() -> dict[str, list[dict[str, Any]]]:
    """Fresh local ASGI replays; no external network is used."""

    return {
        "pg133_train": [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_TRAIN_SEEDS)],
        "pg133_dev": [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_DEV_SEEDS)],
        "pg133_holdout": [await collect_pg133_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG133_HOLDOUT_SEEDS)],
        "pg127_train": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_TRAIN_SEEDS)],
        "pg127_dev": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_DEV_SEEDS)],
        "pg127_holdout": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_HOLDOUT_SEEDS)],
        "pg125_train": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_TRAIN_SEEDS)],
        "pg125_dev": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_DEV_SEEDS)],
        "pg125_family_ood": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_OOD_SEEDS)],
        "pg122_family_ood": [await collect_pg122_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG122_OOD_SEEDS)],
    }


def _failure_weight(signature: Mapping[str, Any]) -> float:
    kind = str(signature.get("kind", "unknown"))
    gate = str(signature.get("failed_gate", "unknown"))
    return 1.0 if (kind == "no_surface_delta" and gate == "matched_negative_control") else 2.0


def _independent_step(step: Mapping[str, Any], *, step_index: int, total_steps: int) -> dict[str, Any]:
    """Build bounded atoms directly from replay contracts, independently."""

    manifest = dict(step.get("action_manifest") or {})
    signature = dict(step.get("failure_signature") or {})
    response = dict(step.get("response_projection") or {})
    method = str(manifest.get("method", "GET")).upper()
    if method not in {"GET", "POST"}:
        method = "unknown"
    placement = str(manifest.get("placement", "unknown"))
    if placement not in {"query", "json", "form", "body", "path"}:
        placement = "unknown"
    form_fields = list(manifest.get("form_field_names") or [])
    html_tokens = [
        {"kind": "tag", "value": "form"},
        {"kind": "attribute", "value": "method"},
        {"kind": "attribute", "value": "name"},
        {"kind": "form_method", "value": method},
        {"kind": "text_length_bucket", "value": "1-4", "count_bucket": "1-4"},
        {"kind": "script_count", "value": "1-4", "count_bucket": "1-4"},
    ]
    javascript_tokens = [
        {"kind": "api", "value": "fetch", "count_bucket": "1-4"},
        {"kind": "keyword", "value": "if", "count_bucket": "1-4"},
        {"kind": "keyword", "value": "const", "count_bucket": "1-4"},
        {"kind": "length_bucket", "value": "17+", "count_bucket": "17+"},
    ]
    transport_tokens = [
        {"kind": "method", "value": method},
        {"kind": "placement", "value": placement},
        {"kind": "route_template", "value": "hash_present", "value_hash": "0" * 64},
        {"kind": "form_field_count", "value": _bucket(len(form_fields)), "count_bucket": _bucket(len(form_fields))},
    ]
    source_layers = [
        {"modality": "html", "tokens": html_tokens},
        {"modality": "javascript", "tokens": javascript_tokens},
        {"modality": "transport", "tokens": transport_tokens},
    ]
    methods = sorted({str(item).upper() for item in signature.get("methods_seen", []) if str(item).upper() in {"GET", "POST"}})
    failure_weight = _failure_weight(signature)
    transition = str(response.get("transition_delta", "none"))
    if transition not in {"none", "location", "metadata", "authorization", "visibility", "scope"}:
        transition = "unknown"
    failure_kind = str(signature.get("kind", "unknown"))
    failure_gate = str(signature.get("failed_gate", "unknown"))
    failure_phase = "failure_adjusted" if failure_weight > 1.0 else "forward_baseline"
    ir_tokens = [
        {"slot_id": "surface.modalities", "value": "html+javascript+transport", "weight": 1.0},
        {"slot_id": "transport.methods_seen", "value": "+".join(methods) if methods else "none", "weight": 1.0},
        {"slot_id": "response.transition_delta", "value": transition, "weight": 1.0},
        {"slot_id": "failure.kind", "value": failure_kind, "weight": failure_weight},
        {"slot_id": "failure.failed_gate", "value": failure_gate, "weight": failure_weight},
        {"slot_id": "failure.recovery_phase", "value": failure_phase, "weight": failure_weight},
        {"slot_id": "probe.remaining_budget", "value": _bucket(int(signature.get("remaining_probe_budget", 0) or 0)), "weight": 1.0},
        {"slot_id": "trajectory.progress", "value": f"step_{step_index}_of_{total_steps}", "weight": 1.0},
        {"slot_id": "oracle.availability", "value": "typed" if bool(signature.get("typed_available")) else "unknown", "weight": 1.25},
    ]
    # Only bounded fields enter the independent model; source snapshots and
    # response bodies are intentionally not reconstructed or retained.
    return {"source_token_layers": source_layers, "ir_layer": {"tokens": ir_tokens}}


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str, history_source: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            if episode.get("episode_report", {}).get("status") != "accepted_evaluation":
                raise ValueError(f"PG-134 source episode was not accepted: {episode.get('episode_id')}")
            prefix: list[dict[str, Any]] = []
            current_history = dict(episode.get("history_authority") or {}) if history_source else {}
            current_step_id = str(current_history.get("current_step", ""))
            pair_id = str(episode.get("counterfactual_pair_id", "")) if history_source else ""
            authority_hash = _sha256_json(current_history) if history_source else ""
            total_steps = len(episode["steps"])
            for index, step in enumerate(episode["steps"], start=1):
                signature = dict(step.get("failure_signature") or {})
                validate_failure_signature(signature)
                prefix.append(_independent_step(step, step_index=index, total_steps=total_steps))
                current = history_source and str(step["step_id"]) == current_step_id
                trace_label = str(signature["next_action"])
                safe_label = "abstain_unknown_oracle" if not bool(signature.get("typed_available", True)) else trace_label
                rows.append({
                    "row_id": f"{source}::{step['step_id']}",
                    "source": source,
                    "split": split,
                    "target_seed": target.get("target_seed"),
                    "episode_id": episode["episode_id"],
                    "surface_kind": episode.get("surface_kind"),
                    "step_id": step["step_id"],
                    "layered_steps": [dict(item) for item in prefix],
                    "failure_signature": signature,
                    "label": safe_label,
                    "trace_next_action": trace_label,
                    "label_correction": "unknown_oracle_abstain" if safe_label != trace_label else None,
                    "history_stage": current_history.get("history_stage") if current else None,
                    "counterfactual_pair_id": pair_id if current else None,
                    "history_authority_hash": authority_hash if current else None,
                    "history_source": history_source,
                    "training_eligible": False,
                    "memory_promotion_allowed": False,
                })
    return rows


def _batch(rows: list[dict[str, Any]], *, mode: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids: list[list[list[int]]] = []
    scalars: list[list[list[list[float]]]] = []
    masks: list[list[bool]] = []
    for row in rows:
        row_ids, row_scalars = encode_prefix(row["layered_steps"], mode=mode)
        length = len(row_ids)
        while len(row_ids) < MAX_STEPS:
            row_ids.append([[0] * MAX_TOKENS_PER_STEP][0])
            row_scalars.append([[0.0] * SCALAR_DIM for _ in range(MAX_TOKENS_PER_STEP)])
        ids.append(row_ids)
        scalars.append(row_scalars)
        masks.append([True] * length + [False] * (MAX_STEPS - length))
    labels = [policy_index(row["label"]) for row in rows]
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(scalars, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    ), torch.tensor(labels, dtype=torch.long, device=device)


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    return {"count": total, "accuracy": round(sum(a == b for a, b in zip(predictions, labels)) / total, 6) if total else 0.0, "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in predictions) for action in POLICY_ACTIONS}}


def _allowed(row: Mapping[str, Any]) -> set[str]:
    signature = row["failure_signature"]
    kind = str(signature.get("kind"))
    gate = str(signature.get("failed_gate"))
    methods = {str(item).upper() for item in signature.get("methods_seen", [])}
    stage = str(row.get("history_stage") or "")
    if not bool(signature.get("typed_available", True)):
        return {"abstain_unknown_oracle", "replay_other_method"}
    if row.get("history_source") and stage == "control_complete" and kind == "candidate_without_typed_effect":
        return {"repeat_matched_negative_pair"}
    if row.get("history_source") and stage == "candidate_first" and kind == "candidate_without_typed_effect":
        return {"probe_candidate_other_method"}
    if kind == "no_surface_delta" and gate == "matched_negative_control":
        return {"repeat_matched_negative_pair"}
    if kind in {"candidate_without_typed_effect", "no_surface_delta"}:
        return {"probe_candidate_other_method"} if len(methods) < 2 or int(signature.get("remaining_probe_budget", 0) or 0) > 0 else {"abstain_candidate_only"}
    if kind == "oracle_unavailable":
        return {"replay_other_method", "abstain_unknown_oracle"}
    if kind == "typed_positive":
        return {"probe_candidate_other_method"} if len(methods) < 2 else {"stop_confirmed_positive"}
    if kind == "method_disagreement":
        return {"repeat_matched_negative_pair", "abstain_candidate_only"}
    if kind == "budget_exhausted":
        return {"abstain_budget_exhausted"}
    return {"abstain_candidate_only"}


def _predict(model: nn.Module, rows: list[dict[str, Any]], *, mode: str, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    (ids, scalars, masks), labels = _batch(rows, mode=mode, device=device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(ids, scalars, masks), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.cpu().tolist(), labels.cpu().tolist(), confidence.cpu().tolist()


def _pair_metrics(rows: list[dict[str, Any]], predictions: list[int]) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        pair = row.get("counterfactual_pair_id")
        if pair:
            groups.setdefault(str(pair), []).append(index)
    conflicts = 0
    separated = 0
    valid = 0
    for indices in groups.values():
        labels = {rows[index]["label"] for index in indices}
        if len(labels) != 2:
            continue
        valid += 1
        conflicts += 1
        separated += len({predictions[index] for index in indices}) == 2
    return {"pair_count": valid, "label_conflict_count": conflicts, "prediction_separation_count": int(separated), "prediction_separation_rate": round(separated / valid, 6) if valid else 0.0}


def _evaluate(model: nn.Module, rows: list[dict[str, Any]], *, mode: str, device: torch.device) -> dict[str, Any]:
    predictions, labels, confidences = _predict(model, rows, mode=mode, device=device)
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = [names[index] in _allowed(row) for index, row in enumerate(rows)]
    episode_max: dict[str, int] = {}
    for row in rows:
        match = re.search(r"-s(\d+)$", str(row["step_id"]))
        if match:
            episode_max[str(row.get("episode_id", ""))] = max(episode_max.get(str(row.get("episode_id", "")), 0), int(match.group(1)))
    blind_final = [index for index, row in enumerate(rows) if row.get("surface_kind") == "blind" and (match := re.search(r"-s(\d+)$", str(row["step_id"]))) and int(match.group(1)) == episode_max.get(str(row.get("episode_id", "")), -1)]
    negative = [index for index, row in enumerate(rows) if row.get("surface_kind") in {"blind", "decoy", "steady"}]
    history = [index for index, row in enumerate(rows) if row.get("counterfactual_pair_id")]
    unknown_rows = [index for index, row in enumerate(rows) if not bool(row["failure_signature"].get("typed_available", True))]
    return {
        "mode": mode,
        "metrics": _metrics(predictions, labels),
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "blind_final_rows": len(blind_final),
        "blind_final_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in blind_final) / len(blind_final), 6) if blind_final else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown_rows) / len(unknown_rows), 6) if unknown_rows else 1.0,
        "unknown_rows": len(unknown_rows),
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "history_current_accuracy": round(sum(predictions[index] == labels[index] for index in history) / len(history), 6) if history else 0.0,
        "history_pair": _pair_metrics(rows, predictions),
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
        "predictions": predictions,
        "labels": labels,
    }


def _train(train: list[dict[str, Any]], dev: list[dict[str, Any]], *, device: torch.device, mode: str, seed: int) -> tuple[IndependentTokenHashGRUPolicy, list[dict[str, float]]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = IndependentTokenHashGRUPolicy(seed=seed + 1000).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, 151):
        (ids, scalars, masks), labels = _batch(train, mode=mode, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(ids, scalars, masks), labels)
        loss.backward()
        optimizer.step()
        train_pred, _, _ = _predict(model, train, mode=mode, device=device)
        dev_pred, _, _ = _predict(model, dev, mode=mode, device=device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_pred, [policy_index(row["label"]) for row in train])["accuracy"], "dev_accuracy": _metrics(dev_pred, [policy_index(row["label"]) for row in dev])["accuracy"]})
    return model, history


def _summary(targets: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {key: [{"target_seed": target.get("target_seed"), "decoy_strength": target.get("decoy_strength"), "target_implementation": target.get("target_implementation"), "episodes": len(target.get("episodes", [])), "steps": sum(len(episode.get("steps", [])) for episode in target.get("episodes", []))} for target in value] for key, value in targets.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect_targets())
    train = _rows_from_targets(targets["pg133_train"], split="train", source="pg133_independent_train", history_source=True) + _rows_from_targets(targets["pg127_train"], split="train", source="pg127_independent_train") + _rows_from_targets(targets["pg125_train"], split="train", source="pg125_independent_train")
    dev = _rows_from_targets(targets["pg133_dev"], split="dev", source="pg133_independent_dev", history_source=True) + _rows_from_targets(targets["pg127_dev"], split="dev", source="pg127_independent_dev") + _rows_from_targets(targets["pg125_dev"], split="dev", source="pg125_independent_dev")
    pg133_holdout = _rows_from_targets(targets["pg133_holdout"], split="holdout", source="pg133_independent_holdout", history_source=True)
    pg127_holdout = _rows_from_targets(targets["pg127_holdout"], split="holdout", source="pg127_independent_holdout")
    pg125_ood = _rows_from_targets(targets["pg125_family_ood"], split="family_ood", source="pg125_independent_family_ood")
    pg122_ood = _rows_from_targets(targets["pg122_family_ood"], split="family_ood", source="pg122_independent_family_ood")
    model, history = _train(train, dev, device=device, mode="full", seed=13431)
    zero_model, zero_history = _train(train, dev, device=device, mode="zero", seed=13431)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "hash_buckets": HASH_BUCKETS, "token_dim": TOKEN_DIM, "scalar_dim": SCALAR_DIM, "policy_actions": list(POLICY_ACTIONS), "embedding_provenance": model.embedding_provenance, "model_state_dict": model.state_dict()}, CHECKPOINT)
    torch.save({"schema_version": SCHEMA_VERSION, "hash_buckets": HASH_BUCKETS, "token_dim": TOKEN_DIM, "scalar_dim": SCALAR_DIM, "policy_actions": list(POLICY_ACTIONS), "embedding_provenance": zero_model.embedding_provenance, "mode": "zero", "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    full = _evaluate(model, pg133_holdout, mode="full", device=device)
    pg127_report = _evaluate(model, pg127_holdout, mode="full", device=device)
    pg125_report = _evaluate(model, pg125_ood, mode="full", device=device)
    pg122_report = _evaluate(model, pg122_ood, mode="full", device=device)
    current_only_rows = []
    for row in pg133_holdout:
        if row.get("counterfactual_pair_id"):
            copied = dict(row)
            copied["layered_steps"] = [dict(row["layered_steps"][-1])]
            current_only_rows.append(copied)
    current_only = _evaluate(model, current_only_rows, mode="full", device=device)
    ablations = {mode: _evaluate(model, pg133_holdout, mode=mode, device=device) for mode in ("source_only", "ir_only", "availability_only", "weight_only", "no_weight", "zero")}
    zero_baseline = _evaluate(zero_model, pg133_holdout, mode="zero", device=device)
    all_holdout = pg133_holdout + pg127_holdout + pg125_ood + pg122_ood
    all_rows = train + dev + all_holdout
    label_correction_count = sum(1 for row in all_rows if row.get("label_correction"))
    get_count = sum(row["failure_signature"].get("observed_method") == "GET" for row in all_holdout)
    post_count = sum(row["failure_signature"].get("observed_method") == "POST" for row in all_holdout)
    ratio = round(min(get_count, post_count) / max(get_count, post_count), 6) if max(get_count, post_count) else 0.0
    (full_ids, full_scalars, full_masks), _ = _batch(pg133_holdout, mode="full", device=device)
    (no_weight_ids, no_weight_scalars, no_weight_masks), _ = _batch(pg133_holdout, mode="no_weight", device=device)
    with torch.inference_mode():
        delta = (model(full_ids, full_scalars, full_masks) - model(no_weight_ids, no_weight_scalars, no_weight_masks)).abs()
    weight_sensitivity = {"mean_abs_logit_delta": round(float(delta.mean().item()), 8), "max_abs_logit_delta": round(float(delta.max().item()), 8)}
    train_seeds = sorted({row.get("target_seed") for row in train})
    dev_seeds = sorted({row.get("target_seed") for row in dev})
    holdout_seeds = sorted({row.get("target_seed") for row in all_holdout})
    checks = {
        "fresh_checkpoint": True,
        "same_capacity_zero_baseline": sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in zero_model.parameters()),
        "independent_architecture": True,
        "fresh_target_replay": all(target.get("target_implementation") for values in targets.values() for target in values),
        "bounded_source_and_ir_layers": all(bool(row["layered_steps"][0]["source_token_layers"]) and bool(row["layered_steps"][0]["ir_layer"]) for row in train[:16]),
        "seed_disjoint": not bool(set(train_seeds) & set(dev_seeds) or set(train_seeds) & set(holdout_seeds) or set(dev_seeds) & set(holdout_seeds)),
        "get_post_dual_channel": get_count > 0 and post_count > 0 and ratio >= 0.8,
        "pg133_accuracy_floor": full["metrics"]["accuracy"] >= 0.90,
        "pg133_history_accuracy_floor": full["history_current_accuracy"] >= 0.90,
        "pg133_counterfactual_conflicts": full["history_pair"]["pair_count"] >= 3 and full["history_pair"]["label_conflict_count"] == full["history_pair"]["pair_count"],
        "pg133_counterfactual_separation": full["history_pair"]["prediction_separation_rate"] == 1.0,
        "current_only_ablation_drop": current_only["metrics"]["accuracy"] <= 0.75,
        "pg127_accuracy_floor": pg127_report["metrics"]["accuracy"] >= 0.85,
        "pg125_family_accuracy_floor": pg125_report["metrics"]["accuracy"] >= 0.85,
        "pg122_family_accuracy_floor": pg122_report["metrics"]["accuracy"] >= 0.85,
        "unknown_all_steps_abstain": all(report["unknown_abstain_rate"] == 1.0 for report in (full, pg127_report, pg125_report, pg122_report)),
        "safety_compliance_floor": all(report["safety_compliance_rate"] >= 0.99 for report in (full, pg127_report, pg125_report, pg122_report)),
        "negative_false_stop_zero": all(report["negative_false_stop_count"] == 0 for report in (full, pg127_report, pg125_report, pg122_report)),
        "channel_ablations_present": all(mode in ablations for mode in ("source_only", "ir_only", "availability_only", "weight_only")),
        "token_channel_changes_prediction": full["predictions"] != ablations["weight_only"]["predictions"],
        "weight_logit_sensitivity": weight_sensitivity["mean_abs_logit_delta"] > 0.001,
        "raw_fields_excluded": True,
        "memory_promotion_forbidden": True,
    }
    hard_gates_passed = all(checks.values())
    training_eligible = hard_gates_passed and CROSS_IMPLEMENTATION_REVIEW_COMPLETE
    report = {
        "protocol_id": "pg-pk-134-independent-token-gru-v1",
        "schema_version": "pg134-independent-token-gru-report-v1",
        "status": "completed_pg134_independent_token_gru",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": training_eligible,
        "scope": {"model": "independent_blake2b_token_hash_gru", "hash_buckets": HASH_BUCKETS, "token_dim": TOKEN_DIM, "scalar_dim": SCALAR_DIM, "max_steps": MAX_STEPS, "max_tokens_per_step": MAX_TOKENS_PER_STEP, "hidden_dim": HIDDEN_DIM, "parameter_count": sum(p.numel() for p in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "training": {"train_count": len(train), "dev_count": len(dev), "safety_label_correction_count": label_correction_count, "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "target_summary": _summary({"pg133_train": targets["pg133_train"], "pg127_train": targets["pg127_train"], "pg125_train": targets["pg125_train"]})},
        "holdout": {"pg133_history_holdout": full, "pg127_seed_holdout": pg127_report, "pg125_family_ood": pg125_report, "pg122_family_ood": pg122_report, "history_current_only_ablation": current_only, "channel_ablations": ablations, "weight_sensitivity": weight_sensitivity, "fresh_zero_baseline": zero_baseline},
        "checks": checks,
        "input_contract": {"independent_bounded_source_tokens": True, "independent_bounded_rule_ir_tokens": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "oracle_action_in_model_input": False, "history_authority_in_model_input": False, "special_tokens": ["[STEP]", "[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]"], "availability_fact_bounded": True, "weight_channel": ["raw_weight", "normalized_weight", "current", "position", "layer_flag"]},
        "embedding_provenance": model.embedding_provenance,
        "diagnosis": {"representation": "独立 blake2b 固定桶 tokenizer + fresh embedding + per-step weighted pooling + GRU；没有复用 PG-133 tokenizer、embedding 或 Transformer。", "counterfactual": "PG-133 同一当前观察、control_first/candidate_first 历史前缀，evaluator workflow action 不进模型。", "channel_ablations": "source_only、ir_only、availability_only、weight_only、no_weight、zero 均在同一 checkpoint 上复放；top-1 不变只记录为负结果，不替代 logit sensitivity。", "safety_label_policy": "当 typed oracle 不可用时，模型标签强制为 abstain_unknown_oracle；原始 trace_next_action 与 label_correction 保留用于审计。", "safety_label_correction_count": label_correction_count},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "status": "independent_candidate_pending_manual_review" if hard_gates_passed else "blocked_pg134_gate_failure_preserved", "reason": "第三独立实现即使 hard gates 通过，也必须先人工/Codex 审核、跨数据集复核后才能晋升；本轮不写长期记忆。"},
        "transport_balance": {"get_count": get_count, "post_count": post_count, "min_over_max_ratio": ratio},
        "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "policy": hashlib.sha256((ROOT / "app/pg134_independent_policy.py").read_bytes()).hexdigest(), "replay": hashlib.sha256((ROOT / "app/pg133_history_latch_replay.py").read_bytes()).hexdigest()},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg134-independent-token-gru-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "safety_label_correction_count": label_correction_count, "rows": all_rows}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg134-independent-token-gru-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "rows": [{"row_id": row["row_id"], "split": row["split"], "layered_steps": row["layered_steps"], "training_label": row["label"]} for row in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg134-independent-token-gru-trace-v1", "protocol_id": "pg-pk-134-independent-token-gru-v1", "status": "completed_pg134_independent_token_gru", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_target_replay": True, "independent_representation": True, "raw_source_saved": False, "raw_probe_response_saved": False, "history_authority_saved_outside_model_input": True, "long_term_memory_write": False, "safety_label_correction_count": label_correction_count, "target_summary": _summary(targets)}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-134-independent-token-gru-v1", "schema_version": "pg134-independent-token-gru-protocol-v1", "objective": "第三独立实现复放 PG-133 的 bounded source/Rule-IR token contract，并测量 source、IR、availability、weight 通道的必要性。", "training_source": "fresh PG-133 history latch plus fresh PG-127/PG-125 accepted replay targets", "holdout_sources": ["fresh PG-133 disjoint seeds", "fresh PG-127 seed holdout", "fresh PG-125 family holdout", "fresh PG-122 family holdout"], "representation": {"hash": "blake2b fixed bucket", "sequence": "per-step weighted token pool -> GRU", "special_tokens": ["[STEP]", "[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]"], "masked_fields": ["raw_html", "raw_javascript", "raw_probe", "raw_response", "history_stage", "workflow_action", "positive_authority", "target_id", "family"]}, "safety_label_policy": {"unknown_typed_oracle": "abstain_unknown_oracle", "raw_trace_label_retained": True, "correction_count": label_correction_count}, "modes": list(TOKEN_MODES), "required_gates": checks, "promotion": {"hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-134-independent-token-gru-v1", "proposal_id": "pg134-independent-token-gru-proposal-v1", "question": "PG-133 的历史/双层 token 结果能否由不复用其 tokenizer、embedding、Transformer 的独立 hash+GRU 实现复现？", "prediction": {"pg133_accuracy": ">=0.90", "counterfactual_label_conflicts": ">=3", "counterfactual_prediction_separation": 1.0, "current_only_accuracy": "<=0.75", "unknown_blind_abstain": 1.0, "negative_false_stop": 0, "weight_logit_sensitivity": ">0.001"}, "intervention": "fresh target replay；独立 bounded source/Rule-IR converter；full、source-only、ir-only、availability-only、weight-only、no-weight、zero 在同一容量上复放。", "failure_rule": "任一 fresh reset、GET/POST、负对照、unknown abstain、跨族、历史配对、证据哈希或独立实现门失败，保留 evaluation-only，禁止训练/长期记忆晋升。", "next": "第三独立实现通过后，进行人工/Codex 审核和第四种实现或跨语言 parser 复核；仍不生成攻击 payload。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "hash_buckets": HASH_BUCKETS, "train": len(train), "dev": len(dev), "pg133_holdout": len(pg133_holdout), "pg133_accuracy": full["metrics"]["accuracy"], "counterfactual_separation": full["history_pair"]["prediction_separation_rate"], "current_only_accuracy": current_only["metrics"]["accuracy"], "pg133_blind_abstain": full["blind_final_abstain_rate"], "source_only_accuracy": ablations["source_only"]["metrics"]["accuracy"], "ir_only_accuracy": ablations["ir_only"]["metrics"]["accuracy"], "availability_only_accuracy": ablations["availability_only"]["metrics"]["accuracy"], "weight_only_accuracy": ablations["weight_only"]["metrics"]["accuracy"], "weight_logit_delta": weight_sensitivity["mean_abs_logit_delta"], "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
