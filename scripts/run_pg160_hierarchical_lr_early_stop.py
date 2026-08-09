"""PG-160: hierarchical learning rates and dev-risk early stopping.

This run starts from the PG-159 shared family-invariant adapter checkpoint.
The large MoE body is either frozen or updated at a smaller learning rate than
the adapter/action heads.  A dev-only risk/coverage score selects the best
epoch; no holdout is used for early stopping.  The purpose is to reduce the
high raw false-stop rate observed in PG-159 without giving up its strong
surface-language preservation.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg148_large_model_posttraining as pg148  # noqa: E402
import run_pg154_multisource_action_replay as pg154  # noqa: E402
import run_pg156_calibration_surface_anchor as pg156  # noqa: E402
import run_pg159_family_invariant_adapter as pg159  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg160-hierarchical-lr-early-stop-v1"
REPORT = RESEARCH / "pg160_hierarchical_lr_early_stop_report_v1.json"
DATASET = RESEARCH / "pg160_hierarchical_lr_early_stop_dataset_v1.json"
PROTOCOL = RESEARCH / "pg160_hierarchical_lr_early_stop_protocol_v1.json"
SOURCE_CHECKPOINT = ROOT / "artifacts" / "pg159-family-invariant-adapter-v1" / "shared_adapter_256.pt"
SEED = 16001
MAX_EPOCHS = 3
PATIENCE = 1
LM_WEIGHT = 0.50
SURFACE_WEIGHT = 0.50
ADVERSARIAL_WEIGHT = 0.05
FALSE_STOP_WEIGHT = 0.20
UNKNOWN_WEIGHT = 0.20
ADVERSARIAL_LAMBDA = 0.10
BOTTLENECK = 256
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_source(vocabulary: pg148._Vocabulary, device: torch.device, family_count: int) -> pg159._FamilyInvariantAdapter:
    model = pg159._load_model(vocabulary, device, BOTTLENECK, family_count)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _dev_selection(model: pg159._FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    raw = pg159._action_metrics(model, rows, vocabulary, device)
    temperature, threshold = pg159._fit_calibration(model, rows, vocabulary, device)
    calibrated = pg159._calibrated(model, rows, vocabulary, device, temperature, threshold)
    key = (int(calibrated["false_stop_count"] == 0), calibrated["coverage"], calibrated["selective_accuracy"], -temperature)
    return {"raw": raw, "temperature": temperature, "threshold": threshold, "calibrated": calibrated, "selection_key": key}


def _evaluate(name: str, model: pg159._FamilyInvariantAdapter, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, family_to_index: dict[str, int], history: list[dict[str, Any]], started: float, train_count: int, config: dict[str, Any], best_epoch: int, best_dev: dict[str, Any]) -> dict[str, Any]:
    temperature, threshold = pg159._fit_calibration(model, data["action_dev"], vocabulary, device)
    family_metrics = {family: pg159._action_metrics(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == family], vocabulary, device) for family in sorted({str(row.get("surface_kind")) for row in data["real_holdout"]})}
    family_calibrated = {family: pg159._calibrated(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == family], vocabulary, device, temperature, threshold) for family in family_metrics}
    before = pg156._load_anchor(vocabulary, device)
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {"variant": name, "config": config, "best_epoch": best_epoch, "train_count": train_count, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "temperature": temperature, "abstain_threshold": threshold, "best_dev_selection": best_dev, "synthetic_holdout": pg159._action_metrics(model, data["synthetic_holdout"], vocabulary, device), "real_pg136_holdout": pg159._action_metrics(model, data["real_holdout"], vocabulary, device), "unseen_authorization_family_holdout": pg159._action_metrics(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"], vocabulary, device), "calibrated_synthetic_holdout": pg159._calibrated(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold), "calibrated_real_pg136_holdout": pg159._calibrated(model, data["real_holdout"], vocabulary, device, temperature, threshold), "calibrated_unseen_authorization_family_holdout": pg159._calibrated(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"], vocabulary, device, temperature, threshold), "family_holdout_metrics": family_metrics, "family_holdout_calibrated": family_calibrated, "family_adversary_on_real_holdout": pg159._family_adversary_metrics(model, data["real_holdout"], vocabulary, device, family_to_index), "surface_unknown": pg159._action_metrics(model, data["surface_unknown"], vocabulary, device), "surface_lm": pg159._lm_metrics(model, data["surface_lm"], vocabulary, device), "language_canary": canary, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg160-hierarchical-lr-early-stop-v1", "variant": name, "config": config, "best_epoch": best_epoch, "temperature": temperature, "abstain_threshold": threshold, "family_to_index": family_to_index, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, config: dict[str, Any], data: dict[str, list[dict[str, Any]]], family_to_index: dict[str, int], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    model = _load_source(vocabulary, device, len(family_to_index))
    if bool(config.get("freeze_body", False)):
        for parameter in model.base.parameters():
            parameter.requires_grad = False
    train_rows = list(data["action_train"]) + list(data["lm_replay"]) + list(data["balanced_replay"]) + list(data["unknown_hints"]) + list(data["surface_anchor"])
    loader = DataLoader(pg159._InvariantDataset(train_rows, vocabulary, family_to_index), batch_size=32, shuffle=True, collate_fn=pg159._collate)
    head_parameters = list(model.adapter.parameters()) + list(model.action_head.parameters()) + list(model.family_head.parameters())
    parameter_groups = [{"params": head_parameters, "lr": float(config["head_lr"]) }]
    if not bool(config.get("freeze_body", False)):
        parameter_groups.append({"params": [parameter for parameter in model.base.parameters() if parameter.requires_grad], "lr": float(config["body_lr"])})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dev: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(config.get("max_epochs", MAX_EPOCHS)) + 1):
        model.train()
        sums = {"loss": 0.0, "action": 0.0, "lm": 0.0, "surface": 0.0, "family": 0.0, "false_stop": 0.0, "unknown": 0.0}
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            family_indices = batch["family_indices"].to(device)
            unknown_hint = batch["unknown_hint"].to(device)
            surface_mask = torch.tensor([bool(row.get("surface_anchor", False)) for row in batch["rows"]], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                hidden, auxiliary, _ = model.base.encode(ids, mask)
                lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
                last = hidden[torch.arange(hidden.shape[0], device=device), lengths - 1]
                pooled = last + 0.50 * model.adapter(last)
                action_logits = model.action_head(pooled)
                action_loss = nn.functional.cross_entropy(action_logits[action_valid], labels[action_valid]) if bool(action_valid.any()) else action_logits.new_zeros(())
                fam_valid = action_valid & family_indices.ge(0)
                family_loss = action_logits.new_zeros(())
                if bool(fam_valid.any()):
                    family_loss = nn.functional.cross_entropy(model.family_logits(pooled[fam_valid], adversarial=True), family_indices[fam_valid])
                lm_logits, _ = pg154.pg152._lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                surface_loss = action_logits.new_zeros(())
                if bool(surface_mask.any()):
                    surface_loss = nn.functional.cross_entropy(lm_logits[surface_mask].reshape(-1, lm_logits.shape[-1]), targets[surface_mask].reshape(-1), ignore_index=0)
                non_stop = action_valid & labels.ne(STOP_INDEX)
                false_stop_loss = action_logits.new_zeros(())
                if bool(non_stop.any()):
                    alternatives = action_logits.clone()
                    alternatives[:, STOP_INDEX] = torch.finfo(alternatives.dtype).min
                    false_stop_loss = nn.functional.softplus(action_logits[:, STOP_INDEX] - alternatives.max(dim=-1).values)[non_stop].mean()
                unknown_loss = action_logits.new_zeros(())
                if bool(unknown_hint.any()):
                    unknown_loss = nn.functional.cross_entropy(action_logits[unknown_hint], torch.full((int(unknown_hint.sum().item()),), UNKNOWN_INDEX, dtype=torch.long, device=device))
                loss = action_loss + LM_WEIGHT * lm_loss + 0.01 * auxiliary + ADVERSARIAL_WEIGHT * family_loss + FALSE_STOP_WEIGHT * false_stop_loss + UNKNOWN_WEIGHT * unknown_loss + SURFACE_WEIGHT * surface_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["action"] += float(action_loss.item())
            sums["lm"] += float(lm_loss.item())
            sums["surface"] += float(surface_loss.item())
            sums["family"] += float(family_loss.item())
            sums["false_stop"] += float(false_stop_loss.item())
            sums["unknown"] += float(unknown_loss.item())
        dev = _dev_selection(model, data["action_dev"], vocabulary, device)
        family_dev = pg159._family_adversary_metrics(model, data["action_dev"], vocabulary, device, family_to_index)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "lm_loss": round(sums["lm"] / max(len(loader), 1), 8), "surface_lm_loss": round(sums["surface"] / max(len(loader), 1), 8), "family_loss": round(sums["family"] / max(len(loader), 1), 8), "false_stop_loss": round(sums["false_stop"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev, "family_adversary_dev": family_dev})
        if best_key is None or tuple(dev["selection_key"]) > best_key:
            best_key = tuple(dev["selection_key"])
            best_dev = dev
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(json.dumps({"variant": name, "epoch": epoch, "dev_accuracy": dev["raw"]["accuracy"], "dev_false_stop": dev["raw"]["false_stop_count"], "dev_calibrated_false_stop": dev["calibrated"]["false_stop_count"], "dev_coverage": dev["calibrated"]["coverage"], "family_adversary_dev_accuracy": family_dev["accuracy"]}, ensure_ascii=False), flush=True)
        if bool(config.get("early_stop", False)) and stale_epochs >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    result = _evaluate(name, model, data, vocabulary, device, family_to_index, history, started, len(train_rows), config, best_epoch, best_dev or {})
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = pg148._Vocabulary(list(json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))["vocabulary"]))
    data, family_to_index, source = pg159._build_data(vocabulary, None)
    configs = [("shared_adapter_source", {"freeze_body": True, "head_lr": 0.0, "body_lr": 0.0, "max_epochs": 0, "early_stop": False}), ("adapter_only_2e", {"freeze_body": True, "head_lr": 1.0e-4, "body_lr": 0.0, "max_epochs": 2, "early_stop": False}), ("body_lr_1e5_2e", {"freeze_body": False, "head_lr": 1.0e-4, "body_lr": 1.0e-5, "max_epochs": 2, "early_stop": False}), ("body_lr_3e5_early", {"freeze_body": False, "head_lr": 1.0e-4, "body_lr": 3.0e-5, "max_epochs": 3, "early_stop": True})]
    results = []
    source_model = _load_source(vocabulary, device, len(family_to_index))
    source_result = _evaluate("shared_adapter_source", source_model, data, vocabulary, device, family_to_index, [], time.perf_counter(), 0, configs[0][1], 0, {})
    del source_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results.append(source_result)
    for name, config in configs[1:]:
        print(json.dumps({"status": "starting_variant", "variant": name, "config": config, "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train(name, config, data, family_to_index, vocabulary, device))
    source = {**source, "source_checkpoint": str(SOURCE_CHECKPOINT.relative_to(ROOT)), "family_labels_as_model_inputs": False, "family_labels_for_adversary_only": True, "variant_count": len(configs)}
    report = {"protocol_id": "pg-pk-160-hierarchical-lr-early-stop-v1", "schema_version": "pg160-hierarchical-lr-early-stop-report-v1", "status": "completed_pg160_hierarchical_lr_early_stop", "device": str(device), "seed": SEED, "source": source, "variants": results, "objective": "hierarchical_body_adapter_learning_rate_dev_risk_early_stop", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "family_labels_as_model_inputs": False, "unseen_authorization_family_training_used": False, "holdout_labels_in_training": False, "calibration_fit_on_holdout": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg160-hierarchical-lr-early-stop-dataset-v1", "source": source, "variants": {name: config for name, config in configs}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "authorization_family": sum(str(row.get("surface_kind")) == "authorization" for row in data["real_holdout"]), "surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-160-hierarchical-lr-early-stop-v1", "schema_version": "pg160-hierarchical-lr-early-stop-protocol-v1", "objective": report["objective"], "variants": [{"name": name, "config": config} for name, config in configs], "source_checkpoint": "PG-159 shared_adapter_256", "learning_rate_groups": {"body": "1e-5 or 3e-5", "adapter_action_family_heads": "1e-4", "frozen_body_control": True}, "early_stop": {"selection_metric": "dev calibrated zero-false-stop lexicographic coverage/selective_accuracy", "patience": PATIENCE, "holdout_used": False}, "loss": {"lm_weight": LM_WEIGHT, "surface_weight": SURFACE_WEIGHT, "adversarial_weight": ADVERSARIAL_WEIGHT, "false_stop_weight": FALSE_STOP_WEIGHT, "unknown_weight": UNKNOWN_WEIGHT}, "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "best_epoch": row["best_epoch"], "synthetic_raw_false_stop": row["synthetic_holdout"]["false_stop_count"], "synthetic_calibrated_false_stop": row["calibrated_synthetic_holdout"]["false_stop_count"], "real_raw_false_stop": row["real_pg136_holdout"]["false_stop_count"], "real_calibrated_coverage": row["calibrated_real_pg136_holdout"]["coverage"], "authorization_accuracy": row["unseen_authorization_family_holdout"]["accuracy"], "surface_ppl": row["surface_lm"]["perplexity"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
