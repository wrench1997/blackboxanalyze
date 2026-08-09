"""PG-155: source-balanced replay, selective replay, and action calibration.

All variants continue from the same PG-154 LM-anchor checkpoint.  Replay
selection is based on source or model uncertainty only; labels are reserved
for the supervised loss and holdout evaluation.  Temperature and abstain
thresholds are fitted on the development split and never on holdouts.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
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
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg155-replay-calibration-v1"
REPORT = RESEARCH / "pg155_replay_calibration_report_v1.json"
DATASET = RESEARCH / "pg155_replay_calibration_dataset_v1.json"
PROTOCOL = RESEARCH / "pg155_replay_calibration_protocol_v1.json"
SOURCE_CHECKPOINT = ROOT / "artifacts" / "pg154-multisource-action-replay-v1" / "lm_anchor.pt"
SEED = 15501
MAX_LEN = 128
EPOCHS = 1
REPLAY_COUNT = 1200
BALANCED_HALF = 600
ACTION_NAMES = pg154.ACTION_NAMES
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_anchor(vocabulary: pg148._Vocabulary, device: torch.device) -> nn.Module:
    model = pg154.pg152._MoEActionModel(pg154._load_body(vocabulary, device), 512).to(device)
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _collect_logits(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    loader = DataLoader(pg154._ActionDataset(rows, vocabulary), batch_size=128, shuffle=False, collate_fn=pg154._collate)
    logits_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    rows_out: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits_out.append(model.action_logits(batch["ids"].to(device), batch["mask"].to(device)).float().cpu())
            labels_out.append(batch["labels"].cpu())
            rows_out.extend(batch["rows"])
    return torch.cat(logits_out), torch.cat(labels_out), rows_out


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = [0.5 + index * 0.05 for index in range(51)]
    losses = [float(nn.functional.cross_entropy(logits / temperature, labels).item()) for temperature in candidates]
    return round(float(candidates[min(range(len(losses)), key=losses.__getitem__)]), 4)


def _selective_from_logits(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], temperature: float, threshold: float) -> dict[str, Any]:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    accepted = confidence.ge(threshold)
    action_valid = torch.tensor([bool(row.get("action_valid", True)) for row in rows], dtype=torch.bool)
    unknown_mask = torch.tensor([bool(row.get("unknown_hint", False) or not row.get("typed_available", True) or row.get("label") == "unknown_oracle") for row in rows], dtype=torch.bool)
    accepted_action = accepted & action_valid
    count = int(action_valid.sum().item())
    accepted_count = int(accepted_action.sum().item())
    correct = int(((predictions == labels) & accepted_action).sum().item())
    false_stop = int(((predictions == STOP_INDEX) & labels.ne(STOP_INDEX) & accepted_action).sum().item())
    unknown_count = int(unknown_mask.sum().item())
    unknown_abstain = int(((predictions == UNKNOWN_INDEX) | (~accepted))[unknown_mask].sum().item())
    return {"count": count, "accepted_count": accepted_count, "coverage": round(accepted_count / max(count, 1), 8), "selective_accuracy": round(correct / max(accepted_count, 1), 8), "false_stop_count": false_stop, "abstain_count": count - accepted_count, "unknown_count": unknown_count, "unknown_abstain_rate": round(unknown_abstain / max(unknown_count, 1), 8) if unknown_count else 1.0}


def _fit_threshold(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], temperature: float) -> float:
    candidates = [round(index / 100, 2) for index in range(0, 100)]
    scored = []
    for threshold in candidates:
        metrics = _selective_from_logits(logits, labels, rows, temperature, threshold)
        if metrics["false_stop_count"] == 0:
            scored.append((metrics["coverage"], -threshold, threshold))
    if scored:
        return float(max(scored)[2])
    return 0.99


def _calibrated_metrics(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, temperature: float, threshold: float) -> dict[str, Any]:
    logits, labels, row_list = _collect_logits(model, rows, vocabulary, device)
    return _selective_from_logits(logits, labels, row_list, temperature, threshold)


def _sample_cycle(rows: list[dict[str, Any]], count: int, prefix: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    result = []
    for index in range(count):
        row = copy.deepcopy(order[index % len(order)])
        row["row_id"] = f"{prefix}-{index:05d}-{row.get('row_id', 'unknown')}"
        row["replay_selection"] = prefix
        result.append(row)
    return result


def _select_uncertain(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, count: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logits, _, row_list = _collect_logits(model, rows, vocabulary, device)
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    priorities = (1.0 - confidence).tolist()
    for index, prediction in enumerate(predictions.tolist()):
        if prediction == STOP_INDEX:
            priorities[index] += 1.0
    indices = sorted(range(len(row_list)), key=lambda index: (-priorities[index], index))[:count]
    selected = []
    for rank, index in enumerate(indices):
        row = copy.deepcopy(row_list[index])
        row["row_id"] = f"pg155-selective-{rank:05d}-{row.get('row_id', 'unknown')}"
        row["replay_selection"] = "model_uncertainty_or_predicted_stop"
        selected.append(row)
    source_counts: dict[str, int] = {}
    for row in selected:
        source = str(row.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected, source_counts


def _lm_metrics(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    # Reuse the causal projection from PG-154 without exposing raw bodies.
    return pg154.pg152._lm_metrics(model, rows, vocabulary, device)


def _evaluate_model(name: str, mode: str, model: nn.Module, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, replay_rows: list[dict[str, Any]], selection_stats: dict[str, Any], started: float, *, trained: bool, history: list[dict[str, Any]]) -> dict[str, Any]:
    dev_logits, dev_labels, dev_rows = _collect_logits(model, data["action_dev"], vocabulary, device)
    temperature = _fit_temperature(dev_logits, dev_labels)
    threshold = _fit_threshold(dev_logits, dev_labels, dev_rows, temperature)
    raw_synthetic = pg154._action_metrics(model, data["synthetic_holdout"], vocabulary, device)
    raw_real = pg154._action_metrics(model, data["real_holdout"], vocabulary, device)
    raw_surface = pg154._action_metrics(model, data["surface_unknown"], vocabulary, device)
    calibrated_synthetic = _calibrated_metrics(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold)
    calibrated_real = _calibrated_metrics(model, data["real_holdout"], vocabulary, device, temperature, threshold)
    calibrated_surface = _calibrated_metrics(model, data["surface_unknown"], vocabulary, device, temperature, threshold)
    before = _load_anchor(vocabulary, device)
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {"variant": name, "mode": mode, "trained": trained, "train_count": len(data["action_train"]) + len(data["lm_replay"]) + len(replay_rows), "replay_count": len(replay_rows), "selection": selection_stats, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "temperature": temperature, "abstain_threshold": threshold, "raw_synthetic_holdout": raw_synthetic, "raw_real_pg136_holdout": raw_real, "raw_surface_unknown": raw_surface, "calibrated_synthetic_holdout": calibrated_synthetic, "calibrated_real_pg136_holdout": calibrated_real, "calibrated_surface_unknown": calibrated_surface, "surface_lm": _lm_metrics(model.base, data["surface_lm"], vocabulary, device), "language_canary": canary, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg155-replay-calibration-v1", "variant": name, "mode": mode, "temperature": temperature, "abstain_threshold": threshold, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train_variant(name: str, mode: str, replay_rows: list[dict[str, Any]], selection_stats: dict[str, Any], data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    model = _load_anchor(vocabulary, device)
    train_rows = list(data["action_train"]) + list(data["lm_replay"]) + list(replay_rows)
    loader = DataLoader(pg154._ActionDataset(train_rows, vocabulary), batch_size=32, shuffle=True, collate_fn=pg154._collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model.action_logits(ids, mask)
                action_loss = nn.functional.cross_entropy(logits[action_valid], labels[action_valid]) if bool(action_valid.any()) else logits.new_zeros(())
                lm_logits, auxiliary = pg154.pg152._lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                loss = action_loss + 0.5 * lm_loss + 0.01 * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "synthetic_dev": pg154._action_metrics(model, data["action_dev"], vocabulary, device), "real_dev": pg154._action_metrics(model, data["real_holdout"], vocabulary, device)})
        print(json.dumps({"variant": name, "epoch": epoch, "synthetic_dev_accuracy": history[-1]["synthetic_dev"]["accuracy"], "synthetic_dev_false_stop": history[-1]["synthetic_dev"]["false_stop_count"]}, ensure_ascii=False), flush=True)
    result = _evaluate_model(name, mode, model, data, vocabulary, device, replay_rows, selection_stats, started, trained=True, history=history)
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
    data, source_stats = pg154._build_data(vocabulary)
    anchor = _load_anchor(vocabulary, device)
    synthetic_rows = [row for row in data["action_train"] if row.get("source") == "pg149_synthetic"]
    real_rows = [row for row in data["action_train"] if row.get("source") == "pg136_real"]
    balanced_replay = _sample_cycle(synthetic_rows, BALANCED_HALF, "pg155-balanced-synthetic", SEED) + _sample_cycle(real_rows, BALANCED_HALF, "pg155-balanced-real", SEED + 1)
    selective_replay, selective_counts = _select_uncertain(anchor, data["action_train"], vocabulary, device, REPLAY_COUNT)
    del anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    selection_metadata = {"balanced_replay": {"source_counts": {"pg149_synthetic": BALANCED_HALF, "pg136_real": BALANCED_HALF}, "selection_used_labels": False}, "selective_replay": {"source_counts": selective_counts, "selection_used_labels": False, "score": "1-max_probability + predicted_stop_bonus"}}
    results = []
    for name, mode, replay_rows, selection_stats in [("balanced_replay", "source_balanced", balanced_replay, selection_metadata["balanced_replay"]), ("selective_replay", "selective_uncertainty", selective_replay, selection_metadata["selective_replay"])]:
        print(json.dumps({"status": "starting_variant", "variant": name, "replay_count": len(replay_rows), "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train_variant(name, mode, replay_rows, selection_stats, data, vocabulary, device))
    baseline = _load_anchor(vocabulary, device)
    baseline_result = _evaluate_model("lm_anchor_baseline", "baseline", baseline, data, vocabulary, device, [], {"selection_used_labels": False}, time.perf_counter(), trained=False, history=[])
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results.insert(0, baseline_result)
    stats = {**source_stats, "balanced_replay_count": len(balanced_replay), "selective_replay_count": len(selective_replay), "selective_source_counts": selective_counts, "selection_used_labels": False}
    report = {"protocol_id": "pg-pk-155-replay-calibration-v1", "schema_version": "pg155-replay-calibration-report-v1", "status": "completed_pg155_replay_calibration", "device": str(device), "seed": SEED, "source": stats, "variants": results, "objective": "source_balanced_selective_replay_temperature_calibration", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg146_training_labels_used": False, "calibration_fit_on_holdout": False, "selection_used_labels": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg155-replay-calibration-dataset-v1", "source": stats, "variants": {"lm_anchor_baseline": {"replay": 0}, "balanced_replay": {"synthetic": BALANCED_HALF, "real": BALANCED_HALF}, "selective_replay": {"count": len(selective_replay), "selection": "model_uncertainty_or_predicted_stop"}}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-155-replay-calibration-v1", "schema_version": "pg155-replay-calibration-protocol-v1", "objective": report["objective"], "variants": ["lm_anchor_baseline", "balanced_replay", "selective_replay"], "temperature_grid": [0.5, 0.55, "...", 3.0], "threshold_fit": "highest dev coverage with zero dev false_stop", "selection": {"source_balanced": "600 synthetic + 600 real", "selective": "low confidence plus predicted stop, no labels"}, "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "temperature": row["temperature"], "threshold": row["abstain_threshold"], "raw_real_false_stop": row["raw_real_pg136_holdout"]["false_stop_count"], "calibrated_real_false_stop": row["calibrated_real_pg136_holdout"]["false_stop_count"], "calibrated_real_coverage": row["calibrated_real_pg136_holdout"]["coverage"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
