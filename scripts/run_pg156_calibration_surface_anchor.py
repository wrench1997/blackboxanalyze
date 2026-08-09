"""PG-156: calibration-aware action loss with a surface-preservation anchor.

The experiment continues from the PG-154 LM-anchor checkpoint and keeps the
PG-146 real-surface projection evaluation-only.  A deterministic subset of
the training-eligible PG-147 abstract corpus is used as a *surface proxy*
language anchor; no PG-145/PG-146 rows or evaluator labels enter training.

Three variants are compared:

* ``lm_anchor_baseline``: the frozen PG-154 LM-anchor checkpoint;
* ``calibration_aware``: source-balanced replay plus a soft false-stop and
  unknown-abstention objective;
* ``surface_anchor_calibrated``: the same objective with an additional
  surface-proxy causal-LM term.

Temperature and the abstain threshold are fit on the development split only.
The holdouts are used once for reporting and never for optimization.
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
import run_pg155_replay_calibration as pg155  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg156-calibration-surface-anchor-v1"
REPORT = RESEARCH / "pg156_calibration_surface_anchor_report_v1.json"
DATASET = RESEARCH / "pg156_calibration_surface_anchor_dataset_v1.json"
PROTOCOL = RESEARCH / "pg156_calibration_surface_anchor_protocol_v1.json"
SOURCE_CHECKPOINT = ROOT / "artifacts" / "pg154-multisource-action-replay-v1" / "lm_anchor.pt"
SEED = 15601
MAX_LEN = 128
EPOCHS = 1
BALANCED_HALF = 600
SURFACE_ANCHOR_COUNT = 600
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


def _surface_proxy(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select a deterministic, label-free surface-shaped LM anchor.

    PG-147 is training-eligible abstract data.  We select rows with the
    HTML/JavaScript/transport/IR sections intact so the anchor protects the
    model's representation of surface transitions without importing any raw
    body, payload, target identity, or evaluator label.
    """
    candidates = []
    for row in rows:
        tokens = list(row.get("tokens", []))
        sections = sum(token in {"[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[IR]", "[OBS]"} for token in tokens)
        if sections >= 4:
            copied = copy.deepcopy(row)
            copied["source"] = "pg147_surface_proxy"
            copied["action_valid"] = False
            copied["unknown_hint"] = False
            copied["surface_anchor"] = True
            copied.pop("label_index", None)
            copied["row_id"] = f"pg156-surface-anchor-{len(candidates):05d}-{row.get('row_id', 'unknown')}"
            candidates.append(copied)
    # Rows are already deterministically ordered in the dataset; cycling keeps
    # the requested count stable even if the corpus is smaller in a fixture.
    if not candidates:
        return []
    return [copy.deepcopy(candidates[index % len(candidates)]) for index in range(count)]


def _build_data(vocabulary: pg148._Vocabulary) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    data, source_stats = pg154._build_data(vocabulary)
    pg147 = json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))
    synthetic_rows = [row for row in data["action_train"] if row.get("source") == "pg149_synthetic"]
    real_rows = [row for row in data["action_train"] if row.get("source") == "pg136_real"]
    balanced = _sample_cycle(synthetic_rows, BALANCED_HALF, "pg156-balanced-synthetic", SEED)
    balanced += _sample_cycle(real_rows, BALANCED_HALF, "pg156-balanced-real", SEED + 1)
    surface_anchor = _surface_proxy(pg147["splits"]["train"], SURFACE_ANCHOR_COUNT)
    data = {
        **data,
        "balanced_replay": balanced,
        "surface_anchor": surface_anchor,
    }
    source_stats = {
        **source_stats,
        "balanced_replay_count": len(balanced),
        "balanced_synthetic_count": BALANCED_HALF,
        "balanced_real_count": BALANCED_HALF,
        "surface_anchor_count": len(surface_anchor),
        "surface_anchor_source": "pg147_training_eligible_abstract_surface_proxy",
        "pg145_training_eligible": False,
        "pg146_training_labels_used": False,
    }
    return data, source_stats


def _lm_metrics(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    return pg154.pg152._lm_metrics(model, rows, vocabulary, device)


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
    return {
        "count": count,
        "accepted_count": accepted_count,
        "coverage": round(accepted_count / max(count, 1), 8),
        "selective_accuracy": round(correct / max(accepted_count, 1), 8),
        "false_stop_count": false_stop,
        "abstain_count": count - accepted_count,
        "unknown_count": unknown_count,
        "unknown_abstain_rate": round(unknown_abstain / max(unknown_count, 1), 8) if unknown_count else 1.0,
    }


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    candidates = [0.5 + index * 0.05 for index in range(51)]
    losses = [float(nn.functional.cross_entropy(logits / temperature, labels).item()) for temperature in candidates]
    return round(float(candidates[min(range(len(losses)), key=losses.__getitem__)]), 4)


def _fit_threshold(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], temperature: float) -> float:
    candidates = [round(index / 100, 2) for index in range(100)]
    scored = []
    for threshold in candidates:
        metrics = _selective_from_logits(logits, labels, rows, temperature, threshold)
        if metrics["false_stop_count"] == 0:
            scored.append((metrics["coverage"], -threshold, threshold))
    return float(max(scored)[2]) if scored else 0.99


def _calibrated_metrics(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, temperature: float, threshold: float) -> dict[str, Any]:
    logits, labels, row_list = _collect_logits(model, rows, vocabulary, device)
    return _selective_from_logits(logits, labels, row_list, temperature, threshold)


def _evaluate(name: str, mode: str, model: nn.Module, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, selection: dict[str, Any], history: list[dict[str, Any]], started: float, *, trained: bool) -> dict[str, Any]:
    dev_logits, dev_labels, dev_rows = _collect_logits(model, data["action_dev"], vocabulary, device)
    temperature = _fit_temperature(dev_logits, dev_labels)
    threshold = _fit_threshold(dev_logits, dev_labels, dev_rows, temperature)
    before = _load_anchor(vocabulary, device)
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        "variant": name,
        "mode": mode,
        "trained": trained,
        "train_count": sum(len(data[key]) for key in {"action_train", "lm_replay", "balanced_replay", "unknown_hints"}) + (len(data["surface_anchor"]) if mode == "surface_anchor_calibrated" else 0),
        "surface_anchor_train_count": len(data["surface_anchor"]) if mode == "surface_anchor_calibrated" else 0,
        "selection": selection,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "raw_synthetic_holdout": pg154._action_metrics(model, data["synthetic_holdout"], vocabulary, device),
        "raw_real_pg136_holdout": pg154._action_metrics(model, data["real_holdout"], vocabulary, device),
        "raw_surface_unknown": pg154._action_metrics(model, data["surface_unknown"], vocabulary, device),
        "calibrated_synthetic_holdout": _calibrated_metrics(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_real_pg136_holdout": _calibrated_metrics(model, data["real_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_surface_unknown": _calibrated_metrics(model, data["surface_unknown"], vocabulary, device, temperature, threshold),
        "surface_lm": _lm_metrics(model.base, data["surface_lm"], vocabulary, device),
        "language_canary": canary,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg156-calibration-surface-anchor-v1", "variant": name, "mode": mode, "temperature": temperature, "abstain_threshold": threshold, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, mode: str, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    model = _load_anchor(vocabulary, device)
    train_rows = list(data["action_train"]) + list(data["lm_replay"]) + list(data["balanced_replay"]) + list(data["unknown_hints"])
    if mode == "surface_anchor_calibrated":
        train_rows += list(data["surface_anchor"])
    loader = DataLoader(pg154._ActionDataset(train_rows, vocabulary), batch_size=32, shuffle=True, collate_fn=pg154._collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"loss": 0.0, "action": 0.0, "lm": 0.0, "surface": 0.0, "false_stop": 0.0, "unknown": 0.0}
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            unknown_hint = batch["unknown_hint"].to(device)
            surface_mask = torch.tensor([bool(row.get("surface_anchor", False)) for row in batch["rows"]], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_logits = model.action_logits(ids, mask)
                action_loss = nn.functional.cross_entropy(action_logits[action_valid], labels[action_valid]) if bool(action_valid.any()) else action_logits.new_zeros(())
                lm_logits, auxiliary = pg154.pg152._lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                surface_lm_loss = action_logits.new_zeros(())
                if bool(surface_mask.any()):
                    surface_lm_loss = nn.functional.cross_entropy(lm_logits[surface_mask].reshape(-1, lm_logits.shape[-1]), targets[surface_mask].reshape(-1), ignore_index=0)
                non_stop = action_valid & labels.ne(STOP_INDEX)
                false_stop_loss = action_logits.new_zeros(())
                if bool(non_stop.any()):
                    alternatives = action_logits.clone()
                    alternatives[:, STOP_INDEX] = torch.finfo(alternatives.dtype).min
                    false_stop_loss = nn.functional.softplus(action_logits[:, STOP_INDEX] - alternatives.max(dim=-1).values)[non_stop].mean()
                unknown_loss = action_logits.new_zeros(())
                if bool(unknown_hint.any()):
                    unknown_loss = nn.functional.cross_entropy(action_logits[unknown_hint], torch.full((int(unknown_hint.sum().item()),), UNKNOWN_INDEX, dtype=torch.long, device=device))
                loss = action_loss + 0.5 * lm_loss + 0.01 * auxiliary + 0.08 * false_stop_loss + 0.20 * unknown_loss
                if mode == "surface_anchor_calibrated":
                    loss = loss + 0.75 * surface_lm_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["action"] += float(action_loss.item())
            sums["lm"] += float(lm_loss.item())
            sums["surface"] += float(surface_lm_loss.item())
            sums["false_stop"] += float(false_stop_loss.item())
            sums["unknown"] += float(unknown_loss.item())
        dev = pg154._action_metrics(model, data["action_dev"], vocabulary, device)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "lm_loss": round(sums["lm"] / max(len(loader), 1), 8), "surface_lm_loss": round(sums["surface"] / max(len(loader), 1), 8), "false_stop_loss": round(sums["false_stop"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev})
        print(json.dumps({"variant": name, "epoch": epoch, "dev_accuracy": dev["accuracy"], "dev_false_stop": dev["false_stop_count"]}, ensure_ascii=False), flush=True)
    result = _evaluate(name, mode, model, data, vocabulary, device, {"selection_used_labels": False, "surface_anchor_labels": False}, history, started, trained=True)
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
    data, source = _build_data(vocabulary)
    results = []
    baseline = _load_anchor(vocabulary, device)
    results.append(_evaluate("lm_anchor_baseline", "baseline", baseline, data, vocabulary, device, {"selection_used_labels": False}, [], time.perf_counter(), trained=False))
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for name, mode in [("calibration_aware", "calibration_aware"), ("surface_anchor_calibrated", "surface_anchor_calibrated")]:
        print(json.dumps({"status": "starting_variant", "variant": name, "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train(name, mode, data, vocabulary, device))
    report = {
        "protocol_id": "pg-pk-156-calibration-surface-anchor-v1",
        "schema_version": "pg156-calibration-surface-anchor-report-v1",
        "status": "completed_pg156_calibration_surface_anchor",
        "device": str(device),
        "seed": SEED,
        "source": source,
        "variants": results,
        "objective": "calibration_aware_false_stop_control_plus_surface_preservation_anchor",
        "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "surface_anchor_labels": False, "calibration_fit_on_holdout": False, "selection_used_labels": False},
        "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False},
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg156-calibration-surface-anchor-dataset-v1", "source": source, "variants": {"lm_anchor_baseline": {"train_count": 0}, "calibration_aware": {"balanced_replay": len(data["balanced_replay"]), "surface_anchor": 0}, "surface_anchor_calibrated": {"balanced_replay": len(data["balanced_replay"]), "surface_anchor": len(data["surface_anchor"])}}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-156-calibration-surface-anchor-v1", "schema_version": "pg156-calibration-surface-anchor-protocol-v1", "objective": report["objective"], "variants": ["lm_anchor_baseline", "calibration_aware", "surface_anchor_calibrated"], "surface_proxy": "PG-147 train only; no PG-145/PG-146 rows", "loss": {"action": 1.0, "lm_anchor": 0.5, "auxiliary": 0.01, "false_stop_softplus": 0.08, "unknown_abstention": 0.20, "surface_anchor_lm": 0.75}, "threshold_fit": "highest dev coverage with zero dev false_stop", "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "raw_real_false_stop": row["raw_real_pg136_holdout"]["false_stop_count"], "calibrated_real_false_stop": row["calibrated_real_pg136_holdout"]["false_stop_count"], "calibrated_real_coverage": row["calibrated_real_pg136_holdout"]["coverage"], "surface_ppl": row["surface_lm"]["perplexity"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
