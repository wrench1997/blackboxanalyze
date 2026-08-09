"""PG-157: source/family-balanced hard-negative mining.

Hard examples are selected from the *training* action split using the frozen
PG-154 LM-anchor checkpoint.  Selection uses confidence and disagreement with
the training label, but never touches dev/holdout labels.  Two losses are
compared: focal action loss and a pairwise contrastive margin loss.  A small
label-free PG-147 surface proxy remains in both runs as an LM anchor; PG-146
real surface rows stay evaluation-only.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
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
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg157-hard-negative-mining-v1"
REPORT = RESEARCH / "pg157_hard_negative_mining_report_v1.json"
DATASET = RESEARCH / "pg157_hard_negative_mining_dataset_v1.json"
PROTOCOL = RESEARCH / "pg157_hard_negative_mining_protocol_v1.json"
SEED = 15701
EPOCHS = 1
HARD_PER_SOURCE = 400
SURFACE_ANCHOR_COUNT = 300
FOCAL_GAMMA = 2.0
CONTRASTIVE_MARGIN = 0.25
ACTION_NAMES = pg154.ACTION_NAMES
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _select_hard_negatives(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logits, labels, row_list = pg156._collect_logits(model, rows, vocabulary, device)
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    source_groups: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    scored_rows: list[tuple[float, int]] = []
    for index, row in enumerate(row_list):
        source = str(row.get("source", "unknown"))
        family = str(row.get("surface_kind", "unknown"))
        priority = float(1.0 - confidence[index].item())
        if int(predictions[index].item()) != int(labels[index].item()):
            priority += 0.75
        if int(predictions[index].item()) == STOP_INDEX and int(labels[index].item()) != STOP_INDEX:
            priority += 0.75
        source_groups[source][family].append((priority, index))
        scored_rows.append((priority, index))
    selected_indices: list[int] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    # Allocate each source budget across its observed surface families first,
    # then fill any shortfall by global priority from that same source.
    for source in sorted(source_groups):
        groups = source_groups[source]
        per_family = max(1, HARD_PER_SOURCE // max(len(groups), 1))
        source_selected: list[tuple[float, int]] = []
        for family in sorted(groups):
            source_selected.extend(sorted(groups[family], key=lambda item: (-item[0], item[1]))[:per_family])
        if len(source_selected) < HARD_PER_SOURCE:
            all_source = sorted((item for values in groups.values() for item in values), key=lambda item: (-item[0], item[1]))
            seen = {index for _, index in source_selected}
            source_selected.extend(item for item in all_source if item[1] not in seen)
        source_selected = sorted(source_selected, key=lambda item: (-item[0], item[1]))[:HARD_PER_SOURCE]
        for priority, index in source_selected:
            selected_indices.append(index)
            row = row_list[index]
            source_counts[source] += 1
            family_counts[str(row.get("surface_kind", "unknown"))] += 1
    selected = []
    for rank, index in enumerate(selected_indices):
        row = copy.deepcopy(row_list[index])
        row["row_id"] = f"pg157-hard-{rank:05d}-{row.get('row_id', 'unknown')}"
        row["hard_negative"] = True
        row["replay_selection"] = "train_confidence_disagreement_source_family_balanced"
        selected.append(row)
    return selected, {"selection_used_holdout_labels": False, "selection_used_dev_labels": False, "score": "1-max_probability + 0.75*label_disagreement + 0.75*false_stop", "source_counts": dict(source_counts), "family_counts": dict(family_counts), "selected_count": len(selected)}


def _collect_logits(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    return pg156._collect_logits(model, rows, vocabulary, device)


def _evaluate(name: str, mode: str, model: nn.Module, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, selection: dict[str, Any], history: list[dict[str, Any]], started: float, train_count: int) -> dict[str, Any]:
    dev_logits, dev_labels, dev_rows = _collect_logits(model, data["action_dev"], vocabulary, device)
    temperature = pg156._fit_temperature(dev_logits, dev_labels)
    threshold = pg156._fit_threshold(dev_logits, dev_labels, dev_rows, temperature)
    before = pg156._load_anchor(vocabulary, device)
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        "variant": name,
        "mode": mode,
        "train_count": train_count,
        "hard_negative_count": len(data["hard_negatives"]),
        "surface_anchor_train_count": len(data["surface_anchor"]),
        "selection": selection,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "raw_synthetic_holdout": pg154._action_metrics(model, data["synthetic_holdout"], vocabulary, device),
        "raw_real_pg136_holdout": pg154._action_metrics(model, data["real_holdout"], vocabulary, device),
        "raw_surface_unknown": pg154._action_metrics(model, data["surface_unknown"], vocabulary, device),
        "calibrated_synthetic_holdout": pg156._calibrated_metrics(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_real_pg136_holdout": pg156._calibrated_metrics(model, data["real_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_surface_unknown": pg156._calibrated_metrics(model, data["surface_unknown"], vocabulary, device, temperature, threshold),
        "surface_lm": pg156._lm_metrics(model.base, data["surface_lm"], vocabulary, device),
        "language_canary": canary,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg157-hard-negative-mining-v1", "variant": name, "mode": mode, "temperature": temperature, "abstain_threshold": threshold, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, mode: str, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, selection: dict[str, Any]) -> dict[str, Any]:
    model = pg156._load_anchor(vocabulary, device)
    train_rows = list(data["action_train"]) + list(data["lm_replay"]) + list(data["balanced_replay"]) + list(data["unknown_hints"]) + list(data["hard_negatives"]) + list(data["surface_anchor"])
    loader = DataLoader(pg154._ActionDataset(train_rows, vocabulary), batch_size=32, shuffle=True, collate_fn=pg154._collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.5e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"loss": 0.0, "action": 0.0, "focal": 0.0, "contrastive": 0.0, "lm": 0.0, "surface": 0.0, "unknown": 0.0}
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            unknown_hint = batch["unknown_hint"].to(device)
            hard_mask = torch.tensor([bool(row.get("hard_negative", False)) for row in batch["rows"]], dtype=torch.bool, device=device)
            surface_mask = torch.tensor([bool(row.get("surface_anchor", False)) for row in batch["rows"]], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_logits = model.action_logits(ids, mask)
                per_example = nn.functional.cross_entropy(action_logits[action_valid], labels[action_valid], reduction="none") if bool(action_valid.any()) else action_logits.new_zeros((0,))
                if mode == "focal_hard_mining" and per_example.numel():
                    pt = torch.exp(-per_example.detach())
                    action_loss = (((1.0 - pt).pow(FOCAL_GAMMA)) * per_example).mean()
                    focal_loss = action_loss
                else:
                    action_loss = per_example.mean() if per_example.numel() else action_logits.new_zeros(())
                    focal_loss = action_logits.new_zeros(())
                contrastive_loss = action_logits.new_zeros(())
                if mode == "contrastive_hard_mining" and bool(action_valid.any()):
                    valid_logits = action_logits[action_valid]
                    valid_labels = labels[action_valid]
                    label_logit = valid_logits.gather(1, valid_labels[:, None]).squeeze(1)
                    alternatives = valid_logits.clone()
                    alternatives.scatter_(1, valid_labels[:, None], torch.finfo(valid_logits.dtype).min)
                    contrastive_loss = nn.functional.softplus(alternatives.max(dim=1).values - label_logit + CONTRASTIVE_MARGIN).mean()
                lm_logits, auxiliary = pg154.pg152._lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                surface_lm_loss = action_logits.new_zeros(())
                if bool(surface_mask.any()):
                    surface_lm_loss = nn.functional.cross_entropy(lm_logits[surface_mask].reshape(-1, lm_logits.shape[-1]), targets[surface_mask].reshape(-1), ignore_index=0)
                unknown_loss = action_logits.new_zeros(())
                if bool(unknown_hint.any()):
                    unknown_loss = nn.functional.cross_entropy(action_logits[unknown_hint], torch.full((int(unknown_hint.sum().item()),), UNKNOWN_INDEX, dtype=torch.long, device=device))
                loss = action_loss + 0.5 * lm_loss + 0.01 * auxiliary + 0.35 * contrastive_loss + 0.20 * unknown_loss + 0.50 * surface_lm_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["action"] += float(action_loss.item())
            sums["focal"] += float(focal_loss.item())
            sums["contrastive"] += float(contrastive_loss.item())
            sums["lm"] += float(lm_loss.item())
            sums["surface"] += float(surface_lm_loss.item())
            sums["unknown"] += float(unknown_loss.item())
        dev = pg154._action_metrics(model, data["action_dev"], vocabulary, device)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "focal_loss": round(sums["focal"] / max(len(loader), 1), 8), "contrastive_loss": round(sums["contrastive"] / max(len(loader), 1), 8), "lm_loss": round(sums["lm"] / max(len(loader), 1), 8), "surface_lm_loss": round(sums["surface"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev})
        print(json.dumps({"variant": name, "epoch": epoch, "dev_accuracy": dev["accuracy"], "dev_false_stop": dev["false_stop_count"]}, ensure_ascii=False), flush=True)
    result = _evaluate(name, mode, model, data, vocabulary, device, selection, history, started, len(train_rows))
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
    data, source = pg156._build_data(vocabulary)
    anchor = pg156._load_anchor(vocabulary, device)
    hard_negatives, selection = _select_hard_negatives(anchor, data["action_train"], vocabulary, device)
    del anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    data["hard_negatives"] = hard_negatives
    data["surface_anchor"] = data["surface_anchor"][:SURFACE_ANCHOR_COUNT]
    baseline = pg156._load_anchor(vocabulary, device)
    baseline_started = time.perf_counter()
    baseline_result = _evaluate("lm_anchor_baseline", "baseline", baseline, data, vocabulary, device, selection, [], baseline_started, 0)
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = [baseline_result]
    for name, mode in [("focal_hard_mining", "focal_hard_mining"), ("contrastive_hard_mining", "contrastive_hard_mining")]:
        print(json.dumps({"status": "starting_variant", "variant": name, "device": str(device), "hard_count": len(hard_negatives)}, ensure_ascii=False), flush=True)
        results.append(_train(name, mode, data, vocabulary, device, selection))
    source = {**source, "hard_negative_count": len(hard_negatives), "hard_negative_source_counts": selection["source_counts"], "hard_negative_family_counts": selection["family_counts"], "surface_anchor_count": len(data["surface_anchor"])}
    report = {"protocol_id": "pg-pk-157-hard-negative-mining-v1", "schema_version": "pg157-hard-negative-mining-report-v1", "status": "completed_pg157_hard_negative_mining", "device": str(device), "seed": SEED, "source": source, "variants": results, "objective": "source_family_balanced_hard_negative_mining_focal_vs_contrastive", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "surface_anchor_labels": False, "selection_used_dev_labels": False, "selection_used_holdout_labels": False, "calibration_fit_on_holdout": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg157-hard-negative-mining-dataset-v1", "source": source, "variants": {"lm_anchor_baseline": {"hard_negative_count": 0}, "focal_hard_mining": {"hard_negative_count": len(hard_negatives), "loss": "focal_gamma_2"}, "contrastive_hard_mining": {"hard_negative_count": len(hard_negatives), "loss": "pairwise_softplus_margin_0.25"}}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-157-hard-negative-mining-v1", "schema_version": "pg157-hard-negative-mining-protocol-v1", "objective": report["objective"], "variants": ["lm_anchor_baseline", "focal_hard_mining", "contrastive_hard_mining"], "mining_pool": "PG-154 action_train only; dev/holdout labels excluded", "source_family_balance": {"per_source_budget": HARD_PER_SOURCE, "score": selection["score"]}, "loss": {"focal_gamma": FOCAL_GAMMA, "contrastive_margin": CONTRASTIVE_MARGIN, "lm_anchor": 0.5, "surface_anchor_lm": 0.5, "unknown_abstention": 0.2}, "threshold_fit": "highest dev coverage with zero dev false_stop", "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "raw_real_false_stop": row["raw_real_pg136_holdout"]["false_stop_count"], "calibrated_real_false_stop": row["calibrated_real_pg136_holdout"]["false_stop_count"], "calibrated_real_coverage": row["calibrated_real_pg136_holdout"]["coverage"], "synthetic_calibrated_false_stop": row["calibrated_synthetic_holdout"]["false_stop_count"], "surface_ppl": row["surface_lm"]["perplexity"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
