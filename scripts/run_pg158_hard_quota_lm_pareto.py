"""PG-158: hard-negative quota x LM-anchor weight Pareto grid.

All runs start from the same PG-154 LM-anchor checkpoint.  Hard negatives are
mined from the training action split only, with equal per-source budgets.  The
``authorization`` real family is present only in the real holdout and is
reported as an explicit unseen-family check; it never enters training or
calibration.
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
ARTIFACT_DIR = ROOT / "artifacts" / "pg158-hard-quota-lm-pareto-v1"
REPORT = RESEARCH / "pg158_hard_quota_lm_pareto_report_v1.json"
DATASET = RESEARCH / "pg158_hard_quota_lm_pareto_dataset_v1.json"
PROTOCOL = RESEARCH / "pg158_hard_quota_lm_pareto_protocol_v1.json"
SEED = 15801
EPOCHS = 1
SURFACE_ANCHOR_COUNT = 300
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mine_quota(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, per_source: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_source <= 0:
        return [], {"selection_used_dev_labels": False, "selection_used_holdout_labels": False, "source_counts": {}, "family_counts": {}, "score": "disabled"}
    logits, labels, row_list = pg156._collect_logits(model, rows, vocabulary, device)
    probabilities = torch.softmax(logits, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    groups: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(row_list):
        source = str(row.get("source", "unknown"))
        family = str(row.get("surface_kind", "unknown"))
        priority = 1.0 - float(confidence[index].item())
        if int(predictions[index].item()) != int(labels[index].item()):
            priority += 0.75
        if int(predictions[index].item()) == STOP_INDEX and int(labels[index].item()) != STOP_INDEX:
            priority += 0.75
        groups[source][family].append((priority, index))
    selected: list[tuple[float, int]] = []
    source_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for source in sorted(groups):
        families = groups[source]
        per_family = max(1, per_source // max(len(families), 1))
        source_rows: list[tuple[float, int]] = []
        for family in sorted(families):
            source_rows.extend(sorted(families[family], key=lambda item: (-item[0], item[1]))[:per_family])
        if len(source_rows) < per_source:
            all_source = sorted((item for values in families.values() for item in values), key=lambda item: (-item[0], item[1]))
            seen = {index for _, index in source_rows}
            source_rows.extend(item for item in all_source if item[1] not in seen)
        source_rows = sorted(source_rows, key=lambda item: (-item[0], item[1]))[:per_source]
        selected.extend(source_rows)
        for _, index in source_rows:
            source_counts[source] += 1
            family_counts[str(row_list[index].get("surface_kind", "unknown"))] += 1
    result = []
    for rank, (_, index) in enumerate(selected):
        row = copy.deepcopy(row_list[index])
        row["row_id"] = f"pg158-hard-{per_source:03d}-{rank:05d}-{row.get('row_id', 'unknown')}"
        row["hard_negative"] = True
        row["replay_selection"] = "train_confidence_disagreement_source_family_balanced"
        result.append(row)
    return result, {"selection_used_dev_labels": False, "selection_used_holdout_labels": False, "score": "1-max_probability + 0.75*label_disagreement + 0.75*false_stop", "source_counts": dict(source_counts), "family_counts": dict(family_counts), "selected_count": len(result)}


def _evaluate(name: str, mode: str, model: nn.Module, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, selection: dict[str, Any], history: list[dict[str, Any]], started: float, train_count: int, lm_weight: float) -> dict[str, Any]:
    dev_logits, dev_labels, dev_rows = pg156._collect_logits(model, data["action_dev"], vocabulary, device)
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
        "lm_weight": lm_weight,
        "train_count": train_count,
        "hard_negative_count": len(data["hard_negatives"]),
        "surface_anchor_train_count": len(data["surface_anchor"]),
        "selection": selection,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temperature": temperature,
        "abstain_threshold": threshold,
        "raw_synthetic_holdout": pg154._action_metrics(model, data["synthetic_holdout"], vocabulary, device),
        "raw_real_pg136_holdout": pg154._action_metrics(model, data["real_holdout"], vocabulary, device),
        "unseen_authorization_family_holdout": pg154._action_metrics(model, data["unseen_authorization"], vocabulary, device),
        "raw_surface_unknown": pg154._action_metrics(model, data["surface_unknown"], vocabulary, device),
        "calibrated_synthetic_holdout": pg156._calibrated_metrics(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_real_pg136_holdout": pg156._calibrated_metrics(model, data["real_holdout"], vocabulary, device, temperature, threshold),
        "calibrated_unseen_authorization_family_holdout": pg156._calibrated_metrics(model, data["unseen_authorization"], vocabulary, device, temperature, threshold),
        "calibrated_surface_unknown": pg156._calibrated_metrics(model, data["surface_unknown"], vocabulary, device, temperature, threshold),
        "surface_lm": pg156._lm_metrics(model.base, data["surface_lm"], vocabulary, device),
        "language_canary": canary,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg158-hard-quota-lm-pareto-v1", "variant": name, "mode": mode, "lm_weight": lm_weight, "temperature": temperature, "abstain_threshold": threshold, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, mode: str, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, selection: dict[str, Any], lm_weight: float) -> dict[str, Any]:
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
                loss = action_loss + lm_weight * lm_loss + 0.01 * auxiliary + 0.08 * false_stop_loss + 0.20 * unknown_loss + (0.50 * lm_weight) * surface_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["action"] += float(action_loss.item())
            sums["lm"] += float(lm_loss.item())
            sums["surface"] += float(surface_loss.item())
            sums["false_stop"] += float(false_stop_loss.item())
            sums["unknown"] += float(unknown_loss.item())
        dev = pg154._action_metrics(model, data["action_dev"], vocabulary, device)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "lm_loss": round(sums["lm"] / max(len(loader), 1), 8), "surface_lm_loss": round(sums["surface"] / max(len(loader), 1), 8), "false_stop_loss": round(sums["false_stop"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev})
        print(json.dumps({"variant": name, "epoch": epoch, "dev_accuracy": dev["accuracy"], "dev_false_stop": dev["false_stop_count"]}, ensure_ascii=False), flush=True)
    result = _evaluate(name, mode, model, data, vocabulary, device, selection, history, started, len(train_rows), lm_weight)
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
    data["surface_anchor"] = data["surface_anchor"][:SURFACE_ANCHOR_COUNT]
    data["unseen_authorization"] = [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"]
    anchor = pg156._load_anchor(vocabulary, device)
    all_hard: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for per_source in (200, 400, 600):
        all_hard[per_source] = _mine_quota(anchor, data["action_train"], vocabulary, device, per_source)
    del anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    baseline_data = {**data, "hard_negatives": []}
    baseline = pg156._load_anchor(vocabulary, device)
    baseline_result = _evaluate("lm_anchor_baseline", "baseline", baseline, baseline_data, vocabulary, device, {"selection_used_holdout_labels": False}, [], time.perf_counter(), 0, 0.5)
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = [baseline_result]
    variants = [("q400_lm025", 200, 0.25), ("q800_lm050", 400, 0.50), ("q1200_lm100", 600, 1.00)]
    selection_meta = {}
    for name, per_source, lm_weight in variants:
        rows, selection = all_hard[per_source]
        variant_data = {**data, "hard_negatives": rows}
        selection_meta[name] = selection
        print(json.dumps({"status": "starting_variant", "variant": name, "device": str(device), "hard_count": len(rows), "lm_weight": lm_weight}, ensure_ascii=False), flush=True)
        results.append(_train(name, name, variant_data, vocabulary, device, selection, lm_weight))
    source = {**source, "surface_anchor_count": len(data["surface_anchor"]), "unseen_authorization_family_count": len(data["unseen_authorization"]), "hard_quota_variants": {str(per_source): {"count": len(rows), "source_counts": selection["source_counts"], "family_counts": selection["family_counts"]} for per_source, (rows, selection) in all_hard.items()}}
    report = {"protocol_id": "pg-pk-158-hard-quota-lm-pareto-v1", "schema_version": "pg158-hard-quota-lm-pareto-report-v1", "status": "completed_pg158_hard_quota_lm_pareto", "device": str(device), "seed": SEED, "source": source, "variants": results, "objective": "hard_negative_quota_x_lm_anchor_weight_pareto_with_unseen_authorization_family", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "unseen_authorization_family_training_used": False, "selection_used_dev_labels": False, "selection_used_holdout_labels": False, "calibration_fit_on_holdout": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg158-hard-quota-lm-pareto-dataset-v1", "source": source, "variants": {"lm_anchor_baseline": {"hard_per_source": 0, "lm_weight": 0.5}, "q400_lm025": {"hard_per_source": 200, "lm_weight": 0.25}, "q800_lm050": {"hard_per_source": 400, "lm_weight": 0.5}, "q1200_lm100": {"hard_per_source": 600, "lm_weight": 1.0}}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "unseen_authorization_family": len(data["unseen_authorization"]), "surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-158-hard-quota-lm-pareto-v1", "schema_version": "pg158-hard-quota-lm-pareto-protocol-v1", "objective": report["objective"], "variants": ["lm_anchor_baseline", "q400_lm025", "q800_lm050", "q1200_lm100"], "mining_pool": "PG-154 action_train only; dev/holdout labels excluded", "hard_quota_grid": {"per_source": [0, 200, 400, 600], "total": [0, 400, 800, 1200]}, "lm_weight_grid": [0.25, 0.5, 1.0], "unseen_family": "authorization appears only in PG-136 real holdout and is never trained or used for calibration", "threshold_fit": "highest dev coverage with zero dev false_stop", "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "raw_real_false_stop": row["raw_real_pg136_holdout"]["false_stop_count"], "calibrated_real_false_stop": row["calibrated_real_pg136_holdout"]["false_stop_count"], "unseen_authorization_accuracy": row["unseen_authorization_family_holdout"]["accuracy"], "unseen_authorization_calibrated_false_stop": row["calibrated_unseen_authorization_family_holdout"]["false_stop_count"], "synthetic_calibrated_false_stop": row["calibrated_synthetic_holdout"]["false_stop_count"], "surface_ppl": row["surface_lm"]["perplexity"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
