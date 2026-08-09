"""PG-161: multi-seed and source-held-out replay of the best adapter-only model."""

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
import run_pg159_family_invariant_adapter as pg159  # noqa: E402
import run_pg160_hierarchical_lr_early_stop as pg160  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg161-adapter-seed-source-v1"
REPORT = RESEARCH / "pg161_adapter_seed_source_report_v1.json"
DATASET = RESEARCH / "pg161_adapter_seed_source_dataset_v1.json"
PROTOCOL = RESEARCH / "pg161_adapter_seed_source_protocol_v1.json"
SEED = 16101
EPOCHS = 2
BATCH_SIZE = 64
HEAD_LR = 1.0e-4
ADVERSARIAL_WEIGHT = 0.05
FALSE_STOP_WEIGHT = 0.20
UNKNOWN_WEIGHT = 0.20
BOTTLENECK = 256
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_source(vocabulary: pg148._Vocabulary, device: torch.device, family_count: int) -> pg159._FamilyInvariantAdapter:
    model = pg160._load_source(vocabulary, device, family_count)
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    return model


def _is_source(row: dict[str, Any], held_out: str | None) -> bool:
    if held_out is None:
        return False
    source = str(row.get("source", ""))
    if held_out == "pg149_synthetic":
        return source in {"pg149_synthetic", "pg149_unknown_hint"} or "balanced-synthetic" in source
    if held_out == "pg136_real":
        return source == "pg136_real" or "balanced-real" in source
    return False


def _filter_data(base: dict[str, list[dict[str, Any]]], held_out: str | None) -> dict[str, list[dict[str, Any]]]:
    data = {key: list(rows) for key, rows in base.items()}
    for key in ("action_train", "action_dev", "balanced_replay", "unknown_hints"):
        data[key] = [copy.deepcopy(row) for row in data[key] if not _is_source(row, held_out)]
    return data


def _dev_selection(model: pg159._FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    return pg160._dev_selection(model, rows, vocabulary, device)


def _evaluate(name: str, model: pg159._FamilyInvariantAdapter, data: dict[str, list[dict[str, Any]]], family_to_index: dict[str, int], vocabulary: pg148._Vocabulary, device: torch.device, history: list[dict[str, Any]], started: float, train_count: int, held_out: str | None, seed: int, best_epoch: int, best_dev: dict[str, Any]) -> dict[str, Any]:
    temperature, threshold = pg159._fit_calibration(model, data["action_dev"], vocabulary, device)
    family_metrics = {family: pg159._action_metrics(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == family], vocabulary, device) for family in sorted({str(row.get("surface_kind")) for row in data["real_holdout"]})}
    before = pg160._load_source(vocabulary, device, len(family_to_index))
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    authorization = [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"]
    result = {"variant": name, "seed": seed, "held_out_source": held_out, "best_epoch": best_epoch, "train_count": train_count, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "temperature": temperature, "abstain_threshold": threshold, "best_dev_selection": best_dev, "synthetic_holdout": pg159._action_metrics(model, data["synthetic_holdout"], vocabulary, device), "real_pg136_holdout": pg159._action_metrics(model, data["real_holdout"], vocabulary, device), "unseen_authorization_family_holdout": pg159._action_metrics(model, authorization, vocabulary, device), "calibrated_synthetic_holdout": pg159._calibrated(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold), "calibrated_real_pg136_holdout": pg159._calibrated(model, data["real_holdout"], vocabulary, device, temperature, threshold), "calibrated_unseen_authorization_family_holdout": pg159._calibrated(model, authorization, vocabulary, device, temperature, threshold), "family_holdout_metrics": family_metrics, "family_adversary_on_real_holdout": pg159._family_adversary_metrics(model, data["real_holdout"], vocabulary, device, family_to_index), "surface_unknown": pg159._action_metrics(model, data["surface_unknown"], vocabulary, device), "surface_lm": pg159._lm_metrics(model, data["surface_lm"], vocabulary, device), "language_canary": canary, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg161-adapter-seed-source-v1", "variant": name, "seed": seed, "held_out_source": held_out, "best_epoch": best_epoch, "temperature": temperature, "abstain_threshold": threshold, "vocabulary": vocabulary.itos, "family_to_index": family_to_index, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, seed: int, held_out: str | None, base_data: dict[str, list[dict[str, Any]]], family_to_index: dict[str, int], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    data = _filter_data(base_data, held_out)
    model = _load_source(vocabulary, device, len(family_to_index))
    # With the body frozen, LM/surface rows have zero gradient.  Keeping the
    # action and unknown rows here makes the source-held-out intervention
    # explicit and avoids pretending replay is doing work it cannot do.
    train_rows = list(data["action_train"]) + list(data["balanced_replay"]) + list(data["unknown_hints"])
    loader = DataLoader(pg159._InvariantDataset(train_rows, vocabulary, family_to_index), batch_size=BATCH_SIZE, shuffle=True, collate_fn=pg159._collate)
    params = list(model.adapter.parameters()) + list(model.action_head.parameters()) + list(model.family_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=HEAD_LR, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_dev: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = {"loss": 0.0, "action": 0.0, "family": 0.0, "false_stop": 0.0, "unknown": 0.0}
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            family_indices = batch["family_indices"].to(device)
            unknown_hint = batch["unknown_hint"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                hidden, _, _ = model.base.encode(ids, mask)
                lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
                last = hidden[torch.arange(hidden.shape[0], device=device), lengths - 1]
                pooled = last + 0.50 * model.adapter(last)
                action_logits = model.action_head(pooled)
                action_loss = nn.functional.cross_entropy(action_logits[action_valid], labels[action_valid]) if bool(action_valid.any()) else action_logits.new_zeros(())
                fam_valid = action_valid & family_indices.ge(0)
                family_loss = action_logits.new_zeros(())
                if bool(fam_valid.any()):
                    family_loss = nn.functional.cross_entropy(model.family_logits(pooled[fam_valid], adversarial=True), family_indices[fam_valid])
                non_stop = action_valid & labels.ne(STOP_INDEX)
                false_stop_loss = action_logits.new_zeros(())
                if bool(non_stop.any()):
                    alternatives = action_logits.clone()
                    alternatives[:, STOP_INDEX] = torch.finfo(alternatives.dtype).min
                    false_stop_loss = nn.functional.softplus(action_logits[:, STOP_INDEX] - alternatives.max(dim=-1).values)[non_stop].mean()
                unknown_loss = action_logits.new_zeros(())
                if bool(unknown_hint.any()):
                    unknown_loss = nn.functional.cross_entropy(action_logits[unknown_hint], torch.full((int(unknown_hint.sum().item()),), UNKNOWN_INDEX, dtype=torch.long, device=device))
                loss = action_loss + ADVERSARIAL_WEIGHT * family_loss + FALSE_STOP_WEIGHT * false_stop_loss + UNKNOWN_WEIGHT * unknown_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += float(loss.item())
            sums["action"] += float(action_loss.item())
            sums["family"] += float(family_loss.item())
            sums["false_stop"] += float(false_stop_loss.item())
            sums["unknown"] += float(unknown_loss.item())
        dev = _dev_selection(model, data["action_dev"], vocabulary, device)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "family_loss": round(sums["family"] / max(len(loader), 1), 8), "false_stop_loss": round(sums["false_stop"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev})
        if best_key is None or tuple(dev["selection_key"]) > best_key:
            best_key = tuple(dev["selection_key"])
            best_dev = dev
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"variant": name, "seed": seed, "epoch": epoch, "dev_accuracy": dev["raw"]["accuracy"], "dev_false_stop": dev["raw"]["false_stop_count"], "dev_coverage": dev["calibrated"]["coverage"]}, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    result = _evaluate(name, model, data, family_to_index, vocabulary, device, history, started, len(train_rows), held_out, seed, best_epoch, best_dev or {})
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = pg148._Vocabulary(list(json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))["vocabulary"]))
    base_data, family_to_index, source = pg159._build_data(vocabulary, None)
    variants = [("seed_16101", 16101, None), ("seed_16102", 16102, None), ("seed_16103", 16103, None), ("source_heldout_pg149", 16104, "pg149_synthetic"), ("source_heldout_pg136", 16105, "pg136_real")]
    results = []
    for name, seed, held_out in variants:
        print(json.dumps({"status": "starting_variant", "variant": name, "seed": seed, "held_out_source": held_out, "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train(name, seed, held_out, base_data, family_to_index, vocabulary, device))
    source = {**source, "variant_count": len(variants), "source_heldout_variants": {name: held for name, _, held in variants if held}, "family_labels_as_model_inputs": False, "pg146_training_labels_used": False}
    report = {"protocol_id": "pg-pk-161-adapter-seed-source-v1", "schema_version": "pg161-adapter-seed-source-report-v1", "status": "completed_pg161_adapter_seed_source", "device": str(device), "seed": SEED, "source": source, "variants": results, "objective": "adapter_only_multiseed_source_heldout_replay", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "family_labels_as_model_inputs": False, "unseen_authorization_family_training_used": False, "source_heldout_labels_in_training": False, "calibration_fit_on_holdout": False, "holdout_labels_in_training": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg161-adapter-seed-source-dataset-v1", "source": source, "variants": {name: {"seed": seed, "held_out_source": held} for name, seed, held in variants}, "holdouts": {"synthetic": len(base_data["synthetic_holdout"]), "real_pg136": len(base_data["real_holdout"]), "authorization_family": sum(str(row.get("surface_kind")) == "authorization" for row in base_data["real_holdout"]), "surface_unknown": len(base_data["surface_unknown"]), "language_canary": len(base_data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-161-adapter-seed-source-v1", "schema_version": "pg161-adapter-seed-source-protocol-v1", "objective": report["objective"], "variants": [{"name": name, "seed": seed, "held_out_source": held} for name, seed, held in variants], "source_checkpoint": "PG-159 shared_adapter_256", "training": {"body_frozen": True, "head_lr": HEAD_LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE}, "source_heldout": "pg149_synthetic and pg136_real are removed from both action train and balanced replay; holdouts remain untouched", "threshold_fit": "training-available dev only, zero false-stop coverage", "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "seed": row["seed"], "held_out_source": row["held_out_source"], "synthetic_raw_false_stop": row["synthetic_holdout"]["false_stop_count"], "real_raw_false_stop": row["real_pg136_holdout"]["false_stop_count"], "authorization_accuracy": row["unseen_authorization_family_holdout"]["accuracy"], "authorization_coverage": row["calibrated_unseen_authorization_family_holdout"]["coverage"], "surface_ppl": row["surface_lm"]["perplexity"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
