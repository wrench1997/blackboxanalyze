"""PG-159: family-invariant adapter with leave-one-family-out validation.

The MoE-Large-4E body is retained as the large language backbone.  A small
adapter is trained on the pooled Rule-IR representation while a gradient-
reversal family discriminator discourages the adapter from encoding the
training surface family.  Family labels are supervision for the adversary
only; they are never tokens or model inputs.  Each leave-one-family-out run
removes that real family from both action training and calibration, then tests
it as a held-out structural family.  PG-146 remains evaluation-only.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
ARTIFACT_DIR = ROOT / "artifacts" / "pg159-family-invariant-adapter-v1"
REPORT = RESEARCH / "pg159_family_invariant_adapter_report_v1.json"
DATASET = RESEARCH / "pg159_family_invariant_adapter_dataset_v1.json"
PROTOCOL = RESEARCH / "pg159_family_invariant_adapter_protocol_v1.json"
SEED = 15901
EPOCHS = 1
MAX_LEN = 128
ADVERSARIAL_LAMBDA = 0.20
ADVERSARIAL_WEIGHT = 0.20
LM_WEIGHT = 0.50
SURFACE_ANCHOR_WEIGHT = 0.50
UNKNOWN_WEIGHT = 0.20
FALSE_STOP_WEIGHT = 0.08
STOP_INDEX = pg154.STOP_INDEX
UNKNOWN_INDEX = pg154.UNKNOWN_INDEX


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def _gradient_reverse(value: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(value, scale)


class _InvariantDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, family_to_index: dict[str, int]) -> None:
        self.rows = rows
        self.vocabulary = vocabulary
        self.family_to_index = family_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        family = str(row.get("surface_kind", ""))
        family_index = self.family_to_index.get(family, -1) if bool(row.get("action_valid", True)) else -1
        return {
            "ids": self.vocabulary.encode(list(row.get("tokens", []))[:MAX_LEN]),
            "label": int(row.get("label_index", UNKNOWN_INDEX)),
            "action_valid": bool(row.get("action_valid", True)),
            "unknown_hint": bool(row.get("unknown_hint", False)),
            "family_index": family_index,
            "row": row,
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {
        "ids": ids,
        "mask": ids.ne(0),
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "action_valid": torch.tensor([item["action_valid"] for item in batch], dtype=torch.bool),
        "unknown_hint": torch.tensor([item["unknown_hint"] for item in batch], dtype=torch.bool),
        "family_indices": torch.tensor([item["family_index"] for item in batch], dtype=torch.long),
        "rows": [item["row"] for item in batch],
    }


class _FamilyInvariantAdapter(nn.Module):
    def __init__(self, body: nn.Module, bottleneck: int, family_count: int) -> None:
        super().__init__()
        self.base = body
        self.bottleneck = int(bottleneck)
        self.adapter = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, self.bottleneck), nn.GELU(), nn.Linear(self.bottleneck, 512))
        self.action_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, len(pg154.ACTION_NAMES)))
        self.family_head = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, max(int(family_count), 1)))

    def _pooled(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden, _, _ = self.base.encode(ids, mask)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return last + 0.50 * self.adapter(last)

    def action_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.action_head(self._pooled(ids, mask))

    def family_logits(self, pooled: torch.Tensor, *, adversarial: bool) -> torch.Tensor:
        value = _gradient_reverse(pooled, ADVERSARIAL_LAMBDA) if adversarial else pooled
        return self.family_head(value)


def _load_body(vocabulary: pg148._Vocabulary, device: torch.device) -> nn.Module:
    return pg154._load_body(vocabulary, device)


def _load_model(vocabulary: pg148._Vocabulary, device: torch.device, bottleneck: int, family_count: int) -> _FamilyInvariantAdapter:
    return _FamilyInvariantAdapter(_load_body(vocabulary, device), bottleneck, family_count).to(device)


def _filter_family(rows: list[dict[str, Any]], held_out: str | None) -> list[dict[str, Any]]:
    if not held_out:
        return list(rows)
    result = []
    for row in rows:
        is_real_action = bool(row.get("action_valid", True)) and str(row.get("source", "")) == "pg136_real"
        if is_real_action and str(row.get("surface_kind", "")) == held_out:
            continue
        result.append(copy.deepcopy(row))
    return result


def _build_data(vocabulary: pg148._Vocabulary, held_out: str | None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, Any]]:
    base, stats = pg156._build_data(vocabulary)
    data = {key: _filter_family(rows, held_out) for key, rows in base.items() if key not in {"surface_unknown", "surface_lm", "language_canary"}}
    data["surface_unknown"] = list(base["surface_unknown"])
    data["surface_lm"] = list(base["surface_lm"])
    data["language_canary"] = list(base["language_canary"])
    action_families = sorted({str(row.get("surface_kind", "unknown")) for row in data["action_train"] if bool(row.get("action_valid", True))})
    family_to_index = {family: index for index, family in enumerate(action_families)}
    stats = {**stats, "held_out_family": held_out, "family_names_in_adversary": action_families, "family_count": len(action_families), "unseen_authorization_count": sum(str(row.get("surface_kind")) == "authorization" for row in base["real_holdout"])}
    return data, family_to_index, stats


def _collect_action_logits(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    loader = DataLoader(_InvariantDataset(rows, vocabulary, {}), batch_size=128, shuffle=False, collate_fn=_collate)
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    row_list: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits.append(model.action_logits(batch["ids"].to(device), batch["mask"].to(device)).float().cpu())
            labels.append(batch["labels"].cpu())
            row_list.extend(batch["rows"])
    return torch.cat(logits), torch.cat(labels), row_list


def _action_metrics(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "accuracy": 0.0, "false_stop_count": 0, "unknown_count": 0, "unknown_abstain_rate": 1.0}
    logits, labels, row_list = _collect_action_logits(model, rows, vocabulary, device)
    predictions = logits.argmax(dim=-1)
    valid = torch.tensor([bool(row.get("action_valid", True)) for row in row_list], dtype=torch.bool)
    unknown = torch.tensor([bool(row.get("unknown_hint", False) or not row.get("typed_available", True) or row.get("label") == "unknown_oracle") for row in row_list], dtype=torch.bool)
    return {"count": int(valid.sum().item()), "accuracy": round(float(((predictions == labels) & valid).sum().item()) / max(int(valid.sum().item()), 1), 8), "false_stop_count": int(((predictions == STOP_INDEX) & labels.ne(STOP_INDEX) & valid).sum().item()), "unknown_count": int(unknown.sum().item()), "unknown_abstain_rate": round(float((predictions[unknown] == UNKNOWN_INDEX).sum().item()) / max(int(unknown.sum().item()), 1), 8) if bool(unknown.any()) else 1.0}


def _family_adversary_metrics(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, family_to_index: dict[str, int]) -> dict[str, Any]:
    valid_rows = [row for row in rows if bool(row.get("action_valid", True)) and str(row.get("surface_kind", "")) in family_to_index]
    if not valid_rows:
        return {"count": 0, "accuracy": 0.0, "family_count": len(family_to_index)}
    loader = DataLoader(_InvariantDataset(valid_rows, vocabulary, family_to_index), batch_size=128, shuffle=False, collate_fn=_collate)
    correct = 0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pooled = model._pooled(batch["ids"].to(device), batch["mask"].to(device))
            logits = model.family_logits(pooled, adversarial=False)
            labels = batch["family_indices"].to(device)
            valid = labels.ge(0)
            correct += int((logits.argmax(dim=-1)[valid] == labels[valid]).sum().item())
            count += int(valid.sum().item())
    return {"count": count, "accuracy": round(correct / max(count, 1), 8), "family_count": len(family_to_index)}


def _selective(logits: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], temperature: float, threshold: float) -> dict[str, Any]:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    accepted = confidence.ge(threshold)
    valid = torch.tensor([bool(row.get("action_valid", True)) for row in rows], dtype=torch.bool)
    unknown = torch.tensor([bool(row.get("unknown_hint", False) or not row.get("typed_available", True) or row.get("label") == "unknown_oracle") for row in rows], dtype=torch.bool)
    accepted_valid = accepted & valid
    count = int(valid.sum().item())
    accepted_count = int(accepted_valid.sum().item())
    return {"count": count, "accepted_count": accepted_count, "coverage": round(accepted_count / max(count, 1), 8), "selective_accuracy": round(float(((predictions == labels) & accepted_valid).sum().item()) / max(accepted_count, 1), 8), "false_stop_count": int(((predictions == STOP_INDEX) & labels.ne(STOP_INDEX) & accepted_valid).sum().item()), "abstain_count": count - accepted_count, "unknown_count": int(unknown.sum().item()), "unknown_abstain_rate": round(float(((predictions == UNKNOWN_INDEX) | (~accepted))[unknown].sum().item()) / max(int(unknown.sum().item()), 1), 8) if bool(unknown.any()) else 1.0}


def _fit_calibration(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[float, float]:
    logits, labels, row_list = _collect_action_logits(model, rows, vocabulary, device)
    temperatures = [0.5 + index * 0.05 for index in range(51)]
    temperature = min(temperatures, key=lambda value: float(nn.functional.cross_entropy(logits / value, labels).item()))
    thresholds = []
    for index in range(100):
        threshold = round(index / 100, 2)
        metrics = _selective(logits, labels, row_list, temperature, threshold)
        if metrics["false_stop_count"] == 0:
            thresholds.append((metrics["coverage"], -threshold, threshold))
    return round(float(temperature), 4), float(max(thresholds)[2]) if thresholds else 0.99


def _calibrated(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, temperature: float, threshold: float) -> dict[str, Any]:
    logits, labels, row_list = _collect_action_logits(model, rows, vocabulary, device)
    return _selective(logits, labels, row_list, temperature, threshold)


def _lm_metrics(model: _FamilyInvariantAdapter, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    return pg154.pg152._lm_metrics(model.base, rows, vocabulary, device)


def _evaluate(name: str, model: _FamilyInvariantAdapter, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, family_to_index: dict[str, int], history: list[dict[str, Any]], started: float, train_count: int, held_out: str | None, bottleneck: int) -> dict[str, Any]:
    temperature, threshold = _fit_calibration(model, data["action_dev"], vocabulary, device)
    family_metrics = {family: _action_metrics(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == family], vocabulary, device) for family in sorted({str(row.get("surface_kind")) for row in data["real_holdout"]})}
    family_calibrated = {family: _calibrated(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == family], vocabulary, device, temperature, threshold) for family in family_metrics}
    before = pg156._load_anchor(vocabulary, device)
    canary = compare_causal_lm_canary(before.base, model.base, data["language_canary"], vocabulary, device=device)
    del before
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {"variant": name, "held_out_family": held_out, "bottleneck": bottleneck, "train_count": train_count, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "known_family_count": len(family_to_index), "temperature": temperature, "abstain_threshold": threshold, "synthetic_holdout": _action_metrics(model, data["synthetic_holdout"], vocabulary, device), "real_pg136_holdout": _action_metrics(model, data["real_holdout"], vocabulary, device), "unseen_authorization_family_holdout": _action_metrics(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"], vocabulary, device), "calibrated_synthetic_holdout": _calibrated(model, data["synthetic_holdout"], vocabulary, device, temperature, threshold), "calibrated_real_pg136_holdout": _calibrated(model, data["real_holdout"], vocabulary, device, temperature, threshold), "calibrated_unseen_authorization_family_holdout": _calibrated(model, [row for row in data["real_holdout"] if str(row.get("surface_kind")) == "authorization"], vocabulary, device, temperature, threshold), "family_holdout_metrics": family_metrics, "family_holdout_calibrated": family_calibrated, "family_adversary_on_real_holdout": _family_adversary_metrics(model, data["real_holdout"], vocabulary, device, family_to_index), "surface_unknown": _action_metrics(model, data["surface_unknown"], vocabulary, device), "surface_lm": _lm_metrics(model, data["surface_lm"], vocabulary, device), "language_canary": canary, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg159-family-invariant-adapter-v1", "variant": name, "held_out_family": held_out, "bottleneck": bottleneck, "temperature": temperature, "abstain_threshold": threshold, "family_to_index": family_to_index, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train(name: str, held_out: str | None, bottleneck: int, all_data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    data, family_to_index, stats = _build_data(vocabulary, held_out)
    model = _load_model(vocabulary, device, bottleneck, len(family_to_index))
    train_rows = list(data["action_train"]) + list(data["lm_replay"]) + list(data["balanced_replay"]) + list(data["unknown_hints"]) + list(data["surface_anchor"])
    loader = DataLoader(_InvariantDataset(train_rows, vocabulary, family_to_index), batch_size=32, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.0e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
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
                    family_logits = model.family_logits(pooled[fam_valid], adversarial=True)
                    family_loss = nn.functional.cross_entropy(family_logits, family_indices[fam_valid])
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
                loss = action_loss + LM_WEIGHT * lm_loss + 0.01 * auxiliary + ADVERSARIAL_WEIGHT * family_loss + FALSE_STOP_WEIGHT * false_stop_loss + UNKNOWN_WEIGHT * unknown_loss + SURFACE_ANCHOR_WEIGHT * surface_loss
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
        dev = _action_metrics(model, data["action_dev"], vocabulary, device)
        fam_dev = _family_adversary_metrics(model, data["action_dev"], vocabulary, device, family_to_index)
        history.append({"epoch": epoch, "mean_loss": round(sums["loss"] / max(len(loader), 1), 8), "action_loss": round(sums["action"] / max(len(loader), 1), 8), "lm_loss": round(sums["lm"] / max(len(loader), 1), 8), "surface_lm_loss": round(sums["surface"] / max(len(loader), 1), 8), "family_adversarial_loss": round(sums["family"] / max(len(loader), 1), 8), "false_stop_loss": round(sums["false_stop"] / max(len(loader), 1), 8), "unknown_loss": round(sums["unknown"] / max(len(loader), 1), 8), "dev": dev, "family_adversary_dev": fam_dev})
        print(json.dumps({"variant": name, "epoch": epoch, "dev_accuracy": dev["accuracy"], "dev_false_stop": dev["false_stop_count"], "family_adversary_dev_accuracy": fam_dev["accuracy"]}, ensure_ascii=False), flush=True)
    result = _evaluate(name, model, data, vocabulary, device, family_to_index, history, started, len(train_rows), held_out, bottleneck)
    result["data_stats"] = stats
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
    base_data, _, base_stats = _build_data(vocabulary, None)
    # Shared model plus three independent leave-one-real-family-out runs.
    variants = [("shared_adapter_256", None, 256), ("loo_steady_adapter_256", "steady", 256), ("loo_blind_adapter_512", "blind", 512), ("loo_scope_adapter_512", "scope", 512)]
    results = []
    for name, held_out, bottleneck in variants:
        print(json.dumps({"status": "starting_variant", "variant": name, "held_out_family": held_out, "bottleneck": bottleneck, "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train(name, held_out, bottleneck, base_data, vocabulary, device))
    source = {**base_stats, "variant_count": len(variants), "variant_specs": [{"name": name, "held_out_family": held_out, "bottleneck": bottleneck} for name, held_out, bottleneck in variants], "family_labels_as_inputs": False, "family_labels_for_adversary_only": True, "pg146_training_labels_used": False}
    report = {"protocol_id": "pg-pk-159-family-invariant-adapter-v1", "schema_version": "pg159-family-invariant-adapter-report-v1", "status": "completed_pg159_family_invariant_adapter", "device": str(device), "seed": SEED, "source": source, "variants": results, "objective": "family_invariant_adapter_gradient_reversal_leave_one_family_out", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg145_training_eligible": False, "pg146_training_labels_used": False, "family_labels_as_model_inputs": False, "family_labels_for_adversary_only": True, "unseen_authorization_family_training_used": False, "calibration_fit_on_holdout": False, "holdout_labels_in_training": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg159-family-invariant-adapter-dataset-v1", "source": source, "variants": {name: {"held_out_family": held_out, "bottleneck": bottleneck} for name, held_out, bottleneck in variants}, "holdouts": {"synthetic": len(base_data["synthetic_holdout"]), "real_pg136": len(base_data["real_holdout"]), "authorization_family": sum(str(row.get("surface_kind")) == "authorization" for row in base_data["real_holdout"]), "surface_unknown": len(base_data["surface_unknown"]), "language_canary": len(base_data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-159-family-invariant-adapter-v1", "schema_version": "pg159-family-invariant-adapter-protocol-v1", "objective": report["objective"], "variants": [{"name": name, "held_out_family": held_out, "bottleneck": bottleneck} for name, held_out, bottleneck in variants], "family_adversary": {"gradient_reversal_lambda": ADVERSARIAL_LAMBDA, "loss_weight": ADVERSARIAL_WEIGHT, "family_labels_are_not_model_inputs": True}, "training": {"lm_weight": LM_WEIGHT, "surface_anchor_weight": SURFACE_ANCHOR_WEIGHT, "unknown_weight": UNKNOWN_WEIGHT, "false_stop_weight": FALSE_STOP_WEIGHT}, "leave_one_family_out": "held-out real family removed from action training and calibration; authorization remains a separate unseen holdout", "threshold_fit": "highest training-available dev coverage with zero dev false_stop", "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "held_out_family": row["held_out_family"], "bottleneck": row["bottleneck"], "synthetic_accuracy": row["synthetic_holdout"]["accuracy"], "real_accuracy": row["real_pg136_holdout"]["accuracy"], "authorization_accuracy": row["unseen_authorization_family_holdout"]["accuracy"], "authorization_calibrated_coverage": row["calibrated_unseen_authorization_family_holdout"]["coverage"], "surface_ppl": row["surface_lm"]["perplexity"], "family_adversary_real_accuracy": row["family_adversary_on_real_holdout"]["accuracy"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
