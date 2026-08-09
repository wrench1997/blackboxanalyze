"""PG-154: multi-source action-head replay with false-stop suppression.

All variants start from the same PG-151 MoE-Large-4E body and receive the
same 25% real PG-136 action mix.  The ablation changes only the objective:
action-only, LM-anchored replay, or LM replay plus an explicit non-stop/unknown
margin.  PG-146 real-surface rows stay evaluation-only; they are never used as
training labels.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg148_large_model_posttraining as pg148  # noqa: E402
import run_pg152_real_mix_architecture as pg152  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402
from app.moe_trace_transformer import MoETraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg154-multisource-action-replay-v1"
REPORT = RESEARCH / "pg154_multisource_action_replay_report_v1.json"
DATASET = RESEARCH / "pg154_multisource_action_replay_dataset_v1.json"
PROTOCOL = RESEARCH / "pg154_multisource_action_replay_protocol_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg151-moe-capacity-v1" / "moe_large_4e.pt"
SEED = 15401
MAX_LEN = 128
EPOCHS = 2
REPLAY_COUNT = 1200
REAL_FRACTION = 0.25
ACTION_NAMES = pg148.ACTION_NAMES
STOP_INDEX = ACTION_NAMES.index("stop_confirmed_positive")
UNKNOWN_INDEX = ACTION_NAMES.index("abstain_unknown_oracle")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class _ActionDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary) -> None:
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "ids": self.vocabulary.encode(row["tokens"][:MAX_LEN]),
            "label": int(row.get("label_index", UNKNOWN_INDEX)),
            "action_valid": bool(row.get("action_valid", True)),
            "unknown_hint": bool(row.get("unknown_hint", False)),
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
        "rows": [item["row"] for item in batch],
    }


def _load_body(vocabulary: pg148._Vocabulary, device: torch.device) -> MoETraceTransformer:
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    model = MoETraceTransformer(len(vocabulary.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), n_experts=int(config["n_experts"]), expert_ff=int(config["expert_ff"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _with_source(row: dict[str, Any], source: str, *, action_valid: bool = True, unknown_hint: bool = False) -> dict[str, Any]:
    copied = dict(row)
    copied["source"] = source
    copied["action_valid"] = action_valid
    copied["unknown_hint"] = unknown_hint
    if action_valid and "label_index" not in copied:
        label = str(copied.get("action_label") or copied.get("label") or "abstain_unknown_oracle")
        copied["label"] = label if label in ACTION_NAMES else "abstain_unknown_oracle"
        copied["label_index"] = ACTION_NAMES.index(copied["label"])
    return copied


def _build_data(vocabulary: pg148._Vocabulary) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pg149 = _load_json(RESEARCH / "pg149_causal_action_alignment_dataset_v1.json")["splits"]
    synthetic_train = [_with_source(row, "pg149_synthetic") for row in pg149["train"]]
    synthetic_dev = [_with_source(row, "pg149_synthetic") for row in pg149["dev"]]
    synthetic_holdout = [_with_source(row, "pg149_synthetic") for row in pg149["holdout"]]
    real_train, real_dev, real_holdout, _, real_stats = pg148._prepare_rows()
    real_train = [_with_source(row, "pg136_real") for row in real_train]
    real_dev = [_with_source(row, "pg136_real") for row in real_dev]
    real_holdout = [_with_source(row, "pg136_real") for row in real_holdout]
    desired_real = int(len(synthetic_train) * REAL_FRACTION / (1.0 - REAL_FRACTION))
    repeated_real = []
    for index in range(desired_real):
        row = copy.deepcopy(real_train[index % len(real_train)])
        row["row_id"] = f"pg154-real-repeat-{index:05d}-{row['row_id']}"
        repeated_real.append(row)
    action_mix = synthetic_train + repeated_real
    random.Random(SEED).shuffle(action_mix)
    pg147 = _load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")["splits"]
    replay = []
    for index, row in enumerate(pg147["train"][:REPLAY_COUNT]):
        replay.append(_with_source({**row, "row_id": f"pg154-replay-{index:05d}-{row.get('row_id', 'unknown')}"}, "pg147_lm_replay", action_valid=False))
    # Unknown-hint rows are derived only from already labelled synthetic
    # trajectories.  The evaluation-only PG-146 rows never become training
    # labels; they are kept for the final abstention check.
    unknown_hints = [copy.deepcopy(row) for row in synthetic_train if str(row.get("label")) == "abstain_unknown_oracle"]
    unknown_hints = [dict(row, row_id=f"pg154-unknown-hint-{index:05d}-{row['row_id']}", source="pg149_unknown_hint", action_valid=False, unknown_hint=True) for index, row in enumerate((unknown_hints * 8)[:120])]
    pg146_rows = [_with_source(row, "pg146_evaluation_only", action_valid=False) for row in _load_json(RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json")["rows"]]
    data = {
        "action_train": action_mix,
        "action_dev": synthetic_dev + real_dev,
        "synthetic_holdout": synthetic_holdout,
        "real_holdout": real_holdout,
        "lm_replay": replay,
        "unknown_hints": unknown_hints,
        "surface_unknown": pg146_rows,
        "language_canary": [_with_source(row, "pg147_old_holdout", action_valid=False) for row in pg147["holdout"]],
        "surface_lm": pg146_rows,
    }
    stats = {**real_stats, "synthetic_train_count": len(synthetic_train), "synthetic_dev_count": len(synthetic_dev), "synthetic_holdout_count": len(synthetic_holdout), "real_train_count": len(real_train), "real_dev_count": len(real_dev), "real_holdout_count": len(real_holdout), "real_repeated_count": len(repeated_real), "action_mix_count": len(action_mix), "lm_replay_count": len(replay), "unknown_hint_count": len(unknown_hints), "surface_unknown_count": len(pg146_rows), "language_canary_count": len(data["language_canary"]), "real_fraction": REAL_FRACTION}
    return data, stats


def _lm_metrics(model: MoETraceTransformer, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"perplexity": 0.0, "next_token_accuracy": 0.0, "token_count": 0}
    loader = DataLoader(_ActionDataset(rows, vocabulary), batch_size=32, shuffle=False, collate_fn=_collate)
    model.eval()
    loss_sum = 0.0
    count = 0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            hidden, _, _ = model.encode(ids[:, :-1], mask[:, :-1])
            logits = model.lm_head(hidden)
            targets = ids[:, 1:]
            valid = targets.ne(0)
            loss_sum += float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            count += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean_loss = loss_sum / max(count, 1)
    return {"perplexity": round(math.exp(min(mean_loss, 20.0)), 8), "next_token_accuracy": round(correct / max(count, 1), 8), "token_count": count}


def _action_metrics(model: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "accuracy": 0.0, "false_stop_count": 0, "unknown_count": 0, "unknown_abstain_rate": 1.0}
    loader = DataLoader(_ActionDataset(rows, vocabulary), batch_size=128, shuffle=False, collate_fn=_collate)
    model.eval()
    correct = 0
    count = 0
    false_stop = 0
    unknown_count = 0
    unknown_abstain = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            logits = model.action_logits(ids, mask)
            predictions = logits.argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()
            valid = batch["action_valid"].tolist()
            for prediction, label, is_valid, row in zip(predictions, labels, valid, batch["rows"]):
                is_unknown = bool(row.get("unknown_hint", False) or not row.get("typed_available", True) or row.get("label") == "unknown_oracle")
                if is_unknown:
                    unknown_count += 1
                    unknown_abstain += int(prediction == UNKNOWN_INDEX)
                if is_valid:
                    count += 1
                    correct += int(prediction == label)
                    false_stop += int(prediction == STOP_INDEX and label != STOP_INDEX)
    return {"count": count, "accuracy": round(correct / max(count, 1), 8), "false_stop_count": false_stop, "unknown_count": unknown_count, "unknown_abstain_rate": round(unknown_abstain / max(unknown_count, 1), 8) if unknown_count else 1.0}


def _evaluate_variant(name: str, mode: str, model: nn.Module, train_count: int, lm_replay_count: int, unknown_hint_count: int, history: list[dict[str, Any]], data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, started: float, *, resumed_from_checkpoint: bool = False) -> dict[str, Any]:
    before_model = _load_body(vocabulary, device)
    language_canary = compare_causal_lm_canary(before_model, model.base, data["language_canary"], vocabulary, device=device)
    del before_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {"variant": name, "mode": mode, "train_count": train_count, "action_train_count": len(data["action_train"]), "lm_replay_count": lm_replay_count, "unknown_hint_count": unknown_hint_count, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "synthetic_holdout": _action_metrics(model, data["synthetic_holdout"], vocabulary, device), "real_pg136_holdout": _action_metrics(model, data["real_holdout"], vocabulary, device), "evaluation_only_surface_unknown": _action_metrics(model, data["surface_unknown"], vocabulary, device), "surface_lm": _lm_metrics(model.base, data["surface_lm"], vocabulary, device), "language_canary": language_canary, "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3), "resumed_from_checkpoint": resumed_from_checkpoint}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg154-multisource-action-replay-v1", "variant": name, "mode": mode, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _train_variant(name: str, mode: str, model: nn.Module, data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    train_rows = list(data["action_train"])
    lm_weight = 0.0
    guard_weight = 0.0
    unknown_weight = 0.0
    if mode in {"lm_anchor", "false_stop_guard"}:
        lm_weight = 0.5
        train_rows += data["lm_replay"]
    if mode == "false_stop_guard":
        guard_weight = 0.20
        unknown_weight = 0.35
        train_rows += data["unknown_hints"]
    loader = DataLoader(_ActionDataset(train_rows, vocabulary), batch_size=32, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        action_sum = 0.0
        guard_sum = 0.0
        unknown_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_valid = batch["action_valid"].to(device)
            unknown_hint = batch["unknown_hint"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_logits = model.action_logits(ids, mask)
                if bool(action_valid.any()):
                    action_loss = nn.functional.cross_entropy(action_logits[action_valid], labels[action_valid])
                else:
                    action_loss = action_logits.new_zeros(())
                loss = action_loss
                lm_loss = action_logits.new_zeros(())
                auxiliary = action_logits.new_zeros(())
                if lm_weight:
                    lm_logits, auxiliary = pg152._lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                    targets = ids[:, 1:]
                    lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                    loss = loss + lm_weight * lm_loss + 0.01 * auxiliary
                false_stop_loss = action_logits.new_zeros(())
                if guard_weight and bool(action_valid.any()):
                    non_stop = action_valid & labels.ne(STOP_INDEX)
                    if bool(non_stop.any()):
                        alternatives = action_logits.clone()
                        alternatives[:, STOP_INDEX] = torch.finfo(alternatives.dtype).min
                        safe_max = alternatives.max(dim=-1).values
                        false_stop_loss = nn.functional.softplus(action_logits[:, STOP_INDEX] - safe_max)[non_stop].mean()
                        loss = loss + guard_weight * false_stop_loss
                unknown_loss = action_logits.new_zeros(())
                if unknown_weight and bool(unknown_hint.any()):
                    unknown_loss = nn.functional.cross_entropy(action_logits[unknown_hint], torch.full((int(unknown_hint.sum().item()),), UNKNOWN_INDEX, dtype=torch.long, device=device))
                    loss = loss + unknown_weight * unknown_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            action_sum += float(action_loss.item())
            guard_sum += float(false_stop_loss.item())
            unknown_sum += float(unknown_loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "action_loss": round(action_sum / max(len(loader), 1), 8), "false_stop_guard_loss": round(guard_sum / max(len(loader), 1), 8), "unknown_hint_loss": round(unknown_sum / max(len(loader), 1), 8), "synthetic_dev": _action_metrics(model, data["action_dev"], vocabulary, device), "real_dev": _action_metrics(model, data["real_holdout"], vocabulary, device)})
        print(json.dumps({"variant": name, "epoch": epoch, "synthetic_dev_accuracy": history[-1]["synthetic_dev"]["accuracy"], "synthetic_dev_false_stop": history[-1]["synthetic_dev"]["false_stop_count"]}, ensure_ascii=False), flush=True)
    return _evaluate_variant(name, mode, model, len(train_rows), len(data["lm_replay"]) if lm_weight else 0, len(data["unknown_hints"]) if unknown_weight else 0, history, data, vocabulary, device, started)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = pg148._Vocabulary(list(_load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")["vocabulary"]))
    data, stats = _build_data(vocabulary)
    variants = [("action_only", "action_only"), ("lm_anchor", "lm_anchor"), ("false_stop_guard", "false_stop_guard")]
    results = []
    for name, mode in variants:
        model = pg152._MoEActionModel(_load_body(vocabulary, device), 512).to(device)
        checkpoint_path = ARTIFACT_DIR / f"{name}.pt"
        if checkpoint_path.exists() and name != "false_stop_guard":
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
            lm_count = len(data["lm_replay"]) if mode in {"lm_anchor", "false_stop_guard"} else 0
            unknown_count = len(data["unknown_hints"]) if mode == "false_stop_guard" else 0
            train_count = len(data["action_train"]) + lm_count + unknown_count
            print(json.dumps({"status": "resuming_evaluation_from_checkpoint", "variant": name, "checkpoint": str(checkpoint_path)}, ensure_ascii=False), flush=True)
            results.append(_evaluate_variant(name, mode, model, train_count, lm_count, unknown_count, [], data, vocabulary, device, time.perf_counter(), resumed_from_checkpoint=True))
        else:
            print(json.dumps({"status": "starting_variant", "variant": name, "mode": mode, "device": str(device)}, ensure_ascii=False), flush=True)
            results.append(_train_variant(name, mode, model, data, vocabulary, device))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"protocol_id": "pg-pk-154-multisource-action-replay-v1", "schema_version": "pg154-multisource-action-replay-report-v1", "status": "completed_pg154_multisource_action_replay", "device": str(device), "seed": SEED, "source": stats, "variants": results, "objective": "multi_source_real_trace_action_head_replay_and_false_stop_suppression", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "pg146_training_labels_used": False, "labels_in_lm_input": False, "old_holdout_used_for_optimization": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg154-multisource-action-replay-dataset-v1", "source": stats, "variants": {"action_only": {"lm_anchor": False, "false_stop_guard": False}, "lm_anchor": {"lm_anchor": True, "false_stop_guard": False}, "false_stop_guard": {"lm_anchor": True, "false_stop_guard": True}}, "holdouts": {"synthetic": len(data["synthetic_holdout"]), "real_pg136": len(data["real_holdout"]), "evaluation_only_surface_unknown": len(data["surface_unknown"]), "language_canary": len(data["language_canary"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-154-multisource-action-replay-v1", "schema_version": "pg154-multisource-action-replay-protocol-v1", "objective": report["objective"], "variants": [name for name, _ in variants], "real_fraction": REAL_FRACTION, "replay_count": REPLAY_COUNT, "epochs": EPOCHS, "false_stop_guard": {"non_stop_margin": "softplus(stop_logit - max_safe_logit)", "unknown_hint_target": "abstain_unknown_oracle"}, "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "synthetic_accuracy": row["synthetic_holdout"]["accuracy"], "synthetic_false_stop": row["synthetic_holdout"]["false_stop_count"], "real_accuracy": row["real_pg136_holdout"]["accuracy"], "real_false_stop": row["real_pg136_holdout"]["false_stop_count"], "surface_unknown_abstain": row["evaluation_only_surface_unknown"]["unknown_abstain_rate"], "forgetting": row["language_canary"]["catastrophic_forgetting_detected"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
