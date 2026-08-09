"""PG-175: hard-gated joint routing/loss search on the 160M checkpoint."""

from __future__ import annotations

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg174_full_holdout_routing import _load_sources  # noqa: E402


RESEARCH = ROOT / "research"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PG168_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PG170_DATASET = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PG172_DATASET = RESEARCH / "pg172_third_generator_dataset_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg173-matched-budget-capacity-v1" / "160m_epoch4.pt"
DATASET_PATH = RESEARCH / "pg175_joint_routing_loss_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg175_joint_routing_loss_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg175_joint_routing_loss_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg175_joint_routing_loss_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg175-joint-routing-loss-v1"
SEED = 17501
ROWS_PER_SOURCE = 250
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_LEN = 128
BASELINE_AGGREGATE = 2.51077335
PER_SPLIT_TOLERANCE = 0.005


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _Dataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.rows = rows
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"ids": [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]], "source": row.get("source", "eval")}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    sources = []
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        sources.append(item["source"])
    return {"ids": ids, "mask": ids.ne(0), "source": sources}


def _metrics(model: nn.Module, rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, float | int]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            logits = model(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            valid = targets.ne(0)
            total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 8), "perplexity": round(math.exp(min(mean, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> CausalTraceTransformer:
    config = checkpoint["config"]
    model = CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _make_train_rows(source_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, (source, pool) in enumerate(source_rows.items()):
        rows.extend(random.Random(SEED + offset).sample(pool, ROWS_PER_SOURCE))
    return rows


def _route(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "uniform":
        ordered = list(rows)
        random.Random(SEED).shuffle(ordered)
        return ordered
    if mode == "old_first":
        return sorted(rows, key=lambda row: (row["source"] != "pg163_base", row["row_id"]))
    groups = {source: [row for row in rows if row["source"] == source] for source in {row["source"] for row in rows}}
    order = ["pg163_base", "pg168_slots", "pg163_base", "pg170_generator", "pg163_base", "pg172_generator"]
    cursors = {source: 0 for source in groups}
    result: list[dict[str, Any]] = []
    while any(cursors[source] < len(group) for source, group in groups.items()):
        for source in order:
            if cursors[source] >= len(groups[source]):
                continue
            take = min(TRAIN_BATCH_SIZE, len(groups[source]) - cursors[source])
            result.extend(groups[source][cursors[source] : cursors[source] + take])
            cursors[source] += take
    return result


def _train(name: str, checkpoint: dict[str, Any], rows: list[dict[str, Any]], weights: dict[str, float], lr: float, route_mode: str, eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _load_model(checkpoint, device)
    ordered = _route(rows, route_mode)
    loader = DataLoader(_Dataset(ordered, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    started = time.perf_counter()
    model.train()
    loss_sum = 0.0
    token_sum = 0
    for batch in loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="none").reshape(targets.shape)
            valid = targets.ne(0)
            batch_weights = torch.tensor([weights.get(source, 1.0) for source in batch["source"]], device=device, dtype=losses.dtype).unsqueeze(1)
            loss = (losses * valid * batch_weights).sum() / (valid * batch_weights).sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        valid_count = int(valid.sum().item())
        loss_sum += float(loss.detach().cpu()) * valid_count
        token_sum += valid_count
    result = {"strategy": name, "route_mode": route_mode, "weights": weights, "lr": lr, "train_row_count": len(ordered), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "pg168_ood": _metrics(model, eval_rows["pg168_ood"], stoi, device), "pg170_ood": _metrics(model, eval_rows["pg170_ood"], stoi, device), "pg172_ood": _metrics(model, eval_rows["pg172_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg175-joint-routing-loss-v1", "strategy": name, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(value: dict[str, Any]) -> float:
    return round(sum(value[key]["perplexity"] for key in ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood")) / 5.0, 8)


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    source_rows, _, source_hashes = _load_sources()
    train_rows = _make_train_rows(source_rows)
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    p168 = json.loads(PG168_DATASET.read_text(encoding="utf-8"))
    p170 = json.loads(PG170_DATASET.read_text(encoding="utf-8"))
    p172 = json.loads(PG172_DATASET.read_text(encoding="utf-8"))
    eval_rows = {"base_holdout": [{"tokens": row["tokens"]} for row in base["base_holdout_rows"]], "typed_holdout": [{"tokens": row["tokens"]} for row in base["typed_holdout_rows"]], "pg168_ood": [{"tokens": row["tokens"]} for row in p168["rows"] if row["split"] == "ood"], "pg170_ood": [{"tokens": row["tokens"]} for row in p170["rows"] if row["split"] == "ood"], "pg172_ood": [{"tokens": row["tokens"]} for row in p172["rows"] if row["split"] == "ood"]}
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    baseline = {key: _metrics(baseline_model, rows, stoi, device) for key, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    specs = {
        "low_lr_replay": ({"pg163_base": 2.0, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0}, 5e-6, "uniform"),
        "balanced_low_lr": ({"pg163_base": 1.5, "pg168_slots": 1.5, "pg170_generator": 1.5, "pg172_generator": 1.5}, 5e-6, "uniform"),
        "old_first_weighted": ({"pg163_base": 3.0, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0}, 1e-5, "old_first"),
        "routed_low_lr": ({"pg163_base": 1.5, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0}, 5e-6, "interleave"),
    }
    results: dict[str, Any] = {}
    for name, (weights, lr, route_mode) in specs.items():
        results[name] = _train(name, checkpoint, train_rows, weights, lr, route_mode, eval_rows, stoi, device)
    baseline_aggregate = _aggregate(baseline)
    aggregate = {name: _aggregate(value) for name, value in results.items()}
    split_gate = {name: all(value[key]["perplexity"] <= baseline[key]["perplexity"] * (1.0 + PER_SPLIT_TOLERANCE) for key in ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood")) for name, value in results.items()}
    strict_gate = {name: aggregate[name] < baseline_aggregate and split_gate[name] for name in results}
    selected = min((name for name, ok in strict_gate.items() if ok), key=lambda name: aggregate[name], default=None)
    dataset = {"schema_version": "pg175-joint-routing-loss-dataset-v1", "purpose": "full-holdout-gated joint replay/loss search", "source_dataset_sha256": source_hashes, "rows_per_source": ROWS_PER_SOURCE, "train_row_count": len(train_rows), "full_eval_counts": {name: len(rows) for name, rows in eval_rows.items()}, "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False}}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    report = {"schema_version": "pg175-joint-routing-loss-report-v1", "protocol_id": "pg-pk-175-joint-routing-loss-v1", "status": "completed_pg175_joint_routing_loss_search", "scope": {"claim": "full-holdout-gated routing/loss search on 160M epoch4", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"train_row_count": len(train_rows), "full_eval_counts": dataset["full_eval_counts"], "source_dataset_sha256": source_hashes}, "baseline": baseline, "baseline_aggregate_ppl": baseline_aggregate, "variants": results, "aggregate_ppl": aggregate, "split_gate": split_gate, "strict_gate": strict_gate, "selection": {"baseline_hard_gate": True, "per_split_tolerance": PER_SPLIT_TOLERANCE, "selected_variant": selected, "promotion_allowed": selected is not None, "vulnerability_claim_allowed": False}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-175-joint-routing-loss-v1", "schema_version": "pg175-joint-routing-loss-protocol-v1", "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "rows_per_source": ROWS_PER_SOURCE, "full_holdout": dataset["full_eval_counts"], "strategies": {name: {"weights": weights, "lr": lr, "route_mode": route_mode} for name, (weights, lr, route_mode) in specs.items()}, "hard_gate": {"baseline_aggregate_ppl": baseline_aggregate, "per_split_tolerance": PER_SPLIT_TOLERANCE, "selected_variant": selected}, "promotion": {"training_artifact_promotion_allowed": selected is not None, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-175 joint routing/loss search", "", f"- baseline aggregate PPL: **{baseline_aggregate}**", f"- variant aggregate PPL: **{aggregate}**", f"- strict gate: **{strict_gate}**", f"- selected: **{selected}**", "", "完整 holdout 是硬门；没有超过基线的分支不会晋级。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "baseline_aggregate_ppl": baseline_aggregate, "aggregate_ppl": aggregate, "strict_gate": strict_gate, "selected_variant": selected, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
