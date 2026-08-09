"""PG-174: full-holdout replay weighting and source routing on the 160M model."""

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
from run_pg173_matched_budget_capacity import _load_sources  # noqa: E402


RESEARCH = ROOT / "research"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PG168_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PG170_DATASET = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PG172_DATASET = RESEARCH / "pg172_third_generator_dataset_v1.json"
REFERENCE_101M = ROOT / "artifacts" / "pg173-matched-budget-capacity-v1" / "101m_epoch4.pt"
REFERENCE_160M = ROOT / "artifacts" / "pg173-matched-budget-capacity-v1" / "160m_epoch4.pt"
DATASET_PATH = RESEARCH / "pg174_full_holdout_routing_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg174_full_holdout_routing_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg174_full_holdout_routing_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg174_full_holdout_routing_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg174-full-holdout-routing-v1"
SEED = 17401
ROWS_PER_SOURCE = 250
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_LEN = 128


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
    source = []
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        source.append(item["source"])
    return {"ids": ids, "mask": ids.ne(0), "source": source}


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
    return CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)


def _make_train_rows(source_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, (source, pool) in enumerate(source_rows.items()):
        sampled = random.Random(SEED + offset).sample(pool, ROWS_PER_SOURCE)
        rows.extend(sampled)
    return rows


def _routed_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {source: [row for row in rows if row["source"] == source] for source in {row["source"] for row in rows}}
    order = ["pg163_base", "pg168_slots", "pg163_base", "pg170_generator", "pg163_base", "pg172_generator"]
    result: list[dict[str, Any]] = []
    cursors = {source: 0 for source in groups}
    while any(cursors[source] < len(group) for source, group in groups.items()):
        for source in order:
            group = groups[source]
            if cursors[source] >= len(group):
                continue
            take = min(TRAIN_BATCH_SIZE, len(group) - cursors[source])
            result.extend(group[cursors[source] : cursors[source] + take])
            cursors[source] += take
    return result


def _train(name: str, checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device, weights: dict[str, float]) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _load_model(checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(_Dataset(rows, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
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
    result = {"strategy": name, "weights": weights, "train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "pg168_ood": _metrics(model, eval_rows["pg168_ood"], stoi, device), "pg170_ood": _metrics(model, eval_rows["pg170_ood"], stoi, device), "pg172_ood": _metrics(model, eval_rows["pg172_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg174-full-holdout-routing-v1", "strategy": name, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint160 = torch.load(REFERENCE_160M, map_location="cpu")
    checkpoint101 = torch.load(REFERENCE_101M, map_location="cpu")
    source_rows, _, source_hashes = _load_sources()
    train_rows = _make_train_rows(source_rows)
    eval_base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    eval168 = json.loads(PG168_DATASET.read_text(encoding="utf-8"))
    eval170 = json.loads(PG170_DATASET.read_text(encoding="utf-8"))
    eval172 = json.loads(PG172_DATASET.read_text(encoding="utf-8"))
    eval_rows = {"base_holdout": [{"tokens": row["tokens"]} for row in eval_base["base_holdout_rows"]], "typed_holdout": [{"tokens": row["tokens"]} for row in eval_base["typed_holdout_rows"]], "pg168_ood": [{"tokens": row["tokens"]} for row in eval168["rows"] if row["split"] == "ood"], "pg170_ood": [{"tokens": row["tokens"]} for row in eval170["rows"] if row["split"] == "ood"], "pg172_ood": [{"tokens": row["tokens"]} for row in eval172["rows"] if row["split"] == "ood"]}
    vocab = list(checkpoint160["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocab)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_models: dict[str, Any] = {}
    for name, checkpoint in (("101m_epoch4", checkpoint101), ("160m_epoch4", checkpoint160)):
        model = _load_model(checkpoint, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        baseline_models[name] = {key: _metrics(model, rows, stoi, device) for key, rows in eval_rows.items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    variants = {"uniform": (train_rows, {source: 1.0 for source in source_rows}), "replay_weighted": (list(train_rows), {"pg163_base": 2.0, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0}), "source_routed": (_routed_order(train_rows), {"pg163_base": 1.5, "pg168_slots": 1.0, "pg170_generator": 1.0, "pg172_generator": 1.0})}
    results: dict[str, Any] = {}
    for name, (rows, weights) in variants.items():
        results[name] = _train(name, checkpoint160, rows, eval_rows, stoi, device, weights)
    report = {"schema_version": "pg174-full-holdout-routing-report-v1", "protocol_id": "pg-pk-174-full-holdout-routing-v1", "status": "completed_pg174_full_holdout_routing", "scope": {"claim": "full holdout loss weighting and source routing diagnostic on PG173 160M epoch4", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"train_row_count": len(train_rows), "source_rows_per_sample": ROWS_PER_SOURCE, "full_eval_counts": {name: len(rows) for name, rows in eval_rows.items()}, "source_dataset_sha256": source_hashes}, "baseline": baseline_models, "variants": results, "selection": {"promotion_allowed": False, "vulnerability_claim_allowed": False, "full_holdout_used": True, "objective": "minimize base+typed+three_ood aggregate without trading away any split"}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "reference_101m_sha256": hashlib.sha256(REFERENCE_101M.read_bytes()).hexdigest(), "reference_160m_sha256": hashlib.sha256(REFERENCE_160M.read_bytes()).hexdigest()}}
    def aggregate(value: dict[str, Any]) -> float:
        return round(sum(value[key]["perplexity"] for key in ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood")) / 5.0, 8)
    report["selection"]["aggregate_ppl"] = {name: aggregate(value) for name, value in results.items()}
    report["selection"]["baseline_101m_aggregate_ppl"] = aggregate(report["baseline"]["101m_epoch4"])
    report["selection"]["baseline_160m_aggregate_ppl"] = aggregate(report["baseline"]["160m_epoch4"])
    best_variant = min(results, key=lambda name: report["selection"]["aggregate_ppl"][name])
    report["selection"]["best_variant"] = best_variant
    report["selection"]["best_variant_beats_160m_baseline"] = report["selection"]["aggregate_ppl"][best_variant] < report["selection"]["baseline_160m_aggregate_ppl"]
    report["selection"]["selected_variant"] = best_variant if report["selection"]["best_variant_beats_160m_baseline"] else None
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-174-full-holdout-routing-v1", "schema_version": "pg174-full-holdout-routing-protocol-v1", "reference_101m": str(REFERENCE_101M.relative_to(ROOT)), "reference_160m": str(REFERENCE_160M.relative_to(ROOT)), "source_datasets": [str(path.relative_to(ROOT)) for path in (BASE_DATASET, PG168_DATASET, PG170_DATASET, PG172_DATASET)], "rows_per_source": ROWS_PER_SOURCE, "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE, "strategies": {name: {"weights": weights, "routing": name == "source_routed"} for name, (_, weights) in variants.items()}, "full_holdout": report["dataset"]["full_eval_counts"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-174 full holdout routing", "", f"- full eval counts: **{report['dataset']['full_eval_counts']}**", f"- baseline 160M aggregate PPL: **{report['selection']['baseline_160m_aggregate_ppl']}**", f"- variant aggregate PPL: **{report['selection']['aggregate_ppl']}**", f"- best variant: **{report['selection']['best_variant']}**", f"- beats baseline: **{report['selection']['best_variant_beats_160m_baseline']}**", "", "本轮使用完整 holdout；若分支未超过未训练基线，不标记为模型增强。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "full_eval_counts": report["dataset"]["full_eval_counts"], "baseline_160m_aggregate_ppl": report["selection"]["baseline_160m_aggregate_ppl"], "aggregate_ppl": report["selection"]["aggregate_ppl"], "best_variant": report["selection"]["best_variant"], "selected_variant": report["selection"]["selected_variant"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
