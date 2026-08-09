"""PG-151: data-augmented top-1 MoE capacity experiment.

This is a model-training experiment over abstract trace tokens.  It compares
two large sparse Mixture-of-Experts Transformer configurations while keeping
the PG-147 holdout untouched.  The extra rows are generated from the same
Rule-IR grammar with a fresh seed; they contain no raw payload, response body,
external URL, or evaluator state.

The experiment deliberately reports three things separately: language-model
next-token fit, routing behaviour, and transfer to the small real-surface
projection.  A lower perplexity is not treated as proof of vulnerability
detection capability.
"""

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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from app.moe_trace_transformer import MoETraceTransformer  # noqa: E402
from run_pg147_model_capacity_sweep import (  # noqa: E402
    _collate,
    _generated_sequence,
)


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg151-moe-capacity-v1"
REPORT = RESEARCH / "pg151_moe_capacity_report_v1.json"
DATASET = RESEARCH / "pg151_moe_capacity_dataset_v1.json"
PROTOCOL = RESEARCH / "pg151_moe_capacity_protocol_v1.json"
PG147_DATASET = RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json"
PG146_DATASET = RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json"
SEED = 15101
MAX_LEN = 128
EXTRA_TARGET = 8000
SPECIAL = ("[PAD]", "[UNK]")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in raw:
        rows: list[dict[str, Any]] = []
        for split, split_rows in raw["splits"].items():
            for row in split_rows:
                if isinstance(row, dict) and isinstance(row.get("tokens"), list):
                    copied = {"row_id": str(row.get("row_id", f"{split}-{len(rows)}")), "tokens": [str(token) for token in row["tokens"][:MAX_LEN]], "source": str(row.get("source", "pg147")), "split": split}
                    rows.append(copied)
        return rows
    if "rows" in raw:
        return [
            {"row_id": str(row.get("row_id", f"row-{index}")), "tokens": [str(token) for token in row["tokens"][:MAX_LEN]], "source": str(row.get("source", "external"))}
            for index, row in enumerate(raw["rows"])
            if isinstance(row, dict) and isinstance(row.get("tokens"), list)
        ]
    return []


def _build_augmented_corpus() -> tuple[list[dict[str, Any]], dict[str, int]]:
    original = _load_rows(PG147_DATASET)
    train = [row for row in original if row.get("split") == "train"]
    dev = [row for row in original if row.get("split") == "dev"]
    holdout = [row for row in original if row.get("split") == "holdout"]
    known = {tuple(row["tokens"]): row for row in original}
    rng = random.Random(SEED)
    generated: list[dict[str, Any]] = []
    attempts = 0
    while len(generated) < EXTRA_TARGET and attempts < EXTRA_TARGET * 30:
        attempts += 1
        sequence = tuple(_generated_sequence(rng, 200000 + len(generated)))
        if sequence in known:
            continue
        row = {"row_id": f"pg151-generated-{len(generated):06d}", "tokens": list(sequence), "source": "pg151_seeded_rule_ir_augmentation", "split": "train"}
        known[sequence] = row
        generated.append(row)
    train.extend(generated)
    rng.shuffle(train)
    corpus = train + dev + holdout
    return corpus, {"base_count": len(original), "base_train_count": len(train) - len(generated), "generated_count": len(generated), "train_count": len(train), "dev_count": len(dev), "holdout_count": len(holdout), "unique_count": len(known), "duplicate_count": len(original) + len(generated) - len(known), "augmentation_attempts": attempts}


class _TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.rows = rows
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        ids = [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]]
        return {"ids": ids, "row_id": row["row_id"]}


def _routing_entropy(loads: torch.Tensor) -> float:
    probabilities = loads.float().clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return float(entropy.mean().item())


def _metrics(model: MoETraceTransformer, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    aux_values: list[float] = []
    load_values: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            hidden, aux, loads = model.encode(ids[:, :-1], mask[:, :-1])
            logits = model.lm_head(hidden)
            targets = ids[:, 1:]
            valid = targets.ne(0)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
            aux_values.append(float(aux.item()))
            load_values.append(loads.detach().cpu().mean(dim=0))
    mean_loss = total_loss / max(total_tokens, 1)
    mean_load = torch.stack(load_values).mean(dim=0) if load_values else torch.zeros(model.n_experts, dtype=torch.float32)
    return {"loss": round(mean_loss, 8), "perplexity": round(math.exp(min(mean_loss, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens, "auxiliary_balance_loss": round(sum(aux_values) / max(len(aux_values), 1), 8), "expert_load": [round(float(value), 8) for value in mean_load.tolist()], "routing_entropy": round(_routing_entropy(mean_load.unsqueeze(0)), 8)}


def _train_variant(name: str, config: dict[str, int | float], train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], real_rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    model = MoETraceTransformer(len(stoi), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), n_experts=int(config["n_experts"]), expert_ff=int(config["expert_ff"]), max_len=MAX_LEN).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(_TraceDataset(train_rows, stoi), batch_size=batch_size, shuffle=True, collate_fn=_collate, drop_last=False)
    dev_loader = DataLoader(_TraceDataset(dev_rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    holdout_loader = DataLoader(_TraceDataset(holdout_rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    real_loader = DataLoader(_TraceDataset(real_rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    aux_weight = float(config["aux_weight"])
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        aux_sum = 0.0
        token_sum = 0
        for batch in train_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                hidden, aux, _ = model.encode(ids[:, :-1], mask[:, :-1])
                logits = model.lm_head(hidden)
                targets = ids[:, 1:]
                ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                loss = ce + aux_weight * aux
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            valid_tokens = int(targets.ne(0).sum().item())
            loss_sum += float(loss.item()) * valid_tokens
            ce_sum += float(ce.item()) * valid_tokens
            aux_sum += float(aux.item()) * valid_tokens
            token_sum += valid_tokens
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(token_sum, 1), 8), "train_ce": round(ce_sum / max(token_sum, 1), 8), "train_aux": round(aux_sum / max(token_sum, 1), 8), "dev": _metrics(model, dev_loader, device)})
        print(json.dumps({"variant": name, "epoch": epoch, "dev_perplexity": history[-1]["dev"]["perplexity"], "dev_accuracy": history[-1]["dev"]["next_token_accuracy"]}, ensure_ascii=False), flush=True)
    result = {"variant": name, "config": config, "parameter_count": parameter_count, "train": _metrics(model, train_loader, device), "dev": _metrics(model, dev_loader, device), "holdout": _metrics(model, holdout_loader, device), "real_surface_projection": _metrics(model, real_loader, device), "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg151-moe-trace-transformer-v1", "variant": name, "config": config, "vocabulary": sorted(stoi, key=stoi.get), "model_state_dict": model.state_dict()}, checkpoint)
    result["checkpoint"] = str(checkpoint.relative_to(ROOT))
    return result


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    corpus, corpus_stats = _build_augmented_corpus()
    train_rows = [row for row in corpus if row.get("split") == "train"]
    dev_rows = [row for row in corpus if row.get("split") == "dev"]
    holdout_rows = [row for row in corpus if row.get("split") == "holdout"]
    real_rows = _load_rows(PG146_DATASET)
    tokens = sorted({token for row in train_rows for token in row["tokens"]})
    itos = list(SPECIAL) + [token for token in tokens if token not in SPECIAL]
    stoi = {token: index for index, token in enumerate(itos)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants: dict[str, dict[str, int | float]] = {
        "moe_large_4e": {"d_model": 512, "nhead": 8, "layers": 4, "n_experts": 4, "expert_ff": 2048, "batch_size": 32, "epochs": 3, "lr": 1.5e-4, "aux_weight": 0.01},
        "moe_xl_4e": {"d_model": 768, "nhead": 12, "layers": 6, "n_experts": 4, "expert_ff": 3072, "batch_size": 16, "epochs": 2, "lr": 1.0e-4, "aux_weight": 0.01},
        "moe_xl_8e": {"d_model": 768, "nhead": 12, "layers": 6, "n_experts": 8, "expert_ff": 2048, "batch_size": 16, "epochs": 2, "lr": 1.0e-4, "aux_weight": 0.01},
    }
    results = []
    for name, config in variants.items():
        print(json.dumps({"status": "starting_variant", "variant": name, "device": str(device), "parameter_estimate": "initializing"}, ensure_ascii=False), flush=True)
        results.append(_train_variant(name, config, train_rows, dev_rows, holdout_rows, real_rows, stoi, device))
    report = {"protocol_id": "pg-pk-151-moe-capacity-v1", "schema_version": "pg151-moe-capacity-report-v1", "status": "completed_pg151_moe_capacity_sweep", "device": str(device), "seed": SEED, "corpus": {**corpus_stats, "vocabulary_size": len(stoi), "max_length": MAX_LEN, "real_surface_eval_count": len(real_rows), "source_files": ["research/pg147_model_capacity_sweep_dataset_v1.json", "research/pg146_public_lab_replay_model_dataset_v1.json"]}, "variants": results, "objective": "large_model_capacity_and_sparse_routing_comparison_with_seeded_data_augmentation", "architecture_notes": {"routing": "top_1_token_choice", "expert_balance_loss": "switch_style_importance_times_load", "labels_in_lm_input": False, "raw_payload_or_response_in_corpus": False, "external_network_targets": False}, "capability_claim_allowed": False, "long_term_memory_promotion_allowed": False, "training_artifact_promotion_allowed": False, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg151-moe-capacity-dataset-v1", "corpus_stats": corpus_stats, "vocabulary": itos, "splits": {"train": train_rows, "dev": dev_rows, "holdout": holdout_rows}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-151-moe-capacity-v1", "schema_version": "pg151-moe-capacity-protocol-v1", "objective": report["objective"], "data_policy": {"seeded_rule_ir_augmentation": True, "raw_payloads": False, "raw_responses": False, "holdout_untouched": True, "labels_in_next_token_input": False}, "variants": variants, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "corpus": report["corpus"], "variants": [{"variant": row["variant"], "parameter_count": row["parameter_count"], "holdout_perplexity": row["holdout"]["perplexity"], "holdout_accuracy": row["holdout"]["next_token_accuracy"], "real_surface_perplexity": row["real_surface_projection"]["perplexity"], "routing_entropy": row["holdout"]["routing_entropy"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
