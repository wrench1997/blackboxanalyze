"""PG-164: 100M-class causal Transformer capacity pressure.

The PG-163 mixed corpus is held fixed.  This run changes only model capacity
to an approximately 100M-parameter Transformer, with gradient accumulation to
fit the local CUDA budget.  It measures ordinary abstract-language holdout,
typed Rule-IR holdout, and whether the larger body damages the base corpus.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg164-xxl-capacity-v1"
SOURCE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
SOURCE_REPORT = RESEARCH / "pg163_large_typed_mix_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg164_xxl_capacity_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg164_xxl_capacity_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg164_xxl_capacity_report_v1.md"
MAX_LEN = 128
SEED = 16401
CONFIG = {"d_model": 1024, "nhead": 16, "layers": 8, "batch_size": 8, "gradient_accumulation_steps": 2, "epochs": 2, "lr": 7e-5}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.rows = rows
        self.stoi = stoi

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        ids = [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]]
        return {"ids": ids, "row_id": row.get("row_id", str(index))}


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {"ids": ids, "mask": ids.ne(0)}


def _metrics(model: CausalTraceTransformer, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
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
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum")
            total_loss += float(loss.item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": round(mean_loss, 8), "perplexity": round(math.exp(min(mean_loss, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    dataset = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    train_rows = list(dataset["train_rows"])
    base_dev_rows = list(dataset["base_dev_rows"])
    base_holdout_rows = list(dataset["base_holdout_rows"])
    typed_holdout_rows = list(dataset["typed_holdout_rows"])
    vocabulary = list(dataset["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocabulary)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTraceTransformer(len(stoi), d_model=CONFIG["d_model"], nhead=CONFIG["nhead"], layers=CONFIG["layers"], max_len=MAX_LEN).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loader = DataLoader(_TraceDataset(train_rows, stoi), batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=_collate)
    base_dev_loader = DataLoader(_TraceDataset(base_dev_rows, stoi), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=_collate)
    base_holdout_loader = DataLoader(_TraceDataset(base_holdout_rows, stoi), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=_collate)
    typed_holdout_loader = DataLoader(_TraceDataset(typed_holdout_rows, stoi), batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        token_sum = 0
        for index, batch in enumerate(train_loader, start=1):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(ids[:, :-1], mask[:, :-1])
                targets = ids[:, 1:]
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                scaled_loss = loss / CONFIG["gradient_accumulation_steps"]
            scaler.scale(scaled_loss).backward()
            valid = int(targets.ne(0).sum().item())
            loss_sum += float(loss.detach().cpu()) * valid
            token_sum += valid
            if index % CONFIG["gradient_accumulation_steps"] == 0 or index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_dev": _metrics(model, base_dev_loader, device), "typed_holdout": _metrics(model, typed_holdout_loader, device)})
    result = {
        "variant": "xxl_typed_mix",
        "config": CONFIG,
        "parameter_count": parameter_count,
        "train": _metrics(model, train_loader, device),
        "base_dev": _metrics(model, base_dev_loader, device),
        "base_holdout": _metrics(model, base_holdout_loader, device),
        "typed_holdout": _metrics(model, typed_holdout_loader, device),
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ARTIFACT_DIR / "xxl_typed_mix.pt"
    torch.save({"schema_version": "pg164-xxl-capacity-v1", "variant": "xxl_typed_mix", "config": CONFIG, "vocabulary": vocabulary, "model_state_dict": model.state_dict()}, checkpoint)
    result["checkpoint"] = str(checkpoint.relative_to(ROOT))
    protocol = {
        "protocol_id": "pg-pk-164-xxl-capacity-v1",
        "schema_version": "pg164-xxl-capacity-protocol-v1",
        "source_dataset": str(SOURCE_DATASET.relative_to(ROOT)),
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "model_contract": {"feature_type": "causal_rule_ir_tokens", "max_len": MAX_LEN, "family_labels_in_tokens": False, "oracle_labels_in_tokens": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False},
        "capacity_contract": CONFIG,
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False},
        "training_artifact_promotion_allowed": False,
        "memory_promotion_allowed": False,
    }
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    report = {
        "schema_version": "pg164-xxl-capacity-report-v1",
        "protocol_id": "pg-pk-164-xxl-capacity-v1",
        "status": "completed_pg164_xxl_capacity",
        "scope": {"claim": "capacity pressure for abstract Rule-IR language modeling", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)},
        "dataset": {"source": str(SOURCE_DATASET.relative_to(ROOT)), "train_count": len(train_rows), "base_dev_count": len(base_dev_rows), "base_holdout_count": len(base_holdout_rows), "typed_holdout_count": len(typed_holdout_rows), "vocabulary_size": len(vocabulary)},
        "result": result,
        "comparison": {"pg163_xl_parameter_count": 57160937, "pg163_xl_typed_holdout_perplexity": 1.58142564, "pg163_xl_base_holdout_perplexity": 2.57545756, "interpretation": "XXL 结果只有在相同 token/data/epoch 与多 seed 复验后才能归因于容量；本轮不宣称漏洞能力提升。"},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False},
        "safety": protocol["safety"],
        "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dataset_sha256": dataset.get("dataset_sha256"), "protocol_sha256": protocol["protocol_sha256"]},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-164 XXL 容量压力实验",
        "",
        f"- parameters: **{parameter_count}**；device: **{device}**",
        f"- base holdout PPL: **{result['base_holdout']['perplexity']}**",
        f"- typed holdout PPL / next-token accuracy: **{result['typed_holdout']['perplexity']} / {result['typed_holdout']['next_token_accuracy']}**",
        "",
        "本轮只验证大容量 causal Rule-IR 表示学习；checkpoint 不进入生产能力或长期记忆。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "parameter_count": parameter_count, "base_holdout": result["base_holdout"], "typed_holdout": result["typed_holdout"], "elapsed_seconds": result["elapsed_seconds"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
