"""PG-173: matched token-budget multi-epoch capacity experiment.

Four family-free sources are balanced to 2,000 rows per source per epoch.
Both 101M and 160M models start from scratch and see exactly the same planned
rows for 1, 2, or 4 epochs.  The run is a language-model training diagnostic,
not a vulnerability detector.
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
BASE_PATH = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
PG168_PATH = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
PG170_PATH = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PG172_PATH = RESEARCH / "pg172_third_generator_dataset_v1.json"
REFERENCE_PATH = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
DATASET_PATH = RESEARCH / "pg173_matched_budget_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg173_matched_budget_capacity_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg173_matched_budget_capacity_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg173_matched_budget_capacity_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg173-matched-budget-capacity-v1"
SEED = 17301
ROWS_PER_SOURCE = 250
EPOCHS_TO_RUN = (1, 2, 4)
MAX_LEN = 128
BATCH_SIZE = 16
EVAL_LIMIT = 128


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _Dataset(Dataset[dict[str, list[int]]]):
    def __init__(self, rows: list[dict[str, Any]], stoi: dict[str, int]) -> None:
        self.ids = [[stoi.get(token, stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]] for row in rows]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"ids": self.ids[index]}


def _collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {"ids": ids, "mask": ids.ne(0)}


def _metrics(model: nn.Module, rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device) -> dict[str, float | int]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
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
            total_loss += float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            total_tokens += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 8), "perplexity": round(math.exp(min(mean, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _load_sources() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, str]]:
    base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
    p168 = json.loads(PG168_PATH.read_text(encoding="utf-8"))
    p170 = json.loads(PG170_PATH.read_text(encoding="utf-8"))
    p172 = json.loads(PG172_PATH.read_text(encoding="utf-8"))
    source_rows = {
        "pg163_base": [{"row_id": row.get("row_id", f"pg163-{index}"), "source": "pg163_base", "tokens": row["tokens"]} for index, row in enumerate(base["train_rows"])],
        "pg168_slots": [{"row_id": row["row_id"], "source": "pg168_slots", "tokens": row["tokens"]} for row in p168["rows"] if row["split"] == "train"],
        "pg170_generator": [{"row_id": row["row_id"], "source": "pg170_generator", "tokens": row["tokens"]} for row in p170["rows"] if row["split"] == "train"],
        "pg172_generator": [{"row_id": row["row_id"], "source": "pg172_generator", "tokens": row["tokens"]} for row in p172["rows"] if row["split"] == "train"],
    }
    hashes = {"pg163_base": base.get("dataset_sha256", ""), "pg168_slots": p168.get("dataset_sha256", ""), "pg170_generator": p170.get("dataset_sha256", ""), "pg172_generator": p172.get("dataset_sha256", "")}
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "pg168_ood": [{"tokens": row["tokens"]} for row in p168["rows"] if row["split"] == "ood"], "pg170_ood": [{"tokens": row["tokens"]} for row in p170["rows"] if row["split"] == "ood"], "pg172_ood": [{"tokens": row["tokens"]} for row in p172["rows"] if row["split"] == "ood"]}
    return source_rows, eval_rows, hashes


def _bounded_eval_rows(eval_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {name: random.Random(SEED + index).sample(rows, min(EVAL_LIMIT, len(rows))) for index, (name, rows) in enumerate(eval_rows.items())}


def _make_plan(source_rows: dict[str, list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    plan: list[list[dict[str, Any]]] = []
    for epoch in range(1, max(EPOCHS_TO_RUN) + 1):
        epoch_rows: list[dict[str, Any]] = []
        for offset, (source, rows) in enumerate(source_rows.items()):
            if len(rows) < ROWS_PER_SOURCE:
                raise RuntimeError(f"source {source} has only {len(rows)} rows")
            sampled = random.Random(SEED + epoch * 100 + offset).sample(rows, ROWS_PER_SOURCE)
            epoch_rows.extend(sampled)
        random.Random(SEED + epoch).shuffle(epoch_rows)
        plan.append(epoch_rows)
    return plan


def _make_model(vocab_size: int, config: dict[str, int], device: torch.device) -> CausalTraceTransformer:
    return CausalTraceTransformer(vocab_size, d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=MAX_LEN).to(device)


def _train(name: str, config: dict[str, int], epochs: int, plan: list[list[dict[str, Any]]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _make_model(len(stoi), config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    total_tokens = 0
    for epoch in range(epochs):
        rows = plan[epoch]
        loader = DataLoader(_Dataset(rows, stoi), batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)
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
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            valid = int(targets.ne(0).sum().item())
            loss_sum += float(loss.detach().cpu()) * valid
            token_sum += valid
        total_tokens += token_sum
        history.append({"epoch": epoch + 1, "train_loss": round(loss_sum / max(token_sum, 1), 8), "token_count": token_sum})
    result = {"strategy": name, "config": config, "epochs": epochs, "parameter_count": parameter_count, "train_row_count": epochs * len(plan[0]), "train_token_count": total_tokens, "history": history, "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "pg168_ood": _metrics(model, eval_rows["pg168_ood"], stoi, device), "pg170_ood": _metrics(model, eval_rows["pg170_ood"], stoi, device), "pg172_ood": _metrics(model, eval_rows["pg172_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg173-matched-budget-capacity-v1", "strategy": name, "config": config, "epochs": epochs, "vocabulary": list(stoi), "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    reference = torch.load(REFERENCE_PATH, map_location="cpu")
    source_rows, eval_rows, source_hashes = _load_sources()
    eval_rows = _bounded_eval_rows(eval_rows)
    plan = _make_plan(source_rows)
    vocab = list(reference["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocab)}
    for epoch_rows in plan:
        for row in epoch_rows:
            missing = [token for token in row["tokens"] if token not in stoi]
            if missing:
                raise RuntimeError(f"frozen vocabulary missing token: {missing[0]}")
    epoch_manifest = [{"epoch": index + 1, "row_count": len(rows), "token_count": sum(len(row["tokens"]) for row in rows), "source_counts": {source: sum(1 for row in rows if row["source"] == source) for source in source_rows}, "row_ids_sha256": _sha256_json([row["row_id"] for row in rows])} for index, rows in enumerate(plan)]
    dataset = {"schema_version": "pg173-matched-budget-dataset-v1", "purpose": "balanced multi-source abstract Rule-IR training budget", "sources": {source: {"row_count": len(rows), "dataset_sha256": source_hashes[source]} for source, rows in source_rows.items()}, "rows_per_source_per_epoch": ROWS_PER_SOURCE, "epoch_manifest": epoch_manifest, "training_contract": {"raw_payloads_stored": False, "raw_responses_stored": False, "vulnerability_labels_stored": False, "oracle_labels_stored": False, "family_labels_stored": False, "memory_promotion_allowed": False}, "vocabulary_size": len(vocab)}
    dataset["dataset_sha256"] = _sha256_json(dataset)
    _write(DATASET_PATH, dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = {"101m": {"d_model": 1024, "nhead": 16, "layers": 8}, "160m": {"d_model": 1152, "nhead": 18, "layers": 10}}
    results: dict[str, Any] = {}
    for capacity_name, config in configs.items():
        for epochs in EPOCHS_TO_RUN:
            name = f"{capacity_name}_epoch{epochs}"
            results[name] = _train(name, config, epochs, plan, eval_rows, stoi, device)
    report = {"schema_version": "pg173-matched-budget-capacity-report-v1", "protocol_id": "pg-pk-173-matched-budget-capacity-v1", "status": "completed_pg173_matched_budget_capacity", "scope": {"claim": "matched token-budget multi-epoch 101M/160M capacity diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"rows_per_source_per_epoch": ROWS_PER_SOURCE, "source_count": len(source_rows), "epoch_token_counts": [entry["token_count"] for entry in epoch_manifest], "total_four_epoch_tokens": sum(entry["token_count"] for entry in epoch_manifest), "evaluation_rows_per_split": EVAL_LIMIT, "evaluation_sampling_seed": SEED}, "capacity_variants": results, "comparison_contract": {"same_epoch_plans": True, "same_token_budget_per_epoch": True, "same_optimizer": True, "same_seed": True, "from_scratch": True}, "selection": {"promotion_allowed": False, "vulnerability_claim_allowed": False, "capacity_gain_requires_old_holdout_and_all_three_ood": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "reference_checkpoint_sha256": hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest(), "source_dataset_sha256": source_hashes, "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-173-matched-budget-capacity-v1", "schema_version": "pg173-matched-budget-capacity-protocol-v1", "reference_checkpoint": str(REFERENCE_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "rows_per_source_per_epoch": ROWS_PER_SOURCE, "epochs": list(EPOCHS_TO_RUN), "configs": configs, "optimizer": {"name": "AdamW", "lr": 7e-5, "weight_decay": 0.01, "batch_size": BATCH_SIZE}, "evaluation_rows_per_split": EVAL_LIMIT, "comparison_contract": report["comparison_contract"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-173 matched budget capacity", "", f"- rows/source/epoch: **{ROWS_PER_SOURCE}**", f"- epoch token counts: **{', '.join(map(str, report['dataset']['epoch_token_counts']))}**", *[f"- {name}: base/typed/PG168/PG170/PG172 OOD PPL = **{value['base_holdout']['perplexity']} / {value['typed_holdout']['perplexity']} / {value['pg168_ood']['perplexity']} / {value['pg170_ood']['perplexity']} / {value['pg172_ood']['perplexity']}**" for name, value in results.items()], "", "该轮只比较匹配 token budget 下的训练预算与容量，不产生漏洞标签，也不晋级长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "epoch_token_counts": report["dataset"]["epoch_token_counts"], "summary": {name: {"parameters": value["parameter_count"], "epochs": value["epochs"], "base_ppl": value["base_holdout"]["perplexity"], "typed_ppl": value["typed_holdout"]["perplexity"], "ood": [value["pg168_ood"]["perplexity"], value["pg170_ood"]["perplexity"], value["pg172_ood"]["perplexity"]]} for name, value in results.items()}, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
