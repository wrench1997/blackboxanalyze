"""PG-150: train a Large body with controlled real/synthetic action mixtures."""

from __future__ import annotations

import copy
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_pg148_large_model_posttraining as pg148  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg150-real-synthetic-mix-v1"
REPORT = RESEARCH / "pg150_real_synthetic_mix_report_v1.json"
DATASET = RESEARCH / "pg150_real_synthetic_mix_dataset_v1.json"
PROTOCOL = RESEARCH / "pg150_real_synthetic_mix_protocol_v1.json"
SEED = 15001
MAX_LEN = 128


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _lm_metrics(model: torch.nn.Module, rows: list[dict[str, Any]], vocab: pg148._Vocabulary, device: torch.device) -> dict[str, float]:
    if not rows:
        return {"perplexity": 0.0, "next_token_accuracy": 0.0, "token_count": 0}
    encoded = [vocab.encode(row["tokens"][:MAX_LEN]) for row in rows]
    width = max(len(seq) for seq in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, seq in enumerate(encoded):
        ids[index, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        logits = model(ids[:, :-1], ids[:, :-1].ne(0))
    targets = ids[:, 1:]
    valid = targets.ne(0)
    count = int(valid.sum().item())
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    accuracy = float(((logits.argmax(dim=-1) == targets) & valid).sum().item() / max(count, 1))
    return {"perplexity": round(float(torch.exp(loss).item()), 8), "next_token_accuracy": round(accuracy, 8), "token_count": count}


def _prepare() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pg148._Vocabulary, dict[str, int]]:
    pg149 = json.loads((RESEARCH / "pg149_causal_action_alignment_dataset_v1.json").read_text(encoding="utf-8"))
    synthetic = pg149["splits"]
    real_train, real_dev, real_holdout, vocab, stats = pg148._prepare_rows()
    pg146 = json.loads((RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json").read_text(encoding="utf-8"))
    real_surface = [{"tokens": row["tokens"], "row_id": row["row_id"], "label_index": 0, "typed_available": False, "surface_kind": "pg146_real_surface"} for row in pg146["rows"]]
    return synthetic["train"], synthetic["dev"], synthetic["holdout"], real_train, real_holdout, vocab, {**stats, "real_surface_count": len(real_surface)}


def _mix(real_train: list[dict[str, Any]], synthetic_train: list[dict[str, Any]], real_fraction: float) -> list[dict[str, Any]]:
    if real_fraction <= 0:
        return list(synthetic_train)
    desired_real = max(1, int(len(synthetic_train) * real_fraction / max(1.0 - real_fraction, 1e-6)))
    repeated = [copy.deepcopy(real_train[index % len(real_train)]) for index in range(desired_real)]
    for index, row in enumerate(repeated):
        row["row_id"] = f"real_repeat_{index}_{row['row_id']}"
    return list(synthetic_train) + repeated


def _train_variant(name: str, real_fraction: float, base: torch.nn.Module, config: dict[str, Any], train_rows: list[dict[str, Any]], synthetic_dev: list[dict[str, Any]], synthetic_holdout: list[dict[str, Any]], real_holdout: list[dict[str, Any]], surface_rows: list[dict[str, Any]], vocab: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    model = pg148._ActionModel(base, int(config["d_model"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    loader = DataLoader(pg148._Dataset(train_rows, vocab), batch_size=64, shuffle=True, collate_fn=pg148._collate)
    syn_dev_loader = DataLoader(pg148._Dataset(synthetic_dev, vocab), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    syn_hold_loader = DataLoader(pg148._Dataset(synthetic_holdout, vocab), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    real_hold_loader = DataLoader(pg148._Dataset(real_holdout, vocab), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, 4):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_loss = nn.functional.cross_entropy(model.action_logits(ids, mask), labels)
                lm_logits = model.base(ids[:, :-1], mask[:, :-1])
                lm_targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1), ignore_index=0)
                loss = action_loss + 0.5 * lm_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "synthetic_dev": pg148._metrics(model, syn_dev_loader, device), "real_holdout": pg148._metrics(model, real_hold_loader, device)})
    result = {
        "variant": name,
        "real_fraction": real_fraction,
        "train_count": len(train_rows),
        "real_rows_in_train": sum(1 for row in train_rows if str(row["row_id"]).startswith("real_repeat_")),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "synthetic_dev": pg148._metrics(model, syn_dev_loader, device),
        "synthetic_holdout": pg148._metrics(model, syn_hold_loader, device),
        "real_pg136_holdout": pg148._metrics(model, real_hold_loader, device),
        "real_surface_lm": _lm_metrics(model.base, surface_rows, vocab, device),
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg150-real-synthetic-mix-v1", "variant": name, "real_fraction": real_fraction, "config": config, "vocabulary": vocab.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    syn_train, syn_dev, syn_holdout, real_train, real_holdout, vocab, source_stats = _prepare()
    base, config = pg148._load_body("large_transformer", vocab, device)
    surface_rows = [{"tokens": row["tokens"], "row_id": row["row_id"], "label_index": 0, "typed_available": False, "surface_kind": "pg146_real_surface"} for row in json.loads((RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json").read_text(encoding="utf-8"))["rows"]]
    fractions = [("synthetic_only", 0.0), ("real_5_percent", 0.05), ("real_25_percent", 0.25)]
    results = []
    for name, fraction in fractions:
        results.append(_train_variant(name, fraction, copy.deepcopy(base), config, _mix(real_train, syn_train, fraction), syn_dev, syn_holdout, real_holdout, surface_rows, vocab, device))
    report = {
        "protocol_id": "pg-pk-150-real-synthetic-mix-v1",
        "schema_version": "pg150-real-synthetic-mix-report-v1",
        "status": "completed_pg150_real_synthetic_mix",
        "device": str(device),
        "source": {**source_stats, "synthetic_train": len(syn_train), "real_action_train": len(real_train), "real_action_holdout": len(real_holdout)},
        "variants": results,
        "objective": "真实/合成监督比例对大模型动作迁移与真实 surface replay 的影响",
        "raw_payloads_in_model": False,
        "raw_responses_in_model": False,
        "capability_claim_allowed": False,
        "memory_promotion_allowed": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg150-real-synthetic-mix-dataset-v1", "source": report["source"], "mixes": [{"name": name, "real_fraction": fraction} for name, fraction in fractions], "raw_payloads": False, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-150-real-synthetic-mix-v1", "schema_version": "pg150-real-synthetic-mix-protocol-v1", "objective": report["objective"], "variants": [name for name, _ in fractions], "real_action_source": "pg136_causal_token_lm_dataset_v1.json", "real_surface_source": "pg146_public_lab_replay_model_dataset_v1.json", "promotion": {"capability_claim_allowed": False, "memory_promotion_allowed": False}}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "real_rows": row["real_rows_in_train"], "synthetic_holdout": row["synthetic_holdout"]["accuracy"], "real_pg136_holdout": row["real_pg136_holdout"]["accuracy"], "real_surface_lm_ppl": row["real_surface_lm"]["perplexity"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

