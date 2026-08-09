"""PG-166: replay-anchored adaptation of the 101M XXL checkpoint.

The only new observations are PG-165 bounded surface-effect projections.  Two
post-training strategies are compared:

* ``replay_anchored`` keeps the full PG-163 corpus in the minibatch stream;
* ``surface_only`` trains on the 28 new projection rows alone.

Both strategies use the same frozen vocabulary and no attestation status,
family name, raw probe or response body as model input.  The experiment is a
direct catastrophic-forgetting diagnostic, not a vulnerability detector.
"""

from __future__ import annotations

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
SURFACE_DATASET = RESEARCH / "pg165_surface_attested_training_dataset_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg166-surface-replay-adaptation-v1"
PROTOCOL_PATH = RESEARCH / "pg166_surface_replay_adaptation_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg166_surface_replay_adaptation_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg166_surface_replay_adaptation_report_v1.md"
MAX_LEN = 128
SEED = 16601


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
        ids = [self.stoi.get(token, self.stoi["[UNK]"]) for token in row["tokens"][:MAX_LEN]]
        return {"ids": ids}


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
    return {"ids": ids, "mask": ids.ne(0)}


def _metrics(model: nn.Module, rows: list[dict[str, Any]], stoi: dict[str, int], device: torch.device, batch_size: int = 8) -> dict[str, float | int]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=batch_size, shuffle=False, collate_fn=_collate)
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
    mean = total_loss / max(total_tokens, 1)
    return {"loss": round(mean, 8), "perplexity": round(math.exp(min(mean, 20.0)), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> CausalTraceTransformer:
    config = checkpoint["config"]
    model = CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _adapt(name: str, model: CausalTraceTransformer, rows: list[dict[str, Any]], all_eval: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device, checkpoint: dict[str, Any], *, lr: float, epochs: int) -> dict[str, Any]:
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
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
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, all_eval["base_holdout"], stoi, device), "typed_holdout": _metrics(model, all_eval["typed_holdout"], stoi, device)})
    result = {"strategy": name, "train_row_count": len(rows), "lr": lr, "epochs": epochs, "base_holdout": _metrics(model, all_eval["base_holdout"], stoi, device), "typed_holdout": _metrics(model, all_eval["typed_holdout"], stoi, device), "surface_rows": _metrics(model, all_eval["surface"], stoi, device), "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg166-surface-replay-adaptation-v1", "strategy": name, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    return result


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    surface = json.loads(SURFACE_DATASET.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    vocabulary = list(checkpoint["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocabulary)}
    # Normalize attestation-only control tokens into the already-frozen
    # unknown-oracle token; attestation status remains an external label.
    surface_rows: list[dict[str, Any]] = []
    for row in surface["rows"]:
        tokens = ["obs.oracle=unknown_oracle" if token == "obs.oracle=safe_surface_attestation" else token for token in row["model_input_tokens"]]
        surface_rows.append({"row_id": row["row_id"], "tokens": tokens})
    base_train = list(base["train_rows"])
    replay_rows = base_train + surface_rows
    eval_rows = {"base_holdout": list(base["base_holdout_rows"]), "typed_holdout": list(base["typed_holdout_rows"]), "surface": surface_rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, value, stoi, device) for key, value in eval_rows.items()}
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    variants = {
        "replay_anchored": (replay_rows, 2e-5, 1),
        "surface_only": (surface_rows, 1e-4, 1),
    }
    results: dict[str, Any] = {}
    for name, (rows, lr, epochs) in variants.items():
        model = _load_model(checkpoint, device)
        results[name] = _adapt(name, model, rows, eval_rows, stoi, device, checkpoint, lr=lr, epochs=epochs)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": "pg166-surface-replay-adaptation-report-v1",
        "protocol_id": "pg-pk-166-surface-replay-adaptation-v1",
        "status": "completed_pg166_surface_replay_adaptation",
        "scope": {"claim": "replay/forgetting training diagnostic for abstract Rule-IR tokens", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)},
        "dataset": {"base_train_count": len(base_train), "surface_row_count": len(surface_rows), "replay_train_count": len(replay_rows), "base_holdout_count": len(eval_rows["base_holdout"]), "typed_holdout_count": len(eval_rows["typed_holdout"])},
        "baseline": baseline,
        "variants": results,
        "interpretation": {"replay_anchor_expected_to_preserve_base": True, "surface_only_expected_to_forget_base": True, "vulnerability_claim_allowed": False, "promotion_allowed": False},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "attestation_labels_in_model": False, "memory_promotion_allowed": False},
        "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "dataset_sha256": base.get("dataset_sha256"), "surface_dataset_sha256": surface.get("dataset_sha256")},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-166-surface-replay-adaptation-v1", "schema_version": "pg166-surface-replay-adaptation-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "surface_dataset": str(SURFACE_DATASET.relative_to(ROOT)), "strategies": {"replay_anchored": {"lr": 2e-5, "epochs": 1, "replay_rows": len(replay_rows)}, "surface_only": {"lr": 1e-4, "epochs": 1, "rows": len(surface_rows)}}, "metrics": ["base_holdout_perplexity", "typed_holdout_perplexity", "surface_effect_projection_perplexity", "next_token_accuracy"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join([
        "# PG-166 surface replay adaptation",
        "",
        f"- baseline base/typed PPL: **{baseline['base_holdout']['perplexity']} / {baseline['typed_holdout']['perplexity']}**",
        f"- replay-anchored base/typed PPL: **{results['replay_anchored']['base_holdout']['perplexity']} / {results['replay_anchored']['typed_holdout']['perplexity']}**",
        f"- surface-only base/typed PPL: **{results['surface_only']['base_holdout']['perplexity']} / {results['surface_only']['typed_holdout']['perplexity']}**",
        "",
        "该轮只比较 replay 与遗忘；表面 attestation 不作为漏洞标签，checkpoint 不晋级长期记忆。",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "baseline": {key: value["perplexity"] for key, value in baseline.items()}, "variants": {key: {"base_ppl": value["base_holdout"]["perplexity"], "typed_ppl": value["typed_holdout"]["perplexity"], "surface_ppl": value["surface_rows"]["perplexity"]} for key, value in results.items()}, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
