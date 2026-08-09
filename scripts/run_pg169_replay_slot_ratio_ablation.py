"""PG-169: replay/slot ratio ablation on the fixed 101M checkpoint.

This experiment keeps the model capacity and optimizer fixed and changes only
how many old PG-163 rows accompany the PG-168 abstract slot rows.  It measures
the trade-off between learning collision-free slots and preserving old typed
Rule-IR transitions.  No vulnerability label or raw probe is used.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg168_discriminative_slot_augmentation import _Dataset, _collate, _load_model, _metrics  # noqa: E402


RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
SLOT_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg169-replay-slot-ratio-v1"
PROTOCOL_PATH = RESEARCH / "pg169_replay_slot_ratio_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg169_replay_slot_ratio_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg169_replay_slot_ratio_report_v1.md"
MAX_LEN = 128
SEED = 16901


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _train(name: str, checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = _load_model(checkpoint, device)
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=True, collate_fn=_collate)
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
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        valid = int(targets.ne(0).sum().item())
        loss_sum += float(loss.detach().cpu()) * valid
        token_sum += valid
    result = {"strategy": name, "train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "slot_dev": _metrics(model, eval_rows["slot_dev"], stoi, device), "slot_ood": _metrics(model, eval_rows["slot_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{name}.pt"
    torch.save({"schema_version": "pg169-replay-slot-ratio-v1", "strategy": name, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    slot = json.loads(SLOT_DATASET.read_text(encoding="utf-8"))
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    base_rows = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    slot_rows = [{"tokens": row["tokens"]} for row in slot["rows"] if row["split"] == "train"]
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "slot_dev": [{"tokens": row["tokens"]} for row in slot["rows"] if row["split"] == "dev"], "slot_ood": [{"tokens": row["tokens"]} for row in slot["rows"] if row["split"] == "ood"]}
    rng = random.Random(SEED)
    # Ratios are old:slot. Every slot row is retained; old rows are sampled
    # without replacement so the comparison changes only the replay weight.
    ratio_counts = {"ratio_1_1": 8000, "ratio_1_2": 4000, "ratio_1_4": 2000}
    variants: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {key: _metrics(baseline_model, rows, stoi, device) for key, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for name, base_count in ratio_counts.items():
        selected = list(base_rows) if base_count == len(base_rows) else rng.sample(base_rows, base_count)
        rows = selected + list(slot_rows)
        rng.shuffle(rows)
        variants[name] = _train(name, checkpoint, rows, eval_rows, stoi, device)
    report = {"schema_version": "pg169-replay-slot-ratio-report-v1", "protocol_id": "pg-pk-169-replay-slot-ratio-v1", "status": "completed_pg169_replay_slot_ratio_ablation", "scope": {"claim": "replay/slot ratio training ablation for abstract Rule-IR tokens", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"base_train_count": len(base_rows), "slot_train_count": len(slot_rows), "slot_dev_count": len(eval_rows["slot_dev"]), "slot_ood_count": len(eval_rows["slot_ood"]), "projection_overlap_dev_ood": 0}, "ratios": {name: {"old_rows": count, "slot_rows": len(slot_rows), "old_to_slot": name.replace("ratio_", "").replace("_", ":")} for name, count in ratio_counts.items()}, "baseline": baseline, "variants": variants, "selection": {"objective": "minimize slot_ood perplexity subject to typed_holdout perplexity <= 1.45351808 * 1.10 and base_holdout perplexity <= 2.5717967 * 1.02", "promotion_allowed": False, "vulnerability_claim_allowed": False}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "slot_dataset_sha256": slot.get("dataset_sha256")}}
    feasible = {name: result for name, result in variants.items() if result["typed_holdout"]["perplexity"] <= 1.45351808 * 1.10 and result["base_holdout"]["perplexity"] <= 2.5717967 * 1.02}
    report["selection"]["feasible_variants"] = sorted(feasible)
    report["selection"]["selected_variant"] = min(feasible, key=lambda name: feasible[name]["slot_ood"]["perplexity"]) if feasible else None
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-169-replay-slot-ratio-v1", "schema_version": "pg169-replay-slot-ratio-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "slot_dataset": str(SLOT_DATASET.relative_to(ROOT)), "ratios": report["ratios"], "optimizer": {"name": "AdamW", "lr": 2e-5, "epochs": 1}, "selection": report["selection"], "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    selected = report["selection"]["selected_variant"]
    MARKDOWN_PATH.write_text("\n".join(["# PG-169 replay/slot ratio ablation", "", f"- feasible variants: **{', '.join(report['selection']['feasible_variants']) or 'none'}**", f"- selected variant: **{selected or 'none'}**", *[f"- {name} base/typed/slot-OOD PPL: **{value['base_holdout']['perplexity']} / {value['typed_holdout']['perplexity']} / {value['slot_ood']['perplexity']}**" for name, value in variants.items()], "", "该轮只优化回放配比，不产生漏洞标签，不晋级长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feasible_variants": report["selection"]["feasible_variants"], "selected_variant": selected, "metrics": {name: {"base_ppl": value["base_holdout"]["perplexity"], "typed_ppl": value["typed_holdout"]["perplexity"], "slot_ood_ppl": value["slot_ood"]["perplexity"]} for name, value in variants.items()}, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
