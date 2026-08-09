"""PG-171: three-seed stability audit for PG-170 cross-generator replay."""

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

from run_pg168_discriminative_slot_augmentation import _Dataset, _collate, _load_model, _metrics  # noqa: E402


RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
GENERATOR_DATASET = RESEARCH / "pg170_cross_generator_dataset_v1.json"
PRIOR_DATASET = RESEARCH / "pg168_discriminative_slot_dataset_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg171-cross-generator-multiseed-v1"
PROTOCOL_PATH = RESEARCH / "pg171_cross_generator_multiseed_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg171_cross_generator_multiseed_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg171_cross_generator_multiseed_report_v1.md"
SEEDS = (17101, 17102, 17103)
SELECTION_SEED = 17001


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _train(seed: int, checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
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
    result = {"seed": seed, "train_row_count": len(rows), "train_loss": round(loss_sum / max(token_sum, 1), 8), "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device), "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device), "generator_dev": _metrics(model, eval_rows["generator_dev"], stoi, device), "generator_ood": _metrics(model, eval_rows["generator_ood"], stoi, device), "prior_slot_ood": _metrics(model, eval_rows["prior_slot_ood"], stoi, device), "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"seed_{seed}.pt"
    torch.save({"schema_version": "pg171-cross-generator-multiseed-v1", "seed": seed, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
    result["checkpoint"] = str(path.relative_to(ROOT))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(results: list[dict[str, Any]], split: str, metric: str) -> dict[str, float]:
    values = [float(item[split][metric]) for item in results]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": round(mean, 8), "std": round(math.sqrt(variance), 8), "min": round(min(values), 8), "max": round(max(values), 8)}


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    generator = json.loads(GENERATOR_DATASET.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_DATASET.read_text(encoding="utf-8"))
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    gen_train = [{"tokens": row["tokens"]} for row in generator["rows"] if row["split"] == "train"]
    gen_dev = [{"tokens": row["tokens"]} for row in generator["rows"] if row["split"] == "dev"]
    gen_ood = [{"tokens": row["tokens"]} for row in generator["rows"] if row["split"] == "ood"]
    prior_ood = [{"tokens": row["tokens"]} for row in prior["rows"] if row["split"] == "ood"]
    if {tuple(row["tokens"]) for row in gen_dev} & {tuple(row["tokens"]) for row in gen_ood}:
        raise RuntimeError("PG-171 generator dev/OOD projection collision")
    if {tuple(row["tokens"]) for row in gen_ood} & {tuple(row["tokens"]) for row in prior["rows"]}:
        raise RuntimeError("PG-171 cross-generator projection collision")
    base_all = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    selected_base = random.Random(SELECTION_SEED).sample(base_all, len(gen_train))
    train_base = selected_base + gen_train
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "generator_dev": gen_dev, "generator_ood": gen_ood, "prior_slot_ood": prior_ood}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {name: _metrics(baseline_model, rows, stoi, device) for name, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows = list(train_base)
        random.Random(seed).shuffle(rows)
        results.append(_train(seed, checkpoint, rows, eval_rows, stoi, device))
    aggregate = {f"{split}_{metric}": _aggregate(results, split, metric) for split in ("base_holdout", "typed_holdout", "generator_ood", "prior_slot_ood") for metric in ("perplexity", "next_token_accuracy")}
    stability = {"base_ppl_max": aggregate["base_holdout_perplexity"]["max"], "typed_ppl_max": aggregate["typed_holdout_perplexity"]["max"], "generator_ood_ppl_std": aggregate["generator_ood_perplexity"]["std"], "old_capability_thresholds_pass": aggregate["base_holdout_perplexity"]["max"] <= 2.5717967 * 1.02 and aggregate["typed_holdout_perplexity"]["max"] <= 1.45351808 * 1.10, "ood_stability_pass": aggregate["generator_ood_perplexity"]["std"] <= 0.10}
    report = {"schema_version": "pg171-cross-generator-multiseed-report-v1", "protocol_id": "pg-pk-171-cross-generator-multiseed-v1", "status": "completed_pg171_cross_generator_multiseed", "scope": {"claim": "three-seed cross-generator abstract Rule-IR stability diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"base_replay_count": len(selected_base), "generator_train_count": len(gen_train), "generator_dev_count": len(gen_dev), "generator_ood_count": len(gen_ood), "prior_slot_ood_count": len(prior_ood), "projection_overlap_dev_ood": 0, "projection_overlap_generator_prior": 0}, "baseline": baseline, "seed_results": results, "aggregate": aggregate, "stability": stability, "interpretation": {"cross_generator_ood_isolated": True, "vulnerability_claim_allowed": False, "promotion_allowed": False, "next_token_lm_only": True}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "generator_dataset_sha256": generator.get("dataset_sha256"), "prior_dataset_sha256": prior.get("dataset_sha256")}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-171-cross-generator-multiseed-v1", "schema_version": "pg171-cross-generator-multiseed-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "generator_dataset": str(GENERATOR_DATASET.relative_to(ROOT)), "prior_dataset": str(PRIOR_DATASET.relative_to(ROOT)), "seeds": list(SEEDS), "replay_ratio": "1:1", "model_parameter_count": 101380329, "stability": stability, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-171 cross-generator multi-seed", "", f"- seeds: **{', '.join(map(str, SEEDS))}**", f"- generator OOD PPL mean/std: **{aggregate['generator_ood_perplexity']['mean']} / {aggregate['generator_ood_perplexity']['std']}**", f"- base/typed PPL max: **{stability['base_ppl_max']} / {stability['typed_ppl_max']}**", f"- stability pass: **{stability['old_capability_thresholds_pass'] and stability['ood_stability_pass']}**", "", "该轮只复验跨生成器稳定性，不产生漏洞标签，也不晋级长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "stability": stability, "generator_ood": aggregate["generator_ood_perplexity"], "base": aggregate["base_holdout_perplexity"], "typed": aggregate["typed_holdout_perplexity"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
