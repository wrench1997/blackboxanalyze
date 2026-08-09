"""PG-216: matched-capacity continuation on real Pikachu process traces.

This is a next-token LM experiment, not a vulnerability classifier.  It keeps
the PG-177 corpus fixed, appends only PG-215 real GET/POST Rule-IR tokens, and
continues matched 160M and 200M checkpoints with the same optimizer budget.
Evaluation is split by fresh seed and by a held-out route.  Checkpoints remain
diagnostic until an independent typed SQL oracle is available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG177 = _load("run_pg177_data_capacity_sweep.py")

RESEARCH = ROOT / "research"
PG215_DATASET = RESEARCH / "pg215_pikachu_real_trace_dataset_v1.json"
PG215_REPORT = RESEARCH / "pg215_pikachu_real_trace_dataset_report_v1.json"
START_DIR = ROOT / "artifacts" / "pg177-data-capacity-v1"
REPORT_PATH = RESEARCH / "pg216_real_trace_capacity_training_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg216_real_trace_capacity_training_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg216_real_trace_capacity_training_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg216_real_trace_capacity_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg216-real-trace-capacity-v1"

SEEDS = (17701, 17702)
VARIANTS = {
    "160m": {"reference": "160m"},
    "200m": {"reference": "200m"},
}
TRAIN_BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 5e-6
MAX_LEN = 128
BASE_TRAIN_ROWS_PER_SOURCE = 75
OLD_EVAL_MAX_ROWS = 300


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_data() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, str], dict[str, int]]:
    dataset = json.loads(PG215_DATASET.read_text(encoding="utf-8-sig"))
    pg215_train = [dict(row) for row in dataset["tokens"] if int(row["seed"]) in set(dataset["train_seeds"]) and str(row["route"]) != str(dataset["route_holdout"])]
    pg215_seed_holdout = [dict(row) for row in dataset["tokens"] if int(row["seed"]) in set(dataset["holdout_seeds"])]
    pg215_route_holdout = [dict(row) for row in dataset["tokens"] if str(row["route"]) == str(dataset["route_holdout"])]
    checkpoint = torch.load(START_DIR / "seed_17701_160m_scratch.pt", map_location="cpu", weights_only=False)
    base_train, base_eval, _, source_hashes = PG177._prepare_data(checkpoint)
    # Keep the capacity comparison runnable on the local 12 GB GPU while
    # preserving every PG-177 source.  This is a stratified diagnostic budget,
    # not a replacement for the full-corpus benchmark.
    stratified: list[dict[str, Any]] = []
    for index, source in enumerate(PG177.OLD_SOURCE_ORDER + tuple(f"pg177_{name}" for name in PG177.NEW_GENERATORS)):
        pool = [row for row in base_train if row.get("source") == source]
        if len(pool) <= BASE_TRAIN_ROWS_PER_SOURCE:
            stratified.extend(pool)
        else:
            stratified.extend(random.Random(21600 + index).sample(pool, BASE_TRAIN_ROWS_PER_SOURCE))
    base_train = stratified
    base_eval = {key: list(rows[:OLD_EVAL_MAX_ROWS]) for key, rows in base_eval.items()}
    # _prepare_data returns (train, eval, dataset, hashes); keep a stable copy
    # of the old evaluations and never put PG-215 holdout rows into training.
    return base_train, base_eval, {"pg215_seed_holdout": pg215_seed_holdout, "pg215_route_holdout": pg215_route_holdout}, {"pg215": dataset, "base": checkpoint}, source_hashes, {"pg215_train": len(pg215_train), "pg215_seed_holdout": len(pg215_seed_holdout), "pg215_route_holdout": len(pg215_route_holdout)}


def _train_one(variant: str, seed: int, base_train: list[dict[str, Any]], old_eval: dict[str, list[dict[str, Any]]], new_eval: dict[str, list[dict[str, Any]]], pg215_train: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    spec = VARIANTS[variant]
    checkpoint_path = START_DIR / f"seed_{seed}_{variant}_scratch.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = list(checkpoint["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocabulary)}
    missing = sorted({token for row in pg215_train for token in row["tokens"] if token not in stoi})
    if missing:
        raise RuntimeError(f"PG-216 vocabulary missing token: {missing[0]}")
    random.seed(seed + (0 if variant == "160m" else 1000))
    torch.manual_seed(seed + (0 if variant == "160m" else 1000))
    model = PG177._load_model(checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ordered = PG177._route(base_train) + list(pg215_train)
    loader = DataLoader(PG177._Dataset(ordered, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=PG177._collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    started = time.perf_counter()
    losses: list[float] = []
    for epoch in range(EPOCHS):
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
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            valid = int(targets.ne(0).sum().item())
            loss_sum += float(loss.detach().cpu()) * valid
            token_sum += valid
        losses.append(round(loss_sum / max(token_sum, 1), 8))
    old_metrics = {key: PG177._metrics(model, rows, stoi, device) for key, rows in old_eval.items()}
    new_metrics = {key: PG177._metrics(model, rows, stoi, device) for key, rows in new_eval.items()}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / f"seed_{seed}_{variant}.pt"
    torch.save({"schema_version": "pg216-real-trace-capacity-v1", "seed": seed, "variant": variant, "config": checkpoint["config"], "vocabulary": vocabulary, "model_state_dict": model.state_dict(), "raw_input_retained": False, "source_dataset": str(PG215_DATASET.relative_to(ROOT))}, artifact)
    result = {
        "seed": seed,
        "variant": variant,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_row_count": len(ordered),
        "pg215_train_row_count": len(pg215_train),
        "train_token_count": token_sum,
        "epoch_losses": losses,
        "old_holdout": old_metrics,
        "pg215_holdout": new_metrics,
        "artifact": str(artifact.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "optimizer_reset": True,
        "raw_input_retained": False,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    checkpoint = torch.load(START_DIR / "seed_17701_160m_scratch.pt", map_location="cpu", weights_only=False)
    base_train, old_eval, new_eval, metadata, source_hashes, data_counts = _load_data()
    pg215_dataset = metadata["pg215"]
    pg215_train = [dict(row) for row in pg215_dataset["tokens"] if int(row["seed"]) in set(pg215_dataset["train_seeds"]) and str(row["route"]) != str(pg215_dataset["route_holdout"])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            results.append(_train_one(variant, seed, base_train, old_eval, new_eval, pg215_train, device))
    grouped = {(row["seed"], row["variant"]): row for row in results}
    comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        small = grouped[(seed, "160m")]
        large = grouped[(seed, "200m")]
        small_new = small["pg215_holdout"]["pg215_seed_holdout"]["perplexity"]
        large_new = large["pg215_holdout"]["pg215_seed_holdout"]["perplexity"]
        small_route = small["pg215_holdout"]["pg215_route_holdout"]["perplexity"]
        large_route = large["pg215_holdout"]["pg215_route_holdout"]["perplexity"]
        comparisons.append({"seed": seed, "160m_seed_holdout": small_new, "200m_seed_holdout": large_new, "160m_route_holdout": small_route, "200m_route_holdout": large_route, "200m_better_seed_holdout": large_new < small_new, "200m_better_route_holdout": large_route < small_route})
    capacity_gain = bool(all(row["200m_better_seed_holdout"] and row["200m_better_route_holdout"] for row in comparisons))
    pg215_report = json.loads(PG215_REPORT.read_text(encoding="utf-8-sig"))
    report = {
        "protocol_id": "pg-pk-216-real-trace-capacity-training-v1",
        "schema_version": "pg216-real-trace-capacity-training-report-v1",
        "status": "completed_real_trace_capacity_sweep",
        "device": str(device),
        "data": {"base_train_rows": len(base_train), "base_train_rows_per_source_cap": BASE_TRAIN_ROWS_PER_SOURCE, "old_eval_rows_per_split_cap": OLD_EVAL_MAX_ROWS, "pg215_train_rows": len(pg215_train), "old_holdout_counts": {key: len(value) for key, value in old_eval.items()}, "pg215_holdout_counts": {key: len(value) for key, value in new_eval.items()}, "source_hashes": source_hashes, "pg215_dataset_sha256": pg215_dataset["dataset_sha256"], "pg215_report_sha256": _digest(pg215_report)},
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": 0.01, "batch_size": TRAIN_BATCH_SIZE, "epochs": EPOCHS, "reset_from_pg177_checkpoint": True},
        "variants": results,
        "capacity_comparison": comparisons,
        "capacity_200m_gain_repeated": capacity_gain,
        "promotion": {"training_eligible": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_generation_claim_allowed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False},
        "safety": {"loopback_only": True, "external_network_targets": False, "script_execution": False, "database_write": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "memory_promotion_allowed": False, "catastrophic_forgetting_gate": True},
    }
    report["report_sha256"] = _digest(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg216-real-trace-capacity-training-protocol-v1", "base_corpus": "PG-177 fixed corpus", "new_corpus": str(PG215_DATASET.relative_to(ROOT)), "capacity_variants": {variant: {str(seed): str((START_DIR / f"seed_{seed}_{variant}_scratch.pt").relative_to(ROOT)) for seed in SEEDS} for variant in VARIANTS}, "same_optimizer_budget": True, "seed_holdout": True, "route_holdout": "/vul/sqli/sqli_x.php", "raw_payload_and_response_excluded": True, "training_promotion_allowed": False, "vulnerability_claim_allowed": False}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(PROTOCOL_PATH, protocol)
    _write(TRACE_PATH, {"schema_version": "pg216-real-trace-capacity-training-trace-v1", "data": report["data"], "capacity_comparison": comparisons, "variants": [{"seed": row["seed"], "variant": row["variant"], "parameter_count": row["parameter_count"], "pg215_holdout": row["pg215_holdout"], "raw_input_retained": False} for row in results], "training_eligible": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    lines = ["# PG-216 real-trace capacity training", "", f"device={device}; base train={len(base_train)}; PG-215 train={len(pg215_train)}; variants={len(results)}", f"capacity comparison={comparisons}", f"200M better on both seed and route holdout across seeds={capacity_gain}", "", "该轮是 next-token Rule-IR 训练，不是漏洞分类器。所有 checkpoint 仍为诊断用途；只有跨 seed/route OOD 与 typed oracle 同时通过，才允许接入发包策略。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "device": str(device), "capacity_200m_gain_repeated": capacity_gain, "comparisons": comparisons, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
