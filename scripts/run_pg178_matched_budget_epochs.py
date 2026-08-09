"""PG-178: matched-token multi-epoch capacity continuation.

PG177 showed that a one-epoch 200M scratch model did not beat 160M.  This
experiment keeps the same 7,200-row routed corpus and extends both scratch
capacities to two and four total epochs, starting from their frozen PG177
one-epoch checkpoints.  The checkpoint artifacts contain only model weights;
the optimizer is intentionally re-created and recorded as a continuation
condition.
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg177_data_capacity_sweep import (  # noqa: E402
    EVAL_BATCH_SIZE,
    MAX_LEN,
    NEW_GENERATORS,
    START_CHECKPOINT,
    _Dataset,
    _collate,
    _load_model,
    _metrics,
    _prepare_data,
    _route,
)


RESEARCH = ROOT / "research"
PG177_REPORT = RESEARCH / "pg177_data_capacity_report_v1.json"
PG177_DATASET = RESEARCH / "pg177_data_capacity_dataset_v1.json"
PROTOCOL_PATH = RESEARCH / "pg178_matched_budget_epochs_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg178_matched_budget_epochs_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg178_matched_budget_epochs_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg178-matched-budget-epochs-v1"

SEEDS = (17701, 17702)
TARGET_EPOCHS = (2, 4)
TRAIN_BATCH_SIZE = 8
PER_SPLIT_TOLERANCE = 0.005
VARIANTS = {
    "160m": {"d_model": 1152, "nhead": 18, "layers": 10},
    "200m": {"d_model": 1280, "nhead": 20, "layers": 10},
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_pg177_checkpoint(seed: int, variant: str) -> dict[str, Any]:
    path = ROOT / "artifacts" / "pg177-data-capacity-v1" / f"seed_{seed}_{variant}_scratch.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": path, "checkpoint": torch.load(path, map_location="cpu")}


def _train_seed_variant(
    seed: int,
    variant: str,
    config: dict[str, int],
    train_rows: list[dict[str, Any]],
    eval_rows: dict[str, list[dict[str, Any]]],
    stoi: dict[str, int],
    device: torch.device,
) -> list[dict[str, Any]]:
    source = _load_pg177_checkpoint(seed, variant)
    checkpoint = source["checkpoint"]
    random.seed(seed)
    torch.manual_seed(seed)
    model = CausalTraceTransformer(len(stoi), d_model=config["d_model"], nhead=config["nhead"], layers=config["layers"], max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    ordered = _route(train_rows)
    loader = torch.utils.data.DataLoader(_Dataset(ordered, stoi), batch_size=TRAIN_BATCH_SIZE, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(2, 5):
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
        if epoch in TARGET_EPOCHS:
            metrics = {key: _metrics(model, rows, stoi, device) for key, rows in eval_rows.items()}
            old_keys = ("base_holdout", "typed_holdout", "pg168_ood", "pg170_ood", "pg172_ood", "pg176_ood")
            new_ood_keys = tuple(f"pg177_{generator}_ood" for generator in NEW_GENERATORS)
            result = {"seed": seed, "variant": variant, "epoch": epoch, "config": config, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "train_loss": round(loss_sum / max(token_sum, 1), 8), "train_token_count": token_sum, **metrics, "aggregate_existing": round(sum(metrics[key]["perplexity"] for key in old_keys) / len(old_keys), 8), "aggregate_new_ood": round(sum(metrics[key]["perplexity"] for key in new_ood_keys) / len(new_ood_keys), 8), "elapsed_seconds": round(time.perf_counter() - started, 3), "optimizer_reset_from_pg177": True}
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            path = ARTIFACT_DIR / f"seed_{seed}_{variant}_epoch{epoch}.pt"
            torch.save({"schema_version": "pg178-matched-budget-epochs-v1", "seed": seed, "variant": variant, "epoch": epoch, "config": config, "vocabulary": list(stoi), "model_state_dict": model.state_dict(), "optimizer_reset_from_pg177": True}, path)
            result["checkpoint"] = str(path.relative_to(ROOT))
            history.append(result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def main() -> None:
    pg177_report = json.loads(PG177_REPORT.read_text(encoding="utf-8"))
    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu")
    train_rows, eval_rows, dataset, source_hashes = _prepare_data(checkpoint)
    stoi = {token: index for index, token in enumerate(checkpoint["vocabulary"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for variant, config in VARIANTS.items():
            results.extend(_train_seed_variant(seed, variant, config, train_rows, eval_rows, stoi, device))
    per_epoch: list[dict[str, Any]] = []
    for epoch in TARGET_EPOCHS:
        rows = {(result["seed"], result["variant"]): result for result in results if result["epoch"] == epoch}
        comparisons = []
        for seed in SEEDS:
            small = rows[(seed, "160m")]
            large = rows[(seed, "200m")]
            comparisons.append({"seed": seed, "epoch": epoch, "160m_existing": small["aggregate_existing"], "200m_existing": large["aggregate_existing"], "160m_new_ood": small["aggregate_new_ood"], "200m_new_ood": large["aggregate_new_ood"], "200m_better_new_ood": large["aggregate_new_ood"] < small["aggregate_new_ood"], "200m_better_existing": large["aggregate_existing"] < small["aggregate_existing"]})
        per_epoch.extend(comparisons)
    best_by_epoch = {str(epoch): [item for item in per_epoch if item["epoch"] == epoch] for epoch in TARGET_EPOCHS}
    report = {"schema_version": "pg178-matched-budget-epochs-report-v1", "protocol_id": "pg-pk-178-matched-budget-epochs-v1", "status": "completed_pg178_matched_budget_epochs", "scope": {"claim": "matched token budget two/four epoch 160M versus 200M capacity continuation", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)}, "dataset": {"path": str(PG177_DATASET.relative_to(ROOT)), "train_row_count": len(train_rows), "full_eval_counts": {key: len(rows) for key, rows in eval_rows.items()}, "source_dataset_sha256": source_hashes}, "pg177_reference_report_sha256": _sha256_json(pg177_report), "variants": results, "capacity_comparison": best_by_epoch, "interpretation": {"same_train_rows": True, "same_route": True, "same_total_epoch_targets": list(TARGET_EPOCHS), "optimizer_reset_from_pg177": True, "capacity_gain_requires_both_seeds_and_new_ood": True}, "selection": {"selected_variant": None, "promotion_allowed": False, "vulnerability_claim_allowed": False}, "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "vulnerability_labels_in_model": False, "oracle_labels_in_model": False, "family_labels_in_model": False, "memory_promotion_allowed": False}, "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "dataset_sha256": dataset["dataset_sha256"]}}
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-178-matched-budget-epochs-v1", "schema_version": "pg178-matched-budget-epochs-protocol-v1", "pg177_dataset": str(PG177_DATASET.relative_to(ROOT)), "pg177_reference_artifacts": [f"artifacts/pg177-data-capacity-v1/seed_{seed}_{variant}_scratch.pt" for seed in SEEDS for variant in ("160m", "200m")], "seeds": list(SEEDS), "target_epochs": list(TARGET_EPOCHS), "capacity_configs": VARIANTS, "optimizer": {"name": "AdamW", "lr": 7e-5, "weight_decay": 0.01, "batch_size": TRAIN_BATCH_SIZE, "reset_at_pg177_boundary": True}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    lines = ["# PG-178 matched-budget epochs", "", f"- train rows: **{len(train_rows)}**", f"- target epochs: **{TARGET_EPOCHS}**", f"- capacity comparison: **{best_by_epoch}**", "", "该轮专门验证更大容量是否需要更多 token budget；optimizer 在 PG177 边界重建，因此不把它当作无条件连续训练。", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "capacity_comparison": best_by_epoch, "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
