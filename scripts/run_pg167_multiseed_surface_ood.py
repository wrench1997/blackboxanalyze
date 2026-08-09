"""PG-167: multi-seed replay adaptation with unseen surface-family holdout.

Five surface families are available to adaptation and two complete families
are held out.  The same 101M checkpoint and frozen vocabulary are trained from
three seeds.  This is a stability/generalization experiment for abstract
Rule-IR tokens; it is not a vulnerability scanner and does not promote memory.
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
CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
BASE_DATASET = RESEARCH / "pg163_large_typed_mix_dataset_v1.json"
SURFACE_DATASET = RESEARCH / "pg165_surface_attested_training_dataset_v1.json"
ARTIFACT_DIR = ROOT / "artifacts" / "pg167-multiseed-surface-ood-v1"
PROTOCOL_PATH = RESEARCH / "pg167_multiseed_surface_ood_protocol_v1.json"
REPORT_PATH = RESEARCH / "pg167_multiseed_surface_ood_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg167_multiseed_surface_ood_report_v1.md"
MAX_LEN = 128
SEEDS = (16701, 16702, 16703)
TRAIN_FAMILIES = {
    "sqli_boolean",
    "sqli_timing",
    "sqli_search",
    "url_redirect",
    "xss_dom_source",
}
HOLDOUT_FAMILIES = {"sqli_string", "xss_reflected_get"}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family(row_id: str) -> str:
    parts = row_id.split("-")
    return "-".join(parts[1:-2])


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
    loader = DataLoader(_Dataset(rows, stoi), batch_size=8, shuffle=False, collate_fn=_collate)
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


def _load_model(checkpoint: dict[str, Any], device: torch.device) -> CausalTraceTransformer:
    config = checkpoint["config"]
    model = CausalTraceTransformer(len(checkpoint["vocabulary"]), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _train_one(seed: int, checkpoint: dict[str, Any], rows: list[dict[str, Any]], eval_rows: dict[str, list[dict[str, Any]]], stoi: dict[str, int], device: torch.device) -> dict[str, Any]:
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
    result = {
        "seed": seed,
        "train_loss": round(loss_sum / max(token_sum, 1), 8),
        "train_row_count": len(rows),
        "base_holdout": _metrics(model, eval_rows["base_holdout"], stoi, device),
        "typed_holdout": _metrics(model, eval_rows["typed_holdout"], stoi, device),
        "seen_surface": _metrics(model, eval_rows["seen_surface"], stoi, device),
        "unseen_surface": _metrics(model, eval_rows["unseen_surface"], stoi, device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"replay_seed_{seed}.pt"
    torch.save({"schema_version": "pg167-multiseed-surface-ood-v1", "seed": seed, "config": checkpoint["config"], "vocabulary": checkpoint["vocabulary"], "model_state_dict": model.state_dict()}, path)
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
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8"))
    surface = json.loads(SURFACE_DATASET.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    vocabulary = list(checkpoint["vocabulary"])
    stoi = {token: index for index, token in enumerate(vocabulary)}
    surface_rows: list[dict[str, Any]] = []
    for source in surface["rows"]:
        tokens = ["obs.oracle=unknown_oracle" if token == "obs.oracle=safe_surface_attestation" else token for token in source["model_input_tokens"]]
        surface_rows.append({"row_id": source["row_id"], "family": _family(source["row_id"]), "tokens": tokens})
    actual_families = {row["family"] for row in surface_rows}
    if actual_families != TRAIN_FAMILIES | HOLDOUT_FAMILIES:
        raise RuntimeError(f"unexpected surface families: {sorted(actual_families)}")
    seen_surface = [{"tokens": row["tokens"], "row_id": row["row_id"]} for row in surface_rows if row["family"] in TRAIN_FAMILIES]
    unseen_surface = [{"tokens": row["tokens"], "row_id": row["row_id"]} for row in surface_rows if row["family"] in HOLDOUT_FAMILIES]
    seen_signatures = {tuple(row["tokens"]) for row in seen_surface}
    unseen_signatures = {tuple(row["tokens"]) for row in unseen_surface}
    projection_overlap_count = len(seen_signatures & unseen_signatures)
    unseen_novel_projection_count = len(unseen_signatures - seen_signatures)
    base_train = [{"tokens": row["tokens"]} for row in base["train_rows"]]
    train_rows = base_train + seen_surface
    eval_rows = {"base_holdout": base["base_holdout_rows"], "typed_holdout": base["typed_holdout_rows"], "seen_surface": seen_surface, "unseen_surface": unseen_surface}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = _load_model(checkpoint, device)
    baseline = {name: _metrics(baseline_model, rows, stoi, device) for name, rows in eval_rows.items()}
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = [_train_one(seed, checkpoint, train_rows, eval_rows, stoi, device) for seed in SEEDS]
    report = {
        "schema_version": "pg167-multiseed-surface-ood-report-v1",
        "protocol_id": "pg-pk-167-multiseed-surface-ood-v1",
        "status": "completed_pg167_multiseed_surface_ood",
        "scope": {"claim": "multi-seed replay adaptation and unseen surface-family generalization diagnostic", "real_vulnerability_scanner_claim_allowed": False, "device": str(device)},
        "split": {"train_families": sorted(TRAIN_FAMILIES), "unseen_holdout_families": sorted(HOLDOUT_FAMILIES), "seen_surface_row_count": len(seen_surface), "unseen_surface_row_count": len(unseen_surface), "seen_unique_projection_count": len(seen_signatures), "unseen_unique_projection_count": len(unseen_signatures), "projection_overlap_count": projection_overlap_count, "unseen_novel_projection_count": unseen_novel_projection_count, "projection_ood_informative": projection_overlap_count == 0},
        "dataset": {"base_train_count": len(base_train), "adaptation_train_count": len(train_rows), "base_holdout_count": len(eval_rows["base_holdout"]), "typed_holdout_count": len(eval_rows["typed_holdout"])},
        "baseline": baseline,
        "seed_results": results,
        "aggregate": {f"{split}_{metric}": _aggregate(results, split, metric) for split in ("base_holdout", "typed_holdout", "seen_surface", "unseen_surface") for metric in ("perplexity", "next_token_accuracy")},
        "interpretation": {"replay_anchor_used": True, "vulnerability_claim_allowed": False, "promotion_allowed": False, "unseen_family_isolation": True, "projection_ood_informative": projection_overlap_count == 0, "diagnosis": "projection_collision_requires_more_diverse_ir_features_before_claiming_family_ood" if projection_overlap_count else "family_ood_projection_isolation_passed"},
        "safety": {"loopback_only": True, "external_network": False, "script_execution": False, "database_write": False, "credential_access": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "family_labels_in_model": False, "attestation_labels_in_model": False, "memory_promotion_allowed": False},
        "source": {"runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "base_dataset_sha256": base.get("dataset_sha256"), "surface_dataset_sha256": surface.get("dataset_sha256")},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    protocol = {"protocol_id": "pg-pk-167-multiseed-surface-ood-v1", "schema_version": "pg167-multiseed-surface-ood-protocol-v1", "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "surface_dataset": str(SURFACE_DATASET.relative_to(ROOT)), "seeds": list(SEEDS), "train_families": sorted(TRAIN_FAMILIES), "unseen_holdout_families": sorted(HOLDOUT_FAMILIES), "optimizer": {"name": "AdamW", "lr": 2e-5, "epochs": 1}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}, "safety": report["safety"]}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    MARKDOWN_PATH.write_text("\n".join(["# PG-167 multi-seed surface OOD", "", f"- train families: **{', '.join(sorted(TRAIN_FAMILIES))}**", f"- unseen families: **{', '.join(sorted(HOLDOUT_FAMILIES))}**", f"- baseline unseen PPL: **{baseline['unseen_surface']['perplexity']}**", f"- replay seed unseen PPL mean/std: **{report['aggregate']['unseen_surface_perplexity']['mean']} / {report['aggregate']['unseen_surface_perplexity']['std']}**", f"- projection overlap between train/holdout: **{projection_overlap_count}**", "", "由于投影碰撞，本轮不宣称族外泛化；先增加能区分表面族的 Rule-IR 特征。该轮不产生漏洞标签，也不晋级 checkpoint 或长期记忆。", ""]) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "projection_overlap_count": projection_overlap_count, "baseline_unseen_ppl": baseline["unseen_surface"]["perplexity"], "seed_unseen_ppl": [item["unseen_surface"]["perplexity"] for item in results], "aggregate": report["aggregate"], "report": str(REPORT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
