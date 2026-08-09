"""PG-152: compare dense and MoE bodies on the same real/synthetic mix.

PG-150 showed that real action rows, rather than more procedural rows alone,
recover performance on a real holdout.  This round keeps that 25% real mix
fixed and changes only the pretrained body: a dense Large Transformer versus
the 38M-parameter MoE-Large-4E checkpoint from PG-151.  The 165M XL result is
already covered by PG-151; using the smaller sparse body here isolates the
architecture variable without turning the action+LM replay into a paging
benchmark on a 12GB GPU.  Both receive the same action head and 0.5 LM anchor.
"""

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
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg148_large_model_posttraining as pg148  # noqa: E402
from app.moe_trace_transformer import MoETraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg152-real-mix-architecture-v1"
REPORT = RESEARCH / "pg152_real_mix_architecture_report_v1.json"
DATASET = RESEARCH / "pg152_real_mix_architecture_dataset_v1.json"
PROTOCOL = RESEARCH / "pg152_real_mix_architecture_protocol_v1.json"
SEED = 15201
MAX_LEN = 128
LM_ANCHOR_WEIGHT = 0.5
EPOCHS = 2


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _MoEActionModel(nn.Module):
    def __init__(self, base: MoETraceTransformer, d_model: int) -> None:
        super().__init__()
        self.base = base
        self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(pg148.ACTION_NAMES)))

    def action_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.base.encode(ids, mask)
        hidden = encoded[0] if isinstance(encoded, tuple) else encoded
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return self.action_head(last)


def _lm_forward(base: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(base, MoETraceTransformer):
        hidden, auxiliary, _ = base.encode(ids, mask)
        return base.lm_head(hidden), auxiliary
    return base(ids, mask), torch.zeros((), device=ids.device)


def _lm_metrics(base: nn.Module, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, float]:
    if not rows:
        return {"perplexity": 0.0, "next_token_accuracy": 0.0, "token_count": 0}
    encoded = [vocabulary.encode(row["tokens"][:MAX_LEN]) for row in rows]
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, item in enumerate(encoded):
        ids[index, : len(item)] = torch.tensor(item, dtype=torch.long, device=device)
    base.eval()
    with torch.inference_mode():
        logits, _ = _lm_forward(base, ids[:, :-1], ids[:, :-1].ne(0))
    targets = ids[:, 1:]
    valid = targets.ne(0)
    count = int(valid.sum().item())
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    accuracy = float(((logits.argmax(dim=-1) == targets) & valid).sum().item() / max(count, 1))
    return {"perplexity": round(float(torch.exp(loss).item()), 8), "next_token_accuracy": round(accuracy, 8), "token_count": count}


def _load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pg148._Vocabulary, dict[str, Any]]:
    pg149 = json.loads((RESEARCH / "pg149_causal_action_alignment_dataset_v1.json").read_text(encoding="utf-8"))
    synthetic = pg149["splits"]
    real_train, real_dev, real_holdout, vocabulary, source_stats = pg148._prepare_rows()
    return synthetic["train"], synthetic["dev"], synthetic["holdout"], real_train, real_holdout, vocabulary, {**source_stats, "synthetic_train_count": len(synthetic["train"]), "synthetic_dev_count": len(synthetic["dev"]), "synthetic_holdout_count": len(synthetic["holdout"]), "real_dev_count": len(real_dev)}


def _mix(real_train: list[dict[str, Any]], synthetic_train: list[dict[str, Any]], fraction: float = 0.25) -> list[dict[str, Any]]:
    desired_real = max(1, int(len(synthetic_train) * fraction / (1.0 - fraction)))
    repeated = [copy.deepcopy(real_train[index % len(real_train)]) for index in range(desired_real)]
    for index, row in enumerate(repeated):
        row["row_id"] = f"pg152-real-repeat-{index:05d}-{row['row_id']}"
    mixed = list(synthetic_train) + repeated
    random.Random(SEED).shuffle(mixed)
    return mixed


def _build_dense(vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(ROOT / "artifacts" / "pg147-model-capacity-sweep-v1" / "large_transformer.pt", map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    body = pg148.CausalTraceTransformer(len(vocabulary.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    body.load_state_dict(checkpoint["model_state_dict"])
    return pg148._ActionModel(body, int(config["d_model"])).to(device), {"body": "dense_large", **config, "source_checkpoint": "artifacts/pg147-model-capacity-sweep-v1/large_transformer.pt"}


def _build_moe(vocabulary: pg148._Vocabulary, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(ROOT / "artifacts" / "pg151-moe-capacity-v1" / "moe_large_4e.pt", map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    body = MoETraceTransformer(len(vocabulary.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), n_experts=int(config["n_experts"]), expert_ff=int(config["expert_ff"]), max_len=MAX_LEN).to(device)
    body.load_state_dict(checkpoint["model_state_dict"])
    return _MoEActionModel(body, int(config["d_model"])).to(device), {"body": "moe_large_4e", **config, "source_checkpoint": "artifacts/pg151-moe-capacity-v1/moe_large_4e.pt"}


def _train_variant(name: str, model: nn.Module, config: dict[str, Any], train_rows: list[dict[str, Any]], syn_dev: list[dict[str, Any]], syn_holdout: list[dict[str, Any]], real_holdout: list[dict[str, Any]], language_holdout: list[dict[str, Any]], surface_rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(pg148._Dataset(train_rows, vocabulary), batch_size=32 if config["body"].startswith("moe") else 64, shuffle=True, collate_fn=pg148._collate)
    syn_dev_loader = DataLoader(pg148._Dataset(syn_dev, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    syn_hold_loader = DataLoader(pg148._Dataset(syn_holdout, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    real_hold_loader = DataLoader(pg148._Dataset(real_holdout, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4 if config["body"].startswith("moe") else 1.5e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    before_lm = _lm_metrics(model.base, language_holdout, vocabulary, device)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_loss = nn.functional.cross_entropy(model.action_logits(ids, mask), labels)
                lm_logits, aux = _lm_forward(model.base, ids[:, :-1], mask[:, :-1])
                lm_targets = ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1), ignore_index=0)
                loss = action_loss + LM_ANCHOR_WEIGHT * lm_loss + 0.01 * aux
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "synthetic_dev": pg148._metrics(model, syn_dev_loader, device), "real_holdout": pg148._metrics(model, real_hold_loader, device)})
        print(json.dumps({"variant": name, "epoch": epoch, "synthetic_dev_accuracy": history[-1]["synthetic_dev"]["accuracy"], "real_holdout_accuracy": history[-1]["real_holdout"]["accuracy"]}, ensure_ascii=False), flush=True)
    result = {"variant": name, "body": config["body"], "real_fraction": 0.25, "train_count": len(train_rows), "real_rows_in_train": sum(1 for row in train_rows if str(row["row_id"]).startswith("pg152-real-repeat-")), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "synthetic_dev": pg148._metrics(model, syn_dev_loader, device), "synthetic_holdout": pg148._metrics(model, syn_hold_loader, device), "real_pg136_holdout": pg148._metrics(model, real_hold_loader, device), "language_before": before_lm, "language_after": _lm_metrics(model.base, language_holdout, vocabulary, device), "real_surface_lm": _lm_metrics(model.base, surface_rows, vocabulary, device), "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    if isinstance(model.base, MoETraceTransformer):
        result["routing"] = {"expert_count": model.base.n_experts, "holdout_routing": _routing_snapshot(model.base, syn_holdout, vocabulary, device)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg152-real-mix-architecture-v1", "variant": name, "config": config, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def _routing_snapshot(base: MoETraceTransformer, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    encoded = [vocabulary.encode(row["tokens"][:MAX_LEN]) for row in rows]
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, item in enumerate(encoded):
        ids[index, : len(item)] = torch.tensor(item, dtype=torch.long, device=device)
    with torch.inference_mode():
        _, _, loads = base.encode(ids, ids.ne(0))
    # encode() returns [layers, experts]; average the layers while retaining
    # one routing probability per expert.
    mean_load = loads.detach().cpu().mean(dim=0)
    probs = mean_load / mean_load.sum().clamp_min(1e-8)
    entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())
    return {"expert_load": [round(float(value), 8) for value in probs.tolist()], "routing_entropy": round(entropy, 8)}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    syn_train, syn_dev, syn_holdout, real_train, real_holdout, vocabulary, source_stats = _load_data()
    mixed = _mix(real_train, syn_train, 0.25)
    language_holdout = [{"tokens": row["tokens"], "row_id": row["row_id"]} for row in json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))["splits"]["holdout"]]
    surface_rows = [{"tokens": row["tokens"], "row_id": row["row_id"]} for row in json.loads((RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json").read_text(encoding="utf-8"))["rows"]]
    builders = [("dense_large_real25", _build_dense), ("moe_large4_real25", _build_moe)]
    results = []
    for name, builder in builders:
        model, config = builder(vocabulary, device)
        print(json.dumps({"status": "starting_variant", "variant": name, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train_variant(name, model, config, mixed, syn_dev, syn_holdout, real_holdout, language_holdout, surface_rows, vocabulary, device))
    report = {"protocol_id": "pg-pk-152-real-mix-architecture-v1", "schema_version": "pg152-real-mix-architecture-report-v1", "status": "completed_pg152_real_mix_architecture", "device": str(device), "seed": SEED, "source": {**source_stats, "mixed_train_count": len(mixed), "real_rows_in_mixed_train": sum(1 for row in mixed if str(row["row_id"]).startswith("pg152-real-repeat-")), "language_holdout_count": len(language_holdout), "surface_eval_count": len(surface_rows)}, "variants": results, "objective": "fixed_25_percent_real_action_mix_dense_vs_moe_with_lm_replay_anchor", "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "raw_payloads_in_model": False, "raw_responses_in_model": False, "external_network_targets": False, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg152-real-mix-architecture-dataset-v1", "source": report["source"], "mix": {"real_fraction": 0.25, "synthetic_source": "pg149_causal_action_alignment_dataset_v1.json", "real_source": "pg136_causal_token_lm_dataset_v1.json", "holdout_untouched": True}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-152-real-mix-architecture-v1", "schema_version": "pg152-real-mix-architecture-protocol-v1", "objective": report["objective"], "variants": [name for name, _ in builders], "common_training": {"real_fraction": 0.25, "lm_anchor_weight": LM_ANCHOR_WEIGHT, "epochs": EPOCHS, "holdout_untouched": True}, "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "parameter_count": row["parameter_count"], "synthetic_holdout_accuracy": row["synthetic_holdout"]["accuracy"], "real_holdout_accuracy": row["real_pg136_holdout"]["accuracy"], "language_before_ppl": row["language_before"]["perplexity"], "language_after_ppl": row["language_after"]["perplexity"], "real_surface_ppl": row["real_surface_lm"]["perplexity"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
