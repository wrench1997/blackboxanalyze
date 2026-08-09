"""PG-153: long-context and replay-gated continual learning on MoE-Large.

The pretrained body is the PG-151 MoE-Large-4E checkpoint.  New mixed
GET/POST/action traces are packed either as individual 128-token examples or
as 256-token multi-step episodes.  A small old-corpus replay set is varied as
an ablation.  The original PG-147 holdout is never used for optimization and
is scored with the causal forgetting canary after each variant.
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
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pg148_large_model_posttraining as pg148  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402
from app.moe_trace_transformer import MoETraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg153-long-context-replay-v1"
REPORT = RESEARCH / "pg153_long_context_replay_report_v1.json"
DATASET = RESEARCH / "pg153_long_context_replay_dataset_v1.json"
PROTOCOL = RESEARCH / "pg153_long_context_replay_protocol_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg151-moe-capacity-v1" / "moe_large_4e.pt"
SEED = 15301
SHORT_LEN = 128
LONG_LEN = 256
EPOCHS = 1
REPLAY_COUNT = 1200


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class _TraceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, max_len: int) -> None:
        self.rows = rows
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"ids": self.vocabulary.encode(row["tokens"][: self.max_len]), "row": row}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    rows = []
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        rows.append(item["row"])
    return {"ids": ids, "mask": ids.ne(0), "rows": rows}


def _load_moe(vocabulary: pg148._Vocabulary, device: torch.device, max_len: int) -> MoETraceTransformer:
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    model = MoETraceTransformer(len(vocabulary.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), n_experts=int(config["n_experts"]), expert_ff=int(config["expert_ff"]), max_len=max_len).to(device)
    source_state = checkpoint["model_state_dict"]
    if max_len == SHORT_LEN:
        model.load_state_dict(source_state)
    else:
        target_state = model.state_dict()
        for key, value in source_state.items():
            if key == "position_embedding.weight":
                target_state[key][: value.shape[0]] = value
            else:
                target_state[key] = value
        model.load_state_dict(target_state)
    return model


def _pack(rows: list[dict[str, Any]], max_len: int, prefix: str) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current: list[str] = []
    component_ids: list[str] = []

    def flush() -> None:
        nonlocal current, component_ids
        if current:
            packed.append({"row_id": f"{prefix}-{len(packed):06d}", "tokens": list(current), "source": "packed_multi_step", "component_count": len(component_ids), "component_ids": list(component_ids)})
        current = []
        component_ids = []

    for row in rows:
        tokens = list(row["tokens"])
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if current and len(current) + len(tokens) > max_len:
            flush()
        current.extend(tokens)
        component_ids.append(str(row.get("row_id", "unknown")))
    flush()
    return packed


def _build_data(vocabulary: pg148._Vocabulary) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pg149 = _load_json(RESEARCH / "pg149_causal_action_alignment_dataset_v1.json")["splits"]
    real_train, _, real_holdout, _, source_stats = pg148._prepare_rows()
    desired_real = int(len(pg149["train"]) * 0.25 / 0.75)
    repeated_real = []
    for index in range(desired_real):
        row = dict(real_train[index % len(real_train)])
        row["row_id"] = f"pg153-real-repeat-{index:05d}-{row['row_id']}"
        repeated_real.append(row)
    new_train = list(pg149["train"]) + repeated_real
    random.Random(SEED).shuffle(new_train)
    new_holdout = list(pg149["holdout"]) + list(real_holdout)
    pg147 = _load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")["splits"]
    replay = []
    for index, original in enumerate(pg147["train"][:REPLAY_COUNT]):
        row = dict(original)
        row["row_id"] = f"pg153-replay-{index:05d}-{row.get('row_id', 'unknown')}"
        replay.append(row)
    data = {
        "short_train_replay": new_train + replay,
        "long_train_replay": _pack(new_train, LONG_LEN, "pg153-long-train") + replay,
        "long_train_no_replay": _pack(new_train, LONG_LEN, "pg153-long-train-no-replay"),
        "new_short_holdout": new_holdout,
        "new_long_holdout": _pack(new_holdout, LONG_LEN, "pg153-long-holdout"),
        "old_canary": list(pg147["holdout"]),
        "surface_eval": _load_json(RESEARCH / "pg146_public_lab_replay_model_dataset_v1.json")["rows"],
    }
    stats = {**source_stats, "synthetic_train_count": len(pg149["train"]), "synthetic_holdout_count": len(pg149["holdout"]), "real_repeated_count": len(repeated_real), "new_train_count": len(new_train), "replay_count": len(replay), "long_train_count": len(data["long_train_replay"]) - len(replay), "long_holdout_count": len(data["new_long_holdout"]), "old_canary_count": len(data["old_canary"]), "surface_eval_count": len(data["surface_eval"]), "real_fraction": 0.25}
    return data, stats


def _metrics(model: MoETraceTransformer, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device, max_len: int) -> dict[str, Any]:
    if not rows:
        return {"perplexity": 0.0, "next_token_accuracy": 0.0, "token_count": 0}
    effective_len = min(max_len, int(model.max_len))
    loader = DataLoader(_TraceDataset(rows, vocabulary, effective_len), batch_size=32 if effective_len == SHORT_LEN else 16, shuffle=False, collate_fn=_collate)
    model.eval()
    loss_sum = 0.0
    token_count = 0
    correct = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            hidden, _, _ = model.encode(ids[:, :-1], mask[:, :-1])
            logits = model.lm_head(hidden)
            targets = ids[:, 1:]
            valid = targets.ne(0)
            loss_sum += float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0, reduction="sum").item())
            token_count += int(valid.sum().item())
            correct += int(((logits.argmax(dim=-1) == targets) & valid).sum().item())
    mean_loss = loss_sum / max(token_count, 1)
    return {"perplexity": round(math.exp(min(mean_loss, 20.0)), 8), "next_token_accuracy": round(correct / max(token_count, 1), 8), "token_count": token_count}


def _train_variant(name: str, model: MoETraceTransformer, train_rows: list[dict[str, Any]], data: dict[str, list[dict[str, Any]]], vocabulary: pg148._Vocabulary, device: torch.device, max_len: int) -> dict[str, Any]:
    loader = DataLoader(_TraceDataset(train_rows, vocabulary, max_len), batch_size=32 if max_len == SHORT_LEN else 16, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    before = _metrics(model, data["new_short_holdout"], vocabulary, device, SHORT_LEN)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                hidden, auxiliary, _ = model.encode(ids[:, :-1], mask[:, :-1])
                logits = model.lm_head(hidden)
                targets = ids[:, 1:]
                ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                loss = ce + 0.01 * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "new_short_holdout": _metrics(model, data["new_short_holdout"], vocabulary, device, SHORT_LEN), "new_long_holdout": _metrics(model, data["new_long_holdout"], vocabulary, device, LONG_LEN)})
        print(json.dumps({"variant": name, "epoch": epoch, "new_short_ppl": history[-1]["new_short_holdout"]["perplexity"], "new_long_ppl": history[-1]["new_long_holdout"]["perplexity"]}, ensure_ascii=False), flush=True)
    # Recreate the pre-update body after the optimizer has released its state;
    # this keeps the canary comparison on the same device without doubling the
    # training peak memory.
    before_model = _load_moe(vocabulary, device, max_len)
    forgetting = compare_causal_lm_canary(before_model, model, data["old_canary"], vocabulary, device=device)
    del before_model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    result = {"variant": name, "context_length": max_len, "train_count": len(train_rows), "replay_rows": sum(1 for row in train_rows if str(row.get("row_id", "")).startswith("pg153-replay-")), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "before_new_short_holdout": before, "after_new_short_holdout": _metrics(model, data["new_short_holdout"], vocabulary, device, SHORT_LEN), "after_new_long_holdout": _metrics(model, data["new_long_holdout"], vocabulary, device, LONG_LEN), "old_canary_forgetting": forgetting, "surface_after": _metrics(model, data["surface_eval"], vocabulary, device, SHORT_LEN), "history": history, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg153-long-context-replay-v1", "variant": name, "context_length": max_len, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = pg148._Vocabulary(list(_load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")["vocabulary"]))
    data, stats = _build_data(vocabulary)
    variants = [
        ("short_context_replay", data["short_train_replay"], SHORT_LEN),
        ("long_context_replay", data["long_train_replay"], LONG_LEN),
        ("long_context_no_replay", data["long_train_no_replay"], LONG_LEN),
    ]
    results = []
    for name, rows, context_length in variants:
        model = _load_moe(vocabulary, device, context_length)
        print(json.dumps({"status": "starting_variant", "variant": name, "context_length": context_length, "train_count": len(rows), "device": str(device)}, ensure_ascii=False), flush=True)
        results.append(_train_variant(name, model, rows, data, vocabulary, device, context_length))
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {"protocol_id": "pg-pk-153-long-context-replay-v1", "schema_version": "pg153-long-context-replay-report-v1", "status": "completed_pg153_long_context_replay", "device": str(device), "seed": SEED, "source": stats, "variants": results, "objective": "source_heldout_long_context_and_continual_replay_forgetting", "data_policy": {"raw_payloads": False, "raw_responses": False, "external_network_targets": False, "old_holdout_used_for_optimization": False, "labels_in_lm_input": False}, "promotion": {"capability_claim_allowed": False, "training_artifact_promotion_allowed": False, "long_term_memory_promotion_allowed": False}, "report_sha256": ""}
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg153-long-context-replay-dataset-v1", "source": stats, "splits": {"short_train_count": len(data["short_train_replay"]), "long_train_replay_count": len(data["long_train_replay"]), "long_train_no_replay_count": len(data["long_train_no_replay"]), "new_short_holdout_count": len(data["new_short_holdout"]), "new_long_holdout_count": len(data["new_long_holdout"]), "old_canary_count": len(data["old_canary"]), "surface_eval_count": len(data["surface_eval"])}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-153-long-context-replay-v1", "schema_version": "pg153-long-context-replay-protocol-v1", "objective": report["objective"], "variants": [name for name, _, _ in variants], "context_lengths": {"short": SHORT_LEN, "long": LONG_LEN}, "replay_count": REPLAY_COUNT, "epochs": EPOCHS, "promotion": report["promotion"]}
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(PROTOCOL, protocol)
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "context_length": row["context_length"], "old_canary_forgetting": row["old_canary_forgetting"]["catastrophic_forgetting_detected"], "new_long_ppl": row["after_new_long_holdout"]["perplexity"], "surface_ppl": row["surface_after"]["perplexity"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
