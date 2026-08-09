"""Train/evaluate the quarantined PG-56 causal abstract-trace baseline."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


DATASET_PATH = ROOT / "research" / "pg56_causal_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg56_causal_trace_pretraining_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg56_causal_trace_pretraining_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg56-causal-trace-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
SEED = 20260803
MAX_LEN = 128
EPOCHS = 60


def _vocabulary(rows: list[dict[str, Any]]) -> dict[str, int]:
    tokens = sorted({token for row in rows for token in row["tokens"]})
    ordered = ["<PAD>", "<UNK>"] + [token for token in tokens if token not in {"<PAD>", "<UNK>"}]
    return {token: index for index, token in enumerate(ordered)}


def _encode(row: dict[str, Any], vocabulary: dict[str, int]) -> list[int]:
    unknown = vocabulary["<UNK>"]
    return [vocabulary.get(token, unknown) for token in row["tokens"][:MAX_LEN]]


def _batch(rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [_encode(row, vocabulary) for row in rows]
    width = max(len(item) for item in encoded)
    tokens = torch.full((len(encoded), width), vocabulary["<PAD>"], dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for index, item in enumerate(encoded):
        tokens[index, :len(item)] = torch.tensor(item, dtype=torch.long)
        mask[index, :len(item)] = True
    return tokens.to(device), mask.to(device)


def _loss_and_accuracy(model: CausalTraceTransformer, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary["<PAD>"], reduction="sum")
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            tokens, mask = _batch(rows[start:start + 32], vocabulary, device)
            if tokens.shape[1] < 2:
                continue
            logits = model(tokens[:, :-1], mask[:, :-1])
            target = tokens[:, 1:]
            target_mask = mask[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            total_loss += float(loss.cpu())
            total_tokens += int(target_mask.sum().cpu())
            predicted = logits.argmax(dim=-1)
            correct += int(((predicted == target) & target_mask).sum().cpu())
    return total_loss / max(total_tokens, 1), correct / max(total_tokens, 1)


def _target_metrics(model: CausalTraceTransformer, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    model.eval()
    markers = {
        # (marker, target offset, causal-logit offset).  A causal logit at
        # position i predicts the token at i+1, so a target two slots after a
        # marker must be read from the logit one slot after that marker.
        "next_action": ("NEXT_ACTION_TARGET", 1, 0),
        "oracle_modality": ("ORACLE_TARGET", 1, 0),
        "oracle_outcome": ("ORACLE_TARGET", 2, 1),
        "rule_effect": ("RULE_IR_TARGET", 1, 0),
    }
    stats = {name: {"correct": 0, "count": 0} for name in markers}
    unknown_family_abstain = 0
    with torch.inference_mode():
        for row in rows:
            token_ids = torch.tensor([_encode(row, vocabulary)], dtype=torch.long, device=device)
            if token_ids.shape[1] < 2:
                continue
            logits = model(token_ids[:, :-1], torch.ones_like(token_ids[:, :-1], dtype=torch.bool))
            ids = row["tokens"][:MAX_LEN]
            for name, (marker, offset, logit_offset) in markers.items():
                try:
                    marker_index = ids.index(marker)
                    target_index = marker_index + offset
                    logit_index = marker_index + logit_offset
                    if target_index >= len(ids) or logit_index >= logits.shape[1]:
                        continue
                    expected = vocabulary.get(ids[target_index], vocabulary["<UNK>"])
                    predicted = int(logits[0, logit_index].argmax().cpu())
                    stats[name]["count"] += 1
                    stats[name]["correct"] += int(predicted == expected)
                except ValueError:
                    continue
            # This baseline never emits family names by construction.
            if row["target"].get("unknown_family"):
                unknown_family_abstain += 1
    for value in stats.values():
        value["accuracy"] = round(value["correct"] / max(value["count"], 1), 6)
    return {
        **stats,
        "unknown_family_naming_attempts": 0,
        "unknown_family_strict_abstain": unknown_family_abstain == sum(row["target"].get("unknown_family", False) for row in rows),
    }


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    rows = dataset["rows"]
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    vocabulary = _vocabulary(train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTraceTransformer(len(vocabulary), max_len=MAX_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary["<PAD>"])
    best_dev_loss = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        total_loss = 0.0
        total_batches = 0
        for start in range(0, len(order), 32):
            batch_rows = [train_rows[index] for index in order[start:start + 32]]
            tokens, mask = _batch(batch_rows, vocabulary, device)
            if tokens.shape[1] < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1], mask[:, :-1])
            target = tokens[:, 1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        dev_loss, dev_accuracy = _loss_and_accuracy(model, dev_rows, vocabulary, device)
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(total_batches, 1), 6), "dev_loss": round(dev_loss, 6), "dev_token_accuracy": round(dev_accuracy, 6)})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    train_loss, train_accuracy = _loss_and_accuracy(model, train_rows, vocabulary, device)
    dev_loss, dev_accuracy = _loss_and_accuracy(model, dev_rows, vocabulary, device)
    holdout_loss, holdout_accuracy = _loss_and_accuracy(model, holdout_rows, vocabulary, device)
    holdout_targets = _target_metrics(model, holdout_rows, vocabulary, device)
    dev_targets = _target_metrics(model, dev_rows, vocabulary, device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "pg56-causal-trace-transformer-checkpoint-v1",
        "model_state": best_state or {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "vocabulary": vocabulary,
        "max_len": MAX_LEN,
        "seed": SEED,
        "device_at_training": str(device),
        "family_name_in_input": False,
        "typed_oracle_before_target_marker": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    report = {
        "protocol_id": "pg-pk-56-causal-trace-pretraining-v1",
        "schema_version": "pg56-causal-trace-pretraining-report-v1",
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "device": str(device),
        "vocabulary_size": len(vocabulary),
        "split_counts": {"train": len(train_rows), "dev": len(dev_rows), "holdout": len(holdout_rows)},
        "metrics": {
            "train": {"loss": round(train_loss, 6), "token_accuracy": round(train_accuracy, 6)},
            "dev": {"loss": round(dev_loss, 6), "token_accuracy": round(dev_accuracy, 6), "targets": dev_targets},
            "holdout": {"loss": round(holdout_loss, 6), "token_accuracy": round(holdout_accuracy, 6), "targets": holdout_targets},
        },
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "history_tail": history[-5:],
        "input_contract": dataset["model_input_contract"],
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "formal_family_capability_claim_allowed": False,
        "status": "pretraining_baseline_complete_family_capability_unproven",
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "\n".join([
            "# PG-56 因果 Trace Transformer 预训练基线",
            "",
            f"设备：`{device}`；train/dev/holdout：`{len(train_rows)}/{len(dev_rows)}/{len(holdout_rows)}`；词表：`{len(vocabulary)}`。",
            f"盲测 token loss/accuracy：`{holdout_loss:.4f}` / `{holdout_accuracy:.4f}`。",
            f"盲测 next-action accuracy：`{holdout_targets['next_action']['accuracy']:.4f}`；oracle modality：`{holdout_targets['oracle_modality']['accuracy']:.4f}`；outcome：`{holdout_targets['oracle_outcome']['accuracy']:.4f}`。",
            "该结果只证明抽象轨迹预测基线已可复现；尚未证明未见漏洞族 Rule IR 泛化，也不会晋升训练集或长期记忆。",
            "",
        ]) + "",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
