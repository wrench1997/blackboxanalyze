"""PG-148: compare post-training strategies on Large/XL causal bodies."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402
from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg148-large-model-posttraining-v1"
REPORT = RESEARCH / "pg148_large_model_posttraining_report_v1.json"
PROTOCOL = RESEARCH / "pg148_large_model_posttraining_protocol_v1.json"
SEED = 14801
MAX_LEN = 128
ACTION_NAMES = (
    "replay_other_method",
    "repeat_matched_negative_pair",
    "probe_candidate_other_method",
    "abstain_candidate_only",
    "abstain_unknown_oracle",
    "stop_confirmed_positive",
    "abstain_budget_exhausted",
)


class _Vocabulary:
    def __init__(self, itos: list[str]) -> None:
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.stoi["[UNK]"]) for token in tokens]


class _Dataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], vocabulary: _Vocabulary) -> None:
        self.rows = rows
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"ids": self.vocabulary.encode(row["tokens"][:MAX_LEN]), "label": int(row.get("label_index", 0)), "row": row}


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["ids"]) for item in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    rows = []
    for index, item in enumerate(batch):
        ids[index, : len(item["ids"])] = torch.tensor(item["ids"], dtype=torch.long)
        rows.append(item["row"])
    return {"ids": ids, "mask": ids.ne(0), "labels": labels, "rows": rows}


class _ActionModel(nn.Module):
    def __init__(self, base: CausalTraceTransformer, d_model: int, *, adapter: bool = False) -> None:
        super().__init__()
        self.base = base
        self.adapter_enabled = bool(adapter)
        self.adapter = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, d_model)) if adapter else nn.Identity()
        self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(ACTION_NAMES)))

    def action_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.base.encode(ids, mask)
        if self.adapter_enabled:
            hidden = hidden + self.adapter(hidden)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return self.action_head(last)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], _Vocabulary, dict[str, Any]]:
    pg147 = _load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")
    vocabulary = _Vocabulary(list(pg147["vocabulary"]))
    source = _load_json(RESEARCH / "pg136_causal_token_lm_dataset_v1.json")
    original = _load_json(RESEARCH / "pg135_balanced_policy_dataset_v1.json")
    label_map = {str(row["row_id"]): str(row["label"]) for row in original["rows"]}
    meta_map = {str(row["row_id"]): row for row in original["rows"]}

    def convert(row: Mapping[str, Any], split: str) -> dict[str, Any]:
        label = str(row.get("action_label") or label_map.get(str(row["row_id"]), "abstain_unknown_oracle"))
        if label not in ACTION_NAMES:
            label = "abstain_unknown_oracle"
        original_row = meta_map.get(str(row["row_id"]), {})
        return {
            "row_id": str(row["row_id"]),
            "tokens": list(row["tokens"]),
            "label": label,
            "label_index": ACTION_NAMES.index(label),
            "split": split,
            "typed_available": bool((original_row.get("failure_signature") or {}).get("typed_available", True)),
            "surface_kind": str(original_row.get("surface_kind", "unknown")),
        }

    train = [convert(row, "train") for row in source["action_finetune_sequences"] if row.get("split") == "train"]
    dev = [convert(row, "dev") for row in source["action_finetune_sequences"] if row.get("split") == "dev"]
    holdout = [convert(row, "holdout") for row in source["holdout_sequences"]]
    return train, dev, holdout, vocabulary, {"label_count": len(label_map), "train_count": len(train), "dev_count": len(dev), "holdout_count": len(holdout)}


def _load_body(name: str, vocabulary: _Vocabulary, device: torch.device) -> tuple[CausalTraceTransformer, dict[str, Any]]:
    checkpoint = torch.load(ROOT / "artifacts" / "pg147-model-capacity-sweep-v1" / f"{name}.pt", map_location=device, weights_only=False)
    config = dict(checkpoint["config"])
    model = CausalTraceTransformer(len(vocabulary.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def _new_body(name: str, vocabulary: _Vocabulary, device: torch.device) -> tuple[CausalTraceTransformer, dict[str, Any]]:
    checkpoint = _load_json if False else None
    config = {"d_model": 512, "nhead": 8, "layers": 6} if name == "large" else {"d_model": 768, "nhead": 12, "layers": 8}
    return CausalTraceTransformer(len(vocabulary.itos), **config, max_len=MAX_LEN).to(device), config


def _metrics(model: _ActionModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = 0
    count = 0
    unknown_count = 0
    unknown_abstain = 0
    false_stop = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            prediction = model.action_logits(ids, mask).argmax(dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()
            for pred, label, row in zip(prediction, labels, batch["rows"]):
                correct += int(pred == label)
                count += 1
                if not row["typed_available"]:
                    unknown_count += 1
                    unknown_abstain += int(ACTION_NAMES[pred] == "abstain_unknown_oracle")
                if ACTION_NAMES[pred] == "stop_confirmed_positive" and ACTION_NAMES[label] != "stop_confirmed_positive":
                    false_stop += 1
    return {
        "count": count,
        "accuracy": round(correct / max(count, 1), 8),
        "unknown_count": unknown_count,
        "unknown_abstain_rate": round(unknown_abstain / max(unknown_count, 1), 8) if unknown_count else 1.0,
        "false_stop_count": false_stop,
    }


def _train(name: str, mode: str, body: CausalTraceTransformer, config: dict[str, Any], train: list[dict[str, Any]], dev: list[dict[str, Any]], holdout: list[dict[str, Any]], vocabulary: _Vocabulary, device: torch.device, lm_rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = copy.deepcopy(body).to(device) if mode != "scratch" else None
    if mode == "scratch":
        body, config = _new_body("large", vocabulary, device)
    action_model = _ActionModel(body, int(config["d_model"]), adapter=mode == "adapter").to(device)
    if mode in {"frozen", "adapter"}:
        for parameter in action_model.base.parameters():
            parameter.requires_grad_(False)
    if mode == "low_lr":
        optimizer = torch.optim.AdamW(
            [{"params": action_model.base.parameters(), "lr": 1e-5}, {"params": action_model.action_head.parameters(), "lr": 3e-4}],
            weight_decay=0.01,
        )
    else:
        optimizer = torch.optim.AdamW([parameter for parameter in action_model.parameters() if parameter.requires_grad], lr=2e-4, weight_decay=0.01)
    train_loader = DataLoader(_Dataset(train, vocabulary), batch_size=32, shuffle=True, collate_fn=_collate)
    dev_loader = DataLoader(_Dataset(dev, vocabulary), batch_size=64, shuffle=False, collate_fn=_collate)
    holdout_loader = DataLoader(_Dataset(holdout, vocabulary), batch_size=64, shuffle=False, collate_fn=_collate)
    lm_loader = DataLoader(_Dataset(lm_rows, vocabulary), batch_size=32, shuffle=True, collate_fn=_collate)
    lm_iter = iter(lm_loader)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    epochs = 20 if mode in {"frozen", "adapter"} else 6
    for epoch in range(1, epochs + 1):
        action_model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            action_loss = nn.functional.cross_entropy(action_model.action_logits(ids, mask), labels)
            loss = action_loss
            if mode == "joint":
                try:
                    lm_batch = next(lm_iter)
                except StopIteration:
                    lm_iter = iter(lm_loader)
                    lm_batch = next(lm_iter)
                lm_ids = lm_batch["ids"].to(device)
                lm_mask = lm_batch["mask"].to(device)
                lm_logits = action_model.base(lm_ids[:, :-1], lm_mask[:, :-1])
                lm_targets = lm_ids[:, 1:]
                lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1), ignore_index=0)
                loss = action_loss + 0.10 * lm_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(epoch_loss / max(len(train_loader), 1), 8), "dev": _metrics(action_model, dev_loader, device)})
    result = {
        "variant": name,
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in action_model.parameters()),
        "train": _metrics(action_model, train_loader, device),
        "dev": _metrics(action_model, dev_loader, device),
        "holdout": _metrics(action_model, holdout_loader, device),
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    if anchor is not None:
        canary = [{"tokens": row["tokens"][:MAX_LEN]} for row in dev[:64] + holdout[:64]]
        result["language_forgetting"] = compare_causal_lm_canary(anchor, action_model.base, canary, vocabulary, device=device)
    else:
        result["language_forgetting"] = {"not_applicable": True}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg148-large-model-action-v1", "variant": name, "mode": mode, "config": config, "vocabulary": vocabulary.itos, "model_state_dict": action_model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, dev, holdout, vocabulary, source_stats = _prepare_rows()
    lm_dataset = _load_json(RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json")
    lm_rows = [row for row in lm_dataset["splits"]["train"][:4096] if isinstance(row.get("tokens"), list)]
    large_body, large_config = _load_body("large_transformer", vocabulary, device)
    xl_body, xl_config = _load_body("xl_transformer", vocabulary, device)
    variants = [
        ("scratch_large", "scratch", large_body, large_config),
        ("frozen_large", "frozen", copy.deepcopy(large_body), large_config),
        ("low_lr_large", "low_lr", copy.deepcopy(large_body), large_config),
        ("adapter_large", "adapter", copy.deepcopy(large_body), large_config),
        ("joint_xl", "joint", xl_body, xl_config),
    ]
    results = [_train(name, mode, body, config, train, dev, holdout, vocabulary, device, lm_rows) for name, mode, body, config in variants]
    report = {
        "protocol_id": "pg-pk-148-large-model-posttraining-v1",
        "schema_version": "pg148-large-model-posttraining-report-v1",
        "status": "completed_pg148_large_model_posttraining",
        "device": str(device),
        "source": source_stats,
        "action_set": list(ACTION_NAMES),
        "lm_replay_rows": len(lm_rows),
        "variants": results,
        "raw_payloads_in_model": False,
        "raw_responses_in_model": False,
        "capability_claim_allowed": False,
        "memory_promotion_allowed": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    protocol = {
        "protocol_id": "pg-pk-148-large-model-posttraining-v1",
        "schema_version": "pg148-large-model-posttraining-protocol-v1",
        "objective": "比较大容量 causal body 在 scratch/frozen/low-lr/joint LM+action 后训练中的动作迁移与语言遗忘。",
        "variants": ["scratch_large", "frozen_large", "low_lr_large", "joint_xl"],
        "language_canary": {"enabled": True, "max_relative_perplexity_increase": 0.20, "max_next_token_accuracy_drop": 0.05, "max_mean_logit_kl": 0.10},
        "promotion": {"capability_claim_allowed": False, "memory_promotion_allowed": False},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "variants": [{"variant": row["variant"], "mode": row["mode"], "parameter_count": row["parameter_count"], "dev_accuracy": row["dev"]["accuracy"], "holdout_accuracy": row["holdout"]["accuracy"], "forgetting": row["language_forgetting"].get("catastrophic_forgetting_detected", False)} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
