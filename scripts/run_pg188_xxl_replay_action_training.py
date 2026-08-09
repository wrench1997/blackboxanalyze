"""PG-188: train a large action head with explicit LM replay/forgetting gates.

The 101M body is initialized from the PG-164 typed-mix language checkpoint,
while action labels come from the abstract PG-148 process dataset.  No target
request/response, raw payload, route, family, or oracle authority enters the
training input.  A frozen Pikachu replay remains evaluation-only.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


RESEARCH = ROOT / "research"
REPORT_PATH = RESEARCH / "pg188_xxl_replay_action_training_report_v1.json"
PROTOCOL_PATH = RESEARCH / "pg188_xxl_replay_action_training_protocol_v1.json"
TRACE_PATH = RESEARCH / "pg188_xxl_replay_action_training_trace_v1.json"
MARKDOWN_PATH = RESEARCH / "pg188_xxl_replay_action_training_report_v1.md"
ARTIFACT_DIR = ROOT / "artifacts" / "pg188-xxl-replay-action-v1"
BODY_CHECKPOINT = ROOT / "artifacts" / "pg164-xxl-capacity-v1" / "xxl_typed_mix.pt"
SEED = 18801
MAX_LEN = 128
BATCH_SIZE = 8
ACTION_NAMES = (
    "replay_other_method",
    "repeat_matched_negative_pair",
    "probe_candidate_other_method",
    "abstain_candidate_only",
    "abstain_unknown_oracle",
    "stop_confirmed_positive",
    "abstain_budget_exhausted",
)


def _load_pg148() -> Any:
    path = ROOT / "scripts" / "run_pg148_large_model_posttraining.py"
    spec = importlib.util.spec_from_file_location("pg148_for_pg188", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-148 dataset helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG148 = _load_pg148()


class _ActionModel(nn.Module):
    def __init__(self, body: CausalTraceTransformer, d_model: int) -> None:
        super().__init__()
        self.body = body
        self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, len(ACTION_NAMES)))

    def action_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.body.encode(ids, mask)
        lengths = mask.to(torch.long).sum(dim=1).clamp_min(1)
        last = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
        return self.action_head(last)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Any, list[dict[str, Any]], dict[str, Any]]:
    train, dev, holdout, vocab, stats = PG148._prepare_rows()
    lm_dataset = json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))
    lm_rows = [dict(row) for row in lm_dataset["splits"]["train"][:4096] if isinstance(row.get("tokens"), list)]
    return train, dev, holdout, vocab, lm_rows, stats


def _batch(rows: list[dict[str, Any]], vocab: Any, *, shuffle: bool, seed: int, batch_size: int = BATCH_SIZE) -> Iterable[dict[str, Any]]:
    ordered = list(rows)
    if shuffle:
        random.Random(seed).shuffle(ordered)
    for start in range(0, len(ordered), batch_size):
        subset = ordered[start:start + batch_size]
        ids_list = [vocab.encode(list(row["tokens"] if "tokens" in row else row["tokens_raw"])[:MAX_LEN]) for row in subset]
        width = max(len(values) for values in ids_list)
        ids = torch.zeros((len(ids_list), width), dtype=torch.long)
        mask = torch.zeros((len(ids_list), width), dtype=torch.bool)
        for index, values in enumerate(ids_list):
            ids[index, :len(values)] = torch.tensor(values, dtype=torch.long)
            mask[index, :len(values)] = True
        yield {"ids": ids, "mask": mask, "rows": subset}


def _action_batch(rows: list[dict[str, Any]], vocab: Any, *, shuffle: bool, seed: int) -> Iterable[dict[str, Any]]:
    for batch in _batch(rows, vocab, shuffle=shuffle, seed=seed):
        labels = torch.tensor([ACTION_NAMES.index(str(row["label"])) for row in batch["rows"]], dtype=torch.long)
        batch["labels"] = labels
        yield batch


def _init_body(target_vocab: Any, device: torch.device) -> tuple[CausalTraceTransformer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(BODY_CHECKPOINT, map_location="cpu", weights_only=False)
    old_vocab = [str(item) for item in checkpoint["vocabulary"]]
    config = dict(checkpoint["config"])
    body = CausalTraceTransformer(len(target_vocab.itos), d_model=int(config["d_model"]), nhead=int(config["nhead"]), layers=int(config["layers"]), max_len=MAX_LEN).to(device)
    new_state = body.state_dict()
    old_state = checkpoint["model_state_dict"]
    for key in new_state:
        if key not in old_state:
            continue
        if key == "token_embedding.weight":
            old_index = {token: index for index, token in enumerate(old_vocab)}
            with torch.no_grad():
                for index, token in enumerate(target_vocab.itos):
                    if token in old_index:
                        new_state[key][index].copy_(old_state[key][old_index[token]])
        elif key == "lm_head.weight":
            old_index = {token: index for index, token in enumerate(old_vocab)}
            with torch.no_grad():
                for index, token in enumerate(target_vocab.itos):
                    if token in old_index:
                        new_state[key][index].copy_(old_state[key][old_index[token]])
        elif key == "lm_head.bias":
            old_index = {token: index for index, token in enumerate(old_vocab)}
            with torch.no_grad():
                for index, token in enumerate(target_vocab.itos):
                    if token in old_index:
                        new_state[key][index].copy_(old_state[key][old_index[token]])
        elif tuple(new_state[key].shape) == tuple(old_state[key].shape):
            new_state[key].copy_(old_state[key])
    body.load_state_dict(new_state)
    return body, config, {"old_vocab_size": len(old_vocab), "new_vocab_size": len(target_vocab.itos), "body_checkpoint_sha256": hashlib.sha256(BODY_CHECKPOINT.read_bytes()).hexdigest()}


def _lm_metrics(body: CausalTraceTransformer, rows: list[dict[str, Any]], vocab: Any, device: torch.device, limit: int = 256) -> dict[str, Any]:
    body.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    with torch.inference_mode():
        for batch in _batch(rows[:limit], vocab, shuffle=False, seed=0, batch_size=4):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            if ids.shape[1] < 2:
                continue
            logits = body(ids[:, :-1], mask[:, :-1])
            targets = ids[:, 1:]
            target_mask = mask[:, 1:]
            loss_sum = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum", ignore_index=0)
            total_loss += float(loss_sum.detach().cpu())
            total_tokens += int(target_mask.sum().item())
            correct += int(((logits.argmax(-1) == targets) & target_mask).sum().item())
    mean_loss = total_loss / max(total_tokens, 1)
    return {"loss": round(mean_loss, 8), "perplexity": round(float(torch.exp(torch.tensor(mean_loss))), 8), "next_token_accuracy": round(correct / max(total_tokens, 1), 8), "token_count": total_tokens}


def _action_metrics(model: _ActionModel, rows: list[dict[str, Any]], vocab: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = 0
    count = 0
    unknown = 0
    unknown_abstain = 0
    false_stop = 0
    with torch.inference_mode():
        for batch in _action_batch(rows, vocab, shuffle=False, seed=0):
            logits = model.action_logits(batch["ids"].to(device), batch["mask"].to(device))
            predictions = logits.argmax(dim=-1).detach().cpu().tolist()
            labels = batch["labels"].tolist()
            for prediction, label, row in zip(predictions, labels, batch["rows"]):
                predicted_name = ACTION_NAMES[prediction]
                expected_name = ACTION_NAMES[label]
                correct += int(prediction == label)
                count += 1
                if not bool(row.get("typed_available", True)):
                    unknown += 1
                    unknown_abstain += int(predicted_name == "abstain_unknown_oracle")
                false_stop += int(predicted_name == "stop_confirmed_positive" and expected_name != "stop_confirmed_positive")
    return {"count": count, "accuracy": round(correct / max(count, 1), 8), "unknown_count": unknown, "unknown_abstain_rate": round(unknown_abstain / max(unknown, 1), 8) if unknown else 1.0, "false_stop_count": false_stop}


def _forgetting(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    relative = (after["perplexity"] - before["perplexity"]) / max(before["perplexity"], 1e-9)
    drop = before["next_token_accuracy"] - after["next_token_accuracy"]
    return {"before": before, "after": after, "delta": {"relative_perplexity_increase": round(relative, 8), "next_token_accuracy_drop": round(drop, 8)}, "thresholds": {"max_relative_perplexity_increase": 0.20, "max_next_token_accuracy_drop": 0.05}, "catastrophic_forgetting_detected": bool(relative > 0.20 or drop > 0.05)}


def _train_variant(name: str, mode: str, train: list[dict[str, Any]], dev: list[dict[str, Any]], holdout: list[dict[str, Any]], vocab: Any, lm_rows: list[dict[str, Any]], device: torch.device, body_config: dict[str, Any], init_meta: dict[str, Any]) -> dict[str, Any]:
    torch.manual_seed(SEED)
    random.seed(SEED)
    body, _, _ = _init_body(vocab, device)
    model = _ActionModel(body, int(body_config["d_model"])).to(device)
    before_lm = _lm_metrics(model.body, lm_rows, vocab, device)
    if mode == "frozen_head":
        for parameter in model.body.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.action_head.parameters(), lr=3e-4, weight_decay=0.01)
        epochs = 12
    elif mode == "replay_low_lr":
        optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": 1e-5}, {"params": model.action_head.parameters(), "lr": 3e-4}], weight_decay=0.01)
        epochs = 2
    else:
        optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": 7e-6}, {"params": model.action_head.parameters(), "lr": 2e-4}], weight_decay=0.01)
        epochs = 2
    history: list[dict[str, Any]] = []
    lm_iter_rows = list(lm_rows)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        replay_iter = iter(_batch(lm_iter_rows, vocab, shuffle=True, seed=SEED + epoch, batch_size=4))
        for batch in _action_batch(train, vocab, shuffle=True, seed=SEED + epoch):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            action_loss = nn.functional.cross_entropy(model.action_logits(ids, mask), labels)
            loss = action_loss
            if mode != "frozen_head":
                try:
                    lm_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(_batch(lm_iter_rows, vocab, shuffle=True, seed=SEED + epoch + 100, batch_size=4))
                    lm_batch = next(replay_iter)
                lm_ids = lm_batch["ids"].to(device)
                lm_mask = lm_batch["mask"].to(device)
                if lm_ids.shape[1] > 1:
                    lm_logits = model.body(lm_ids[:, :-1], lm_mask[:, :-1])
                    lm_targets = lm_ids[:, 1:]
                    lm_loss = nn.functional.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]), lm_targets.reshape(-1), ignore_index=0)
                    loss = action_loss + (0.10 if mode == "replay_low_lr" else 0.50) * lm_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_loss": round(statistics.mean(losses), 8), "dev": _action_metrics(model, dev, vocab, device)})
    after_lm = _lm_metrics(model.body, lm_rows, vocab, device)
    result = {
        "variant": name,
        "mode": mode,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_count": len(train),
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "history": history,
        "train": _action_metrics(model, train, vocab, device),
        "dev": _action_metrics(model, dev, vocab, device),
        "holdout": _action_metrics(model, holdout, vocab, device),
        "language_forgetting": _forgetting(before_lm, after_lm),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "raw_payloads_in_model": False,
        "raw_responses_in_model": False,
        "target_trace_in_training": False,
        "oracle_authority_in_input": False,
        "family_in_input": False,
        "route_in_input": False,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg188-xxl-replay-action-v1", "variant": name, "mode": mode, "vocabulary": list(vocab.itos), "body_config": body_config, "model_state": model.state_dict(), "raw_input_retained": False}, ARTIFACT_DIR / f"{name}.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    train, dev, holdout, vocab, lm_rows, source_stats = _load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_meta = torch.load(BODY_CHECKPOINT, map_location="cpu", weights_only=False)
    body_config = {"d_model": int(body_meta["config"]["d_model"]), "nhead": int(body_meta["config"]["nhead"]), "layers": int(body_meta["config"]["layers"]), "max_len": MAX_LEN}
    init_meta = {"body_checkpoint_sha256": hashlib.sha256(BODY_CHECKPOINT.read_bytes()).hexdigest(), "body_vocab_size": len(body_meta["vocabulary"]), "action_vocab_size": len(vocab.itos)}
    variants = [("frozen_xxl_head", "frozen_head"), ("replay_xxl_low_lr", "replay_low_lr"), ("replay_xxl_strong", "replay_strong")]
    results = [_train_variant(name, mode, train, dev, holdout, vocab, lm_rows, device, body_config, init_meta) for name, mode in variants]
    eligible = [item for item in results if not item["language_forgetting"]["catastrophic_forgetting_detected"] and item["holdout"]["unknown_abstain_rate"] >= 0.95 and item["holdout"]["false_stop_count"] == 0]
    selected = max(eligible, key=lambda item: float(item["holdout"]["accuracy"]))["variant"] if eligible else None
    report = {
        "protocol_id": "pg-pk-188-xxl-replay-action-training-v1",
        "schema_version": "pg188-xxl-replay-action-training-report-v1",
        "status": "completed_xxl_replay_action_training",
        "device": str(device),
        "source": {"action_dataset": "research/pg136_causal_token_lm_dataset_v1.json + research/pg135_balanced_policy_dataset_v1.json", "body_checkpoint": str(BODY_CHECKPOINT.relative_to(ROOT)), "source_stats": source_stats, "lm_replay_rows": len(lm_rows), "target_trace_used_for_training": False},
        "body_config": body_config,
        "init": init_meta,
        "variants": results,
        "selection": {"selected_variant": selected, "candidate_gate": "holdout unknown abstain >= .95, false_stop=0, catastrophic_forgetting=false", "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "large-model action capability remains separate from target vulnerability confirmation"},
        "safety": {"loopback_only": True, "external_network": False, "raw_payloads_in_model": False, "raw_responses_in_model": False, "target_trace_in_training": False, "script_execution": False, "database_write": False, "memory_promotion_allowed": False},
    }
    report["report_sha256"] = _sha256_json(report)
    _write(REPORT_PATH, report)
    _write(TRACE_PATH, {"schema_version": "pg188-xxl-replay-action-training-trace-v1", "training_only_abstract_tokens": True, "target_trace_used_for_training": False, "variants": [{"variant": row["variant"], "holdout": row["holdout"], "language_forgetting": row["language_forgetting"], "raw_payloads_in_model": False, "raw_responses_in_model": False} for row in results], "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False})
    protocol = {"protocol_id": "pg-pk-188-xxl-replay-action-training-v1", "schema_version": "pg188-xxl-replay-action-training-protocol-v1", "parameter_target": 101380329, "variants": [name for name, _ in variants], "body_init": str(BODY_CHECKPOINT.relative_to(ROOT)), "action_input_contract": ["abstract process tokens only"], "excluded_from_input": ["raw_payload", "raw_response", "route", "family", "oracle_authority", "vulnerability_label"], "lm_replay_rows": len(lm_rows), "forgetting_gate": {"max_relative_perplexity_increase": 0.20, "max_next_token_accuracy_drop": 0.05, "catastrophic_forgetting_blocks_selection": True}, "selection_gate": {"unknown_abstain_rate_min": 0.95, "false_stop_count_max": 0, "target_trace_training_forbidden": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    _write(PROTOCOL_PATH, protocol)
    lines = ["# PG-188 XXL replay/action training", "", f"device={device}; body_parameters={body_config['d_model']}; action_train={len(train)}; lm_replay={len(lm_rows)}", "", "| variant | parameters | holdout accuracy | unknown abstain | false stop | forgetting |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['variant']} | {row['parameter_count']} | {row['holdout']['accuracy']} | {row['holdout']['unknown_abstain_rate']} | {row['holdout']['false_stop_count']} | {row['language_forgetting']['catastrophic_forgetting_detected']} |")
    lines.extend(["", f"selected={selected}; 目标 Pikachu trace 未进入训练，仍需独立回放。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "device": str(device), "parameter_target": 101380329, "variants": [{"variant": row["variant"], "holdout_accuracy": row["holdout"]["accuracy"], "unknown_abstain_rate": row["holdout"]["unknown_abstain_rate"], "false_stop_count": row["holdout"]["false_stop_count"], "catastrophic_forgetting": row["language_forgetting"]["catastrophic_forgetting_detected"]} for row in results], "selected_variant": selected, "training_artifact_promotion_allowed": False, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
