"""PG-149: align large causal bodies with abstract next-action supervision."""

from __future__ import annotations

import hashlib
import json
import copy
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_pg148_large_model_posttraining as pg148  # noqa: E402
from app.causal_forgetting import compare_causal_lm_canary  # noqa: E402
from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402
from run_pg147_model_capacity_sweep import _generated_sequence  # noqa: E402


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg149-causal-action-alignment-v1"
REPORT = RESEARCH / "pg149_causal_action_alignment_report_v1.json"
DATASET = RESEARCH / "pg149_causal_action_alignment_dataset_v1.json"
PROTOCOL = RESEARCH / "pg149_causal_action_alignment_protocol_v1.json"
SEED = 14901
GENERATED_TARGET = 8000
MAX_LEN = 128
ACTION_NAMES = pg148.ACTION_NAMES


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _replace_prefix(tokens: list[str], prefix: str, replacement: str) -> None:
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            tokens[index] = replacement


def _labeled_sequence(rng: random.Random, index: int) -> tuple[list[str], str, bool]:
    tokens = list(_generated_sequence(rng, index))
    label = rng.choice(ACTION_NAMES)
    mapping = {
        "replay_other_method": ("auth_boundary", "candidate", "typed"),
        "repeat_matched_negative_pair": ("no_surface_delta", "negative_control", "typed"),
        "probe_candidate_other_method": ("shape_delta", "candidate", "typed"),
        "abstain_candidate_only": ("parse_error_signature", "uncertain", "typed"),
        "abstain_unknown_oracle": ("timeout_signature", "uncertain", "unknown_oracle"),
        "stop_confirmed_positive": ("shape_delta", "candidate", "typed"),
        "abstain_budget_exhausted": ("timeout_signature", "recovery", "typed"),
    }
    failure, belief, oracle = mapping[label]
    _replace_prefix(tokens, "ir.failure.kind=", f"ir.failure.kind={failure}")
    _replace_prefix(tokens, "ir.belief.phase=", f"ir.belief.phase={belief}")
    _replace_prefix(tokens, "obs.oracle=", f"obs.oracle={oracle}")
    # Keep the generated label off the LM input; it is only stored in the
    # supervised side channel below.
    return tokens, label, oracle == "typed"


def _build_dataset(vocabulary: pg148._Vocabulary) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(SEED)
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    attempts = 0
    while len(unique) < GENERATED_TARGET and attempts < GENERATED_TARGET * 20:
        attempts += 1
        tokens, label, typed = _labeled_sequence(rng, len(unique))
        key = tuple(tokens + [f"__label__={label}"])
        if key in unique:
            continue
        unique[key] = {
            "row_id": f"pg149-generated-{len(unique):06d}",
            "tokens": tokens[:MAX_LEN],
            "label": label,
            "label_index": ACTION_NAMES.index(label),
            "typed_available": typed,
            "surface_kind": "procedural_causal_family",
        }
    rows = list(unique.values())
    rng.shuffle(rows)
    train_cut = int(len(rows) * 0.8)
    dev_cut = int(len(rows) * 0.9)
    for index, row in enumerate(rows):
        row["split"] = "train" if index < train_cut else "dev" if index < dev_cut else "holdout"
    return rows, {"generated_count": len(rows), "train_count": train_cut, "dev_count": dev_cut - train_cut, "holdout_count": len(rows) - dev_cut}


def _lm_metrics(model: CausalTraceTransformer, rows: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, float]:
    encoded = [vocabulary.encode(row["tokens"][:MAX_LEN]) for row in rows]
    if not encoded:
        return {"perplexity": 0.0, "next_token_accuracy": 0.0, "token_count": 0}
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, item in enumerate(encoded):
        ids[index, : len(item)] = torch.tensor(item, dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        logits = model(ids[:, :-1], ids[:, :-1].ne(0))
    targets = ids[:, 1:]
    valid = targets.ne(0)
    count = int(valid.sum().item())
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    accuracy = float(((logits.argmax(dim=-1) == targets) & valid).sum().item() / max(count, 1))
    return {"perplexity": round(float(torch.exp(loss).item()), 8), "next_token_accuracy": round(accuracy, 8), "token_count": count}


def _train_variant(name: str, base: CausalTraceTransformer, config: dict[str, Any], mode: str, train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], real_holdout: list[dict[str, Any]], vocabulary: pg148._Vocabulary, device: torch.device) -> dict[str, Any]:
    model = pg148._ActionModel(base, int(config["d_model"])).to(device)
    if mode == "action_only":
        lm_weight = 0.0
    elif mode == "anchor_half":
        lm_weight = 0.5
    else:
        lm_weight = 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01)
    loader = DataLoader(pg148._Dataset(train_rows, vocabulary), batch_size=64 if int(config["d_model"]) <= 512 else 32, shuffle=True, collate_fn=pg148._collate)
    dev_loader = DataLoader(pg148._Dataset(dev_rows, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    holdout_loader = DataLoader(pg148._Dataset(holdout_rows, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    epochs = 3
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_loss = nn.functional.cross_entropy(model.action_logits(ids, mask), labels)
                loss = action_loss
                if lm_weight:
                    logits = model.base(ids[:, :-1], mask[:, :-1])
                    targets = ids[:, 1:]
                    lm_loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
                    loss = action_loss + lm_weight * lm_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
        history.append({"epoch": epoch, "mean_loss": round(loss_sum / max(len(loader), 1), 8), "dev": pg148._metrics(model, dev_loader, device)})
    real_loader = DataLoader(pg148._Dataset(real_holdout, vocabulary), batch_size=128, shuffle=False, collate_fn=pg148._collate)
    result = {
        "variant": name,
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "synthetic_dev": pg148._metrics(model, dev_loader, device),
        "synthetic_holdout": pg148._metrics(model, holdout_loader, device),
        "real_pg136_holdout": pg148._metrics(model, real_loader, device),
        "lm_weight": lm_weight,
        "history": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg149-causal-action-alignment-v1", "variant": name, "mode": mode, "config": config, "vocabulary": vocabulary.itos, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{name}.pt")
    return result


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pg147_dataset = json.loads((RESEARCH / "pg147_model_capacity_sweep_dataset_v1.json").read_text(encoding="utf-8"))
    vocabulary = pg148._Vocabulary(list(pg147_dataset["vocabulary"]))
    rows, stats = _build_dataset(vocabulary)
    train_rows = [row for row in rows if row["split"] == "train"]
    dev_rows = [row for row in rows if row["split"] == "dev"]
    holdout_rows = [row for row in rows if row["split"] == "holdout"]
    real_train, real_dev, real_holdout, _, _ = pg148._prepare_rows()
    large_body, large_config = pg148._load_body("large_transformer", vocabulary, device)
    xl_body, xl_config = pg148._load_body("xl_transformer", vocabulary, device)
    variants = [
        ("large_action_only_alignment", copy_model(large_body), large_config, "action_only"),
        ("large_anchor_half_alignment", copy_model(large_body), large_config, "anchor_half"),
        ("xl_anchor_full_alignment", xl_body, xl_config, "anchor_full"),
    ]
    results = [_train_variant(name, body, config, mode, train_rows, dev_rows, holdout_rows, real_holdout, vocabulary, device) for name, body, config, mode in variants]
    report = {
        "protocol_id": "pg-pk-149-causal-action-alignment-v1",
        "schema_version": "pg149-causal-action-alignment-report-v1",
        "status": "completed_pg149_causal_action_alignment",
        "device": str(device),
        "corpus": stats,
        "variants": results,
        "real_holdout_source": "pg136_causal_token_lm_dataset_v1.json",
        "labels_in_lm_input": False,
        "raw_payloads_in_model": False,
        "raw_responses_in_model": False,
        "capability_claim_allowed": False,
        "memory_promotion_allowed": False,
        "report_sha256": "",
    }
    report["report_sha256"] = _sha256_json({key: value for key, value in report.items() if key != "report_sha256"})
    dataset = {"schema_version": "pg149-causal-action-alignment-dataset-v1", "stats": stats, "splits": {"train": train_rows, "dev": dev_rows, "holdout": holdout_rows}, "labels_in_lm_input": False, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"})
    protocol = {"protocol_id": "pg-pk-149-causal-action-alignment-v1", "schema_version": "pg149-causal-action-alignment-protocol-v1", "objective": "用带 belief/failure/next-action 结构的抽象轨迹把大模型语言表征与动作监督对齐。", "variants": ["large_action_only_alignment", "large_anchor_half_alignment", "xl_anchor_full_alignment"], "label_policy": "action label is supervision-only and never tokenized", "promotion": {"capability_claim_allowed": False, "memory_promotion_allowed": False}}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "corpus": stats, "variants": [{"variant": row["variant"], "synthetic_holdout": row["synthetic_holdout"]["accuracy"], "real_pg136_holdout": row["real_pg136_holdout"]["accuracy"], "elapsed_seconds": row["elapsed_seconds"]} for row in results], "report": str(REPORT)}, ensure_ascii=False, indent=2))


def copy_model(model: CausalTraceTransformer) -> CausalTraceTransformer:
    return copy.deepcopy(model)


if __name__ == "__main__":
    main()
