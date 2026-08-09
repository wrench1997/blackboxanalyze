#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from app.synthetic_http_curriculum import feature_vector, generate_examples, inference_context, POSITIVE_SURFACES, NEGATIVE_SURFACES, render_prompt  # noqa: E402
from train_neural_url_set_head import TinyRuleSetGPT, SetPromptDataset, evaluate_set, set_collate  # noqa: E402
from train_rule_memory_pilot import Example, records_to_examples  # noqa: E402


SEED = 20262097
CHECKPOINT = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529/tiny_rule_set_gpt.pt"
OUTPUT_DIR = ROOT / "artifacts/neural-juice-loop-12-response-head-v2-20262097-rerun"
OUTPUT_CHECKPOINT = OUTPUT_DIR / "tiny_rule_set_gpt.pt"
TARGET_PARAMETERS = 908546
MAX_LENGTH = 639
URL_SLOTS = 120
RESPONSE_SLOTS = 8
REGRESSION_FAMILIES = {"numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or", "string_suffix_primitive", "url_hostname_primitive"}


class HttpDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], max_length: int = MAX_LENGTH):
        self.rows = rows
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = [byte + 1 for byte in row["prompt"].encode("utf-8")][- (self.max_length - 1):] + [257]
        features = torch.zeros(128, dtype=torch.float32)
        features[URL_SLOTS:] = torch.tensor(row["response_features"], dtype=torch.float32)
        return {"tokens": tokens, "label": int(row["label"]), "features": features}


def http_collate(rows):
    width = max(len(row["tokens"]) for row in rows)
    tokens = torch.zeros((len(rows), width), dtype=torch.long)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        tokens[index, :len(row["tokens"])] = torch.tensor(row["tokens"], dtype=torch.long)
        lengths[index] = len(row["tokens"])
    return {"tokens": tokens, "lengths": lengths, "labels": torch.tensor([row["label"] for row in rows]), "features": torch.stack([row["features"] for row in rows])}


def http_rows(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    surfaces = POSITIVE_SURFACES + NEGATIVE_SURFACES
    rows = []
    for index in range(count):
        query = rng.choice(surfaces)
        context = [rng.choice(surfaces) for _ in range(8)]
        rows.append({"prompt": render_prompt(context, query), "label": query.label, "response_features": feature_vector(query), "record_id": f"http-v2-{seed}-{index}"})
    return rows


def old_regression_examples():
    records = generate_curriculum(2700, 20, SEED)
    rows = [record for record in records if record["family"] in REGRESSION_FAMILIES]
    return records_to_examples(rows, random.Random(SEED + 3), 4, 8, routed_semantic_features=True, canonical_url_slots=True)


def old_loader(examples):
    return DataLoader(SetPromptDataset(examples, MAX_LENGTH), batch_size=128, shuffle=False, collate_fn=old_set_collate)


def old_set_collate(rows):
    batch = set_collate(rows)
    # The last eight slots are response-only under the modality gate.
    batch["url_set_features"][:, URL_SLOTS:] = 0.0
    return batch


def evaluate_old(model, examples, device):
    return evaluate_set(model, old_loader(examples), device)


@torch.inference_mode()
def evaluate_http(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        logits = model(batch["tokens"].to(device), batch["lengths"].to(device), batch["features"].to(device))
        predictions = logits.argmax(dim=-1)
        labels = batch["labels"].to(device)
        correct += int(predictions.eq(labels).sum())
        total += len(labels)
    return {"accuracy": round(correct / total, 6), "correct": correct, "total": total}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(source["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != TARGET_PARAMETERS:
        raise RuntimeError("parameter budget changed")
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    initial_url_weights = model.url_set_weights.detach().clone()
    train_rows = http_rows(2400, SEED)
    validation_rows = http_rows(600, SEED + 1)
    train_loader = DataLoader(HttpDataset(train_rows), batch_size=128, shuffle=True, collate_fn=http_collate)
    validation_loader = DataLoader(HttpDataset(validation_rows), batch_size=128, shuffle=False, collate_fn=http_collate)
    optimizer = torch.optim.AdamW([model.url_set_weights], lr=0.05, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    best = copy.deepcopy(model.state_dict())
    best_score = -1.0
    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"].to(device), batch["lengths"].to(device), batch["features"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            with torch.no_grad():
                model.url_set_weights.grad[:URL_SLOTS].zero_()
            optimizer.step()
            with torch.no_grad():
                model.url_set_weights[:URL_SLOTS].copy_(initial_url_weights[:URL_SLOTS])
            total_loss += float(loss.detach()) * len(batch["labels"])
            total += len(batch["labels"])
        validation = evaluate_http(model, validation_loader, device)
        if validation["accuracy"] > best_score:
            best_score = validation["accuracy"]
            best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        row = {"epoch": epoch, "train_loss": round(total_loss / total, 6), "response_validation_accuracy": validation["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best)
    regression = old_regression_examples()
    intervention_regression = evaluate_old(model, regression, device)
    frozen = TinyRuleSetGPT().to(device)
    frozen.load_state_dict(source["model_state"])
    frozen_regression = evaluate_old(frozen, regression, device)
    response_validation = evaluate_http(model, validation_loader, device)
    report = {
        "schema_version": "sift-juice-shop-loop-12-response-head-v2-training-v1",
        "status": "trained",
        "prior_run_invalidated": "artifacts/neural-juice-loop-12-response-head-v2-20262097/report.json",
        "seed": SEED,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "url_slots_preserved": URL_SLOTS,
        "response_slots_trained": RESPONSE_SLOTS,
        "response_train_examples": len(train_rows),
        "response_validation_examples": len(validation_rows),
        "history": history,
        "response_validation": response_validation,
        "frozen_regression": frozen_regression,
        "intervention_regression": intervention_regression,
        "regression_deltas": {family: round(intervention_regression["by_family"][family] - frozen_regression["by_family"][family], 6) for family in frozen_regression["by_family"]},
        "worst_regression_delta": round(min(intervention_regression["by_family"][family] - frozen_regression["by_family"][family] for family in frozen_regression["by_family"]), 6),
        "response_validation_pass": response_validation["accuracy"] >= 0.90,
        "old_regression_pass": min(intervention_regression["by_family"][family] - frozen_regression["by_family"][family] for family in frozen_regression["by_family"]) >= -0.02,
        "parameter_budget_pass": sum(parameter.numel() for parameter in model.parameters()) == TARGET_PARAMETERS,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_checkpoint": str(OUTPUT_CHECKPOINT.relative_to(ROOT)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"parameters": TARGET_PARAMETERS, "response_slots": RESPONSE_SLOTS, "training_seed": SEED}}, OUTPUT_CHECKPOINT)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "response_validation": response_validation["accuracy"], "worst_regression_delta": report["worst_regression_delta"], "accepted_for_shadow": report["response_validation_pass"] and report["old_regression_pass"] and report["parameter_budget_pass"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
