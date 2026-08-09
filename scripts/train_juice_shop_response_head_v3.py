#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_neural_url_set_head import TinyRuleSetGPT  # noqa: E402
from train_juice_shop_response_head_v2 import HttpDataset, evaluate_http, http_collate, http_rows, old_loader, old_regression_examples, evaluate_old, URL_SLOTS  # noqa: E402


SEED = 20262113
BASE = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529/tiny_rule_set_gpt.pt"
OUTPUT = ROOT / "artifacts/neural-juice-loop-12-response-head-v3-20262113"
CHECKPOINT = OUTPUT / "tiny_rule_set_gpt.pt"
TARGET_PARAMETERS = 908546


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.load(BASE, map_location="cpu", weights_only=False)
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(source["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != TARGET_PARAMETERS:
        raise RuntimeError("parameter budget changed")
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    original_url_weights = model.url_set_weights.detach().clone()
    train_rows = http_rows(3000, SEED)
    validation_rows = http_rows(900, SEED + 1)
    train_loader = DataLoader(HttpDataset(train_rows), batch_size=128, shuffle=True, collate_fn=http_collate)
    validation_loader = DataLoader(HttpDataset(validation_rows), batch_size=128, shuffle=False, collate_fn=http_collate)
    optimizer = torch.optim.AdamW([model.url_set_weights], lr=0.05, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    started = time.perf_counter()
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
                model.url_set_weights[:URL_SLOTS].copy_(original_url_weights[:URL_SLOTS])
            total_loss += float(loss.detach()) * len(batch["labels"])
            total += len(batch["labels"])
        validation = evaluate_http(model, validation_loader, device)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        row = {"epoch": epoch, "train_loss": round(total_loss / total, 6), "response_validation_accuracy": validation["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    regression = old_regression_examples()
    intervention_regression = evaluate_old(model, regression, device)
    frozen = TinyRuleSetGPT().to(device)
    frozen.load_state_dict(source["model_state"])
    frozen_regression = evaluate_old(frozen, regression, device)
    response_validation = evaluate_http(model, validation_loader, device)
    regression_deltas = {family: round(intervention_regression["by_family"][family] - frozen_regression["by_family"][family], 6) for family in frozen_regression["by_family"]}
    report = {
        "schema_version": "sift-juice-shop-loop-12-response-head-v3-training-v1",
        "status": "trained",
        "seed": SEED,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "response_train_examples": len(train_rows),
        "response_validation_examples": len(validation_rows),
        "history": history,
        "response_validation": response_validation,
        "frozen_regression": frozen_regression,
        "intervention_regression": intervention_regression,
        "regression_deltas": regression_deltas,
        "worst_regression_delta": min(regression_deltas.values()),
        "response_validation_pass": response_validation["accuracy"] >= 0.95,
        "old_regression_pass": min(regression_deltas.values()) >= -0.02,
        "parameter_budget_pass": sum(parameter.numel() for parameter in model.parameters()) == TARGET_PARAMETERS,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"parameters": TARGET_PARAMETERS, "response_slots": 8, "training_seed": SEED}}, CHECKPOINT)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "response_validation": response_validation["accuracy"], "worst_regression_delta": report["worst_regression_delta"], "accepted_for_fresh_seeds": report["response_validation_pass"] and report["old_regression_pass"] and report["parameter_budget_pass"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
