#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_neural_url_set_head import TinyRuleSetGPT  # noqa: E402
from train_juice_shop_response_head_v2 import HttpDataset, evaluate_http, http_collate, http_rows  # noqa: E402


CHECKPOINT = ROOT / "artifacts/neural-juice-loop-12-response-head-v3-20262113/tiny_rule_set_gpt.pt"
SEEDS = [20262117, 20262119, 20262123, 20262129]
OUTPUT = ROOT / "research/juice_shop_loop_12_response_head_v3_fresh_seeds.json"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    rows = []
    for seed in SEEDS:
        examples = http_rows(900, seed)
        loader = DataLoader(HttpDataset(examples), batch_size=128, shuffle=False, collate_fn=http_collate)
        rows.append({"seed": seed, **evaluate_http(model, loader, device)})
    report = {
        "schema_version": "sift-juice-shop-loop-12-response-head-v3-fresh-seeds-v1",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "device": str(device),
        "seeds": rows,
        "mean_accuracy": round(sum(row["accuracy"] for row in rows) / len(rows), 6),
        "minimum_accuracy": min(row["accuracy"] for row in rows),
        "accepted": min(row["accuracy"] for row in rows) >= 0.90,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
