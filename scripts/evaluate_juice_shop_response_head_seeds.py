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

from train_juice_shop_response_head_v2 import HttpDataset, http_collate, evaluate_http  # noqa: E402
from train_neural_url_set_head import TinyRuleSetGPT  # noqa: E402


CHECKPOINT = ROOT / "artifacts/neural-juice-loop-12-response-head-v2-20262097-rerun/tiny_rule_set_gpt.pt"
SEEDS = [20262103, 20262107, 20262111]
OUTPUT = ROOT / "research/juice_shop_loop_12_response_head_fresh_seeds.json"


def main() -> None:
    from train_juice_shop_response_head_v2 import http_rows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state"])
    model.eval()
    rows = []
    for seed in SEEDS:
        examples = http_rows(600, seed)
        loader = DataLoader(HttpDataset(examples), batch_size=128, shuffle=False, collate_fn=http_collate)
        result = evaluate_http(model, loader, device)
        rows.append({"seed": seed, **result})
    report = {
        "schema_version": "sift-juice-shop-loop-12-response-head-fresh-seeds-v1",
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
