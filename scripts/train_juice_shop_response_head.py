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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from app.synthetic_http_curriculum import generate_examples  # noqa: E402
from train_neural_url_set_head import TinyRuleSetGPT, SetPromptDataset, evaluate_set, set_collate  # noqa: E402
from train_rule_memory_pilot import records_to_examples, stratified_iid_split  # noqa: E402


SEED = 20262091
CHECKPOINT = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529/tiny_rule_set_gpt.pt"
OUTPUT_DIR = ROOT / "artifacts/neural-juice-loop-12-response-head-20262091"
OUTPUT_CHECKPOINT = OUTPUT_DIR / "tiny_rule_set_gpt.pt"
TARGET_PARAMETERS = 908546
MAX_LENGTH = 639
REGRESSION_FAMILIES = {
    "numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or",
    "string_suffix_primitive", "url_hostname_primitive",
}
TRAIN_FAMILIES = {
    "numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or",
    "string_suffix_primitive", "markup_lexeme_primitive", "url_hostname_primitive",
    "html_entity_decode_primitive", "casefold_primitive", "numeric_coercion_primitive",
}


def examples_to_examples(rows: list[dict[str, Any]]):
    from train_rule_memory_pilot import Example
    return [Example(row["prompt"], int(row["label"]), row["family"], row["record_id"], int(row["intended_label"])) for row in rows]


def loader(examples, shuffle: bool) -> DataLoader:
    return DataLoader(SetPromptDataset(examples, MAX_LENGTH), batch_size=64, shuffle=shuffle, collate_fn=set_collate)


def make_old_regression_examples():
    records = generate_curriculum(2700, 20, SEED)
    all_train = [record for record in records if record["family"] in TRAIN_FAMILIES]
    train_records, _ = stratified_iid_split(all_train)
    regression_records = [record for record in records if record["family"] in REGRESSION_FAMILIES]
    url_meta_records = [record for record in train_records if record["family"] in {"string_suffix_primitive", "url_hostname_primitive"}]
    stable_label_records = [record for record in train_records if record["family"] not in {"string_suffix_primitive", "url_hostname_primitive"}]
    train_examples = records_to_examples(url_meta_records, random.Random(SEED), 4, 8, routed_semantic_features=True, canonical_url_slots=True, meta_label_permutation=True, permutation_seed=SEED)
    train_examples.extend(records_to_examples(stable_label_records, random.Random(SEED + 17), 4, 8, routed_semantic_features=True, canonical_url_slots=True))
    rehearsal = train_examples[:1200]
    regression = records_to_examples(regression_records, random.Random(SEED + 3), 4, 8, routed_semantic_features=True, canonical_url_slots=True)
    return rehearsal, regression


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = TinyRuleSetGPT().to(device)
    model.load_state_dict(checkpoint["model_state"])
    if sum(parameter.numel() for parameter in model.parameters()) != TARGET_PARAMETERS:
        raise RuntimeError("parameter budget changed")
    model.url_set_weights.requires_grad = False

    response_train = examples_to_examples(generate_examples(2400, SEED))
    response_validation = examples_to_examples(generate_examples(600, SEED + 1, validation=True))
    rehearsal, regression = make_old_regression_examples()
    mixed = response_train + rehearsal
    random.Random(SEED + 2).shuffle(mixed)
    train_loader = loader(mixed, True)
    response_loader = loader(response_validation, False)
    regression_loader = loader(regression, False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    best_score = -1.0
    for epoch in range(1, 5):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            labels = batch["labels"].to(device)
            logits = model(batch["tokens"].to(device), batch["lengths"].to(device), batch["url_set_features"].to(device))
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        validation = evaluate_set(model, response_loader, device)
        if validation["accuracy"] > best_score:
            best_score = validation["accuracy"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        row = {"epoch": epoch, "train_loss": round(total_loss / seen, 6), "response_validation_accuracy": validation["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    intervention_response = evaluate_set(model, response_loader, device)
    intervention_regression = evaluate_set(model, regression_loader, device)
    frozen_model = TinyRuleSetGPT().to(device)
    frozen_model.load_state_dict(checkpoint["model_state"])
    frozen_regression = evaluate_set(frozen_model, regression_loader, device)
    report = {
        "schema_version": "sift-juice-shop-loop-12-response-head-training-v1",
        "status": "trained",
        "seed": SEED,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "base_checkpoint_sha256": "37A627806AC5F796E2D90408AD04DCD73F9642178E1B68DB9E32DA20E4F1DE32",
        "response_train_examples": len(response_train),
        "response_validation_examples": len(response_validation),
        "rehearsal_examples": len(rehearsal),
        "epochs": 4,
        "history": history,
        "response_validation": intervention_response,
        "frozen_regression": frozen_regression,
        "intervention_regression": intervention_regression,
        "worst_regression_delta": round(min(intervention_regression["by_family"][family] - frozen_regression["by_family"][family] for family in frozen_regression["by_family"]), 6),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_checkpoint": str(OUTPUT_CHECKPOINT.relative_to(ROOT)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"parameters": TARGET_PARAMETERS, "response_training_seed": SEED}}, OUTPUT_CHECKPOINT)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "response_validation_accuracy": intervention_response["accuracy"], "worst_regression_delta": report["worst_regression_delta"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
