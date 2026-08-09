#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.research_events import emit_event  # noqa: E402
from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from train_rule_memory_pilot import (  # noqa: E402
    PAD,
    PromptDataset,
    TinyRuleGPT,
    collate,
    evaluate,
    records_to_examples,
    stratified_iid_split,
)


SEED = 20261529
MAX_LENGTH = 639
TARGET_PARAMETERS = 908546
TRAIN_FAMILIES = {
    "numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or",
    "string_suffix_primitive", "markup_lexeme_primitive", "url_hostname_primitive",
    "html_entity_decode_primitive", "casefold_primitive", "numeric_coercion_primitive",
}
REGRESSION_FAMILIES = {
    "numeric_boundary", "truthiness_gate", "substring_origin", "authorization_or",
    "string_suffix_primitive", "url_hostname_primitive",
}
URL_META_FAMILIES = {"string_suffix_primitive", "url_hostname_primitive"}


def _parse_url(value: str) -> tuple[str, str, str, str] | None:
    match = re.fullmatch(r"url\.value=u:([^|]+)\|([^|]+)\|p(\d+)\|(.+)", value)
    return match.groups() if match else None


def url_set_features(prompt: str) -> torch.Tensor:
    features = torch.zeros(128, dtype=torch.float32)
    trace = prompt.split("<TRACE>", 1)[1].split("<RULEMEM>", 1)[0]
    query_text = prompt.split("<QUERY>", 1)[1].split("<ANSWER>", 1)[0]
    query = _parse_url(query_text)
    if query is None or "url.value=" not in trace:
        return features
    rows: list[tuple[int, int]] = []
    for segment in re.split(r"\|(?=url\.value=)", trace):
        match = re.fullmatch(r"(url\.value=u:[^|]+\|[^|]+\|p\d+\|.+):([01])", segment)
        if not match:
            continue
        parsed = _parse_url(match.group(1))
        if parsed is None:
            continue
        equality_mask = sum((1 << index) for index, (left, right) in enumerate(zip(parsed, query)) if left == right)
        rows.append((equality_mask, int(match.group(2))))
    if not rows:
        return features
    for label in (1, 0):
        masks = [mask for mask, row_label in rows if row_label == label]
        if not masks:
            continue
        conjunction_offset = 0 if label == 1 else 16
        conjunction_any_offset = 32 if label == 1 else 48
        exact_offset = 64 if label == 1 else 80
        exact_any_offset = 96 if label == 1 else 112
        for pattern in range(16):
            conjunction = [int(mask & pattern == pattern) for mask in masks]
            exact = [int(mask == pattern) for mask in masks]
            features[conjunction_offset + pattern] = sum(conjunction) / len(conjunction)
            features[conjunction_any_offset + pattern] = max(conjunction)
            features[exact_offset + pattern] = sum(exact) / len(exact)
            features[exact_any_offset + pattern] = max(exact)
    return features


class SetPromptDataset(PromptDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        row["url_set_features"] = url_set_features(row["example"].prompt)
        return row


def set_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate(rows)
    batch["url_set_features"] = torch.stack([row["url_set_features"] for row in rows])
    return batch


class TinyRuleSetGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyRuleGPT(MAX_LENGTH, 128, 4, 4)
        self.url_set_weights = nn.Parameter(torch.zeros(128))

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, url_features: torch.Tensor, disable_set_head: bool = False) -> torch.Tensor:
        logits = self.base(tokens, lengths)
        if disable_set_head:
            return logits
        score = url_features.to(logits.dtype).matmul(self.url_set_weights)
        return logits + torch.stack((-score / 2, score / 2), dim=-1)


@torch.inference_mode()
def evaluate_set(model: TinyRuleSetGPT, loader: DataLoader, device: torch.device, disable_set_head: bool = False) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    families: dict[str, list[int]] = {}
    failures = []
    for batch in loader:
        labels = batch["labels"].to(device)
        predictions = model(
            batch["tokens"].to(device),
            batch["lengths"].to(device),
            batch["url_set_features"].to(device),
            disable_set_head=disable_set_head,
        ).argmax(dim=-1)
        matches = predictions.eq(labels)
        correct += int(matches.sum())
        total += len(labels)
        for example, prediction, expected, matched in zip(batch["examples"], predictions.cpu(), labels.cpu(), matches.cpu()):
            stats = families.setdefault(example.family, [0, 0])
            stats[0] += int(matched)
            stats[1] += 1
            if not matched and len(failures) < 12:
                failures.append({"family": example.family, "record_id": example.record_id, "expected": int(expected), "predicted": int(prediction), "prompt": example.prompt[:600]})
    return {
        "accuracy": round(correct / total, 6),
        "correct": correct,
        "total": total,
        "by_family": {family: round(good / count, 6) for family, (good, count) in families.items()},
        "failures": failures,
    }


def loader(examples, shuffle: bool) -> DataLoader:
    return DataLoader(SetPromptDataset(examples, MAX_LENGTH), batch_size=64, shuffle=shuffle, collate_fn=set_collate)


def feature_name(index: int) -> str:
    blocks = ["positive_mean_conjunction", "negative_mean_conjunction", "positive_any_conjunction", "negative_any_conjunction", "positive_mean_exact", "negative_mean_exact", "positive_any_exact", "negative_any_exact"]
    return f"{blocks[index // 16]}_mask_{index % 16:04b}"


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    records = generate_curriculum(2700, 20, SEED)
    all_train = [record for record in records if record["family"] in TRAIN_FAMILIES]
    train_records, validation_records = stratified_iid_split(all_train)
    target_records = [record for record in records if record["family"] == "url_scheme_downgrade"]
    regression_records = [record for record in records if record["family"] in REGRESSION_FAMILIES]
    common = {"routed_semantic_features": True, "canonical_url_slots": True}
    url_meta_records = [record for record in train_records if record["family"] in URL_META_FAMILIES]
    stable_label_records = [record for record in train_records if record["family"] not in URL_META_FAMILIES]
    train_examples = records_to_examples(url_meta_records, random.Random(SEED), 4, 8, meta_label_permutation=True, permutation_seed=SEED, **common)
    train_examples.extend(records_to_examples(stable_label_records, random.Random(SEED + 17), 4, 8, **common))
    random.Random(SEED + 31).shuffle(train_examples)
    validation_examples = records_to_examples(validation_records, random.Random(SEED + 1), 4, 8, **common)
    target_examples = records_to_examples(target_records, random.Random(SEED + 2), 4, 8, **common)
    regression_examples = records_to_examples(regression_records, random.Random(SEED + 3), 4, 8, **common)
    train_loader = loader(train_examples, True)
    validation_loader = loader(validation_examples, False)
    target_loader = loader(target_examples, False)
    regression_loader = loader(regression_examples, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRuleSetGPT().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != TARGET_PARAMETERS:
        raise RuntimeError(f"parameter budget violated: {parameter_count} != {TARGET_PARAMETERS}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_score = -math.inf
    best_state = None
    started = time.perf_counter()
    emit_event(actor="set-head-trainer", tool="tiny-rule-set-gpt.train", phase="training", status="running", message="启动固定参数预算 URL set-head 训练", payload={"seed": SEED, "parameters": parameter_count, "target_examples_in_training": 0})
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch["tokens"].to(device), batch["lengths"].to(device), batch["url_set_features"].to(device))
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        validation = evaluate_set(model, validation_loader, device)
        score = validation["accuracy"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        row = {"epoch": epoch, "train_loss": round(running_loss / seen, 6), "validation_accuracy": validation["accuracy"]}
        history.append(row)
        print(json.dumps(row), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    target = evaluate_set(model, target_loader, device)
    target_without_head = evaluate_set(model, target_loader, device, disable_set_head=True)
    regression = evaluate_set(model, regression_loader, device)
    regression_without_head = evaluate_set(model, regression_loader, device, disable_set_head=True)
    baseline_checkpoint = torch.load(ROOT / "artifacts/neural-url-loop-11-pilot-20261231/tiny_rule_gpt.pt", map_location="cpu", weights_only=False)
    baseline_model = TinyRuleGPT(640, 128, 4, 4).to(device)
    baseline_model.load_state_dict(baseline_checkpoint["model_state"])
    baseline_regression_loader = DataLoader(PromptDataset(regression_examples, 640), batch_size=64, shuffle=False, collate_fn=collate)
    baseline_regression = evaluate(baseline_model, baseline_regression_loader, device, max_length=640)
    regression_comparison = {
        family: {
            "canonical_meta_baseline": baseline_regression["by_family"][family],
            "set_head_candidate": regression["by_family"][family],
            "delta": round(regression["by_family"][family] - baseline_regression["by_family"][family], 6),
        }
        for family in sorted(REGRESSION_FAMILIES)
    }
    weights = model.url_set_weights.detach().cpu()
    strongest = sorted(
        ({"feature": feature_name(index), "weight": round(float(weight), 6)} for index, weight in enumerate(weights)),
        key=lambda row: abs(row["weight"]),
        reverse=True,
    )[:16]
    report = {
        "schema_version": "sift-neural-url-meta-v2-pilot-v1",
        "experiment": "neural-url-loop-11-url-meta-v2-pilot",
        "status": "completed",
        "device": str(device),
        "model": {"parameters": parameter_count, "base_positions": MAX_LENGTH, "set_head_parameters": 128, "hidden": 128, "layers": 4, "heads": 4},
        "data": {"programs": 2700, "train_examples": len(train_examples), "validation_examples": len(validation_examples), "target_examples": len(target_examples), "target_family_examples_in_training": 0, "meta_label_families": sorted(URL_META_FAMILIES)},
        "results": {
            "target_neural": target,
            "target_same_checkpoint_without_set_head": target_without_head,
            "set_head_ablation_gain": round(target["accuracy"] - target_without_head["accuracy"], 6),
            "regression": regression,
            "regression_without_head": regression_without_head,
            "regression_comparison": regression_comparison,
            "worst_regression_delta": min(row["delta"] for row in regression_comparison.values()),
        },
        "interpretability": {"strongest_learned_weights": strongest},
        "training": {"seed": SEED, "epochs": 10, "seconds": round(time.perf_counter() - started, 3), "history": history},
        "decision": {
            "pilot_target_passes": target["accuracy"] >= 0.70,
            "head_ablation_passes": target["accuracy"] - target_without_head["accuracy"] >= 0.10,
            "old_regression_passes": min(row["delta"] for row in regression_comparison.values()) >= -0.02,
            "ready_for_fresh_confirmation": target["accuracy"] >= 0.70 and target["accuracy"] - target_without_head["accuracy"] >= 0.10 and min(row["delta"] for row in regression_comparison.values()) >= -0.02,
        },
        "lineage": {"preregistration": "research/neural_url_loop_11_url_meta_v2_preregistration.json", "parent_pilot": "research/neural_url_loop_11_selective_meta_pilot.json"},
    }
    output_dir = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"max_length": MAX_LENGTH, "hidden": 128, "layers": 4, "heads": 4, "parameters": parameter_count}}, output_dir / "tiny_rule_set_gpt.pt")
    report_path = ROOT / "research/neural_url_loop_11_url_meta_v2_pilot.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(actor="set-head-trainer", tool="tiny-rule-set-gpt.evaluate", phase="family_holdout", status="complete", message=f"Set-head URL pilot {target['accuracy']:.2%}; ablation {report['results']['set_head_ablation_gain']:+.2%}", payload={"target": target["accuracy"], "ablation": report["results"]["set_head_ablation_gain"], "worst_regression": report["results"]["worst_regression_delta"], "decision": report["decision"]}, artifact=str(report_path.relative_to(ROOT)))
    print(json.dumps({"target": target["accuracy"], "without_head": target_without_head["accuracy"], "ablation_gain": report["results"]["set_head_ablation_gain"], "worst_regression": report["results"]["worst_regression_delta"], "decision": report["decision"], "strongest_weights": strongest[:6]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
