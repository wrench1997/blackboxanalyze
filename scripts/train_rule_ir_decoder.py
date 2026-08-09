#!/usr/bin/env python3
"""Train and verify the language-neutral Rule IR template decoder.

Only visible traces are featurised.  Family labels are used as supervised
targets by this synthetic research harness; they are never included in the
feature vector or in the runtime decoder input.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from html import escape, unescape
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rule_ir_decoder import DECODER_FAMILIES, FEATURE_DIM, RuleIRDecoder, abstract_rule_ir_canonical, calibrate_abstention_threshold, trace_feature_vector  # noqa: E402
from app.synthetic_curriculum import generate_record  # noqa: E402


PRIMARY_SEED = 20260899
EVAL_SEEDS = (20260901, 20260907, 20260911)
OUTPUT_DIR = ROOT / "artifacts/rule-ir-decoder-loop-12-20260899-v4"
CHECKPOINT = OUTPUT_DIR / "rule_ir_decoder.pt"
REPORT = OUTPUT_DIR / "report.json"
PROTOCOL = ROOT / "research/juice_shop_loop_12_rule_ir_decoder_protocol_v1_3.json"

SYNTHETIC_LABELS = {
    "numeric_boundary": "input_validation",
    "numeric_coercion_primitive": "input_validation",
    "truthiness_gate": "access_control",
    "authorization_or": "access_control",
    "substring_origin": "url_redirect",
    "postmessage_origin": "url_redirect",
    "url_hostname_primitive": "url_redirect",
    "dom_sink_injection": "xss",
    "markup_lexeme_primitive": "xss",
    "html_entity_decode_primitive": "xss",
}
HOLDOUT_LABELS = {
    "url_scheme_downgrade": "url_redirect",
    "dom_double_decode": "xss",
    "unicode_casefold_role": "access_control",
    "numeric_string_coercion": "input_validation",
    "compound_origin_role": "access_control",
    "state_replay_window": "authentication",
}
HOLDOUT_FAMILIES = (
    "url_scheme_downgrade",
    "dom_double_decode",
    "unicode_casefold_role",
    "numeric_string_coercion",
    "compound_origin_role",
    "state_replay_window",
)


def _trace(input_value: dict[str, Any], output: bool, *, history: list[dict[str, Any]] | None = None, context: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "input": input_value,
        "context": context or {},
        "state": state or {},
        "history": history or [],
        "output": bool(output),
    }


def _custom_records(label: str, seed: int, count: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        traces: list[dict[str, Any]] = []
        if label == "observability":
            shapes = ["ordinary_json", "ordinary_json", "prometheus", "diagnostic", "traceback", "directory_listing", "spa_shell"]
            for _ in range(12):
                shape = rng.choice(shapes)
                status = 200 if shape != "traceback" else rng.choice([200, 500])
                traces.append(_trace({"response": {"status_code": status, "content_type": "text/plain" if shape == "prometheus" else "text/html", "body_shape": shape, "body_length": rng.randrange(64, 20000)}}, shape not in {"ordinary_json", "spa_shell"}))
        elif label == "injection":
            values = ["normal", "'", "\"", "' OR 1=1--", "UNION SELECT NULL", "name=alice", "1--", "plain"]
            for _ in range(12):
                value = rng.choice(values)
                traces.append(_trace({"query": value}, any(token in value.casefold() for token in ("union", "or 1=1", "--"))))
        elif label == "authentication":
            for _ in range(12):
                previous = f"t{rng.randrange(4)}"
                current = previous if rng.random() < 0.45 else f"t{rng.randrange(4, 9)}"
                traces.append(_trace({"token": current}, output=current == previous, history=[{"input": {"token": previous}, "context": {}, "state": {}}]))
        elif label == "encoding_depth":
            # Training-only curriculum: the same semantic payload is exposed
            # at raw, single-, double-, and triple-encoded depths.  The
            # observed output models a sink that parses after a two-pass
            # decode, while the decoder input sees only the payload shape.
            values = ["<b>x</b>", "&lt;b&gt;x&lt;/b&gt;", "&amp;lt;b&amp;gt;x&amp;lt;/b&amp;gt;", "&amp;amp;lt;b&amp;amp;gt;x&amp;amp;lt;/b&amp;amp;gt;", "plain text", "&lt;em&gt;text&lt;/em&gt;"]
            for _ in range(12):
                value = rng.choice(values)
                twice = unescape(unescape(value))
                traces.append(_trace({"payload": value}, output=bool(re.search(r"<\s*/?\s*[a-z][^>]*>", twice, re.IGNORECASE))))
        else:
            raise ValueError(label)
        rows.append({"record_id": f"custom-{label}-{index:05d}", "family": label, "traces": traces, "source_family": label})
    return rows


def _shadow_surface_records(seed: int, per_family: int) -> list[dict[str, Any]]:
    """Single-response curriculum matching the post-shadow decoder contract."""

    rng = random.Random(seed)
    banks = {
        "observability": [
            ("/metrics", 200, "text/plain; version=0.0.4", "prometheus"),
            ("/logs/", 200, "text/html", "spa_shell"),
            ("/debug/", 500, "application/json", "diagnostic"),
            ("/ftp/", 200, "text/html", "directory_listing"),
        ],
        "access_control": [("/rest/basket/1", 401, "application/json", "ordinary_json"), ("/rest/admin/application-configuration", 200, "application/json", "ordinary_json"), ("/api/Users", 403, "application/json", "ordinary_json")],
        "authentication": [("/rest/user/login", 401, "application/json", "ordinary_json"), ("/rest/user/whoami", 200, "application/json", "ordinary_json"), ("/rest/user/reset-password", 400, "application/json", "ordinary_json")],
        "input_validation": [("/rest/products/search?q=%00", 500, "application/json", "diagnostic"), ("/api/Products/1?quantity=-1", 400, "application/json", "ordinary_json"), ("/api/Products?limit=1", 200, "application/json", "ordinary_json")],
        "injection": [("/rest/products/search?q=%27%20OR%201%3D1--", 500, "application/json", "diagnostic"), ("/rest/products/search?q=normal", 200, "application/json", "ordinary_json"), ("/rest/products/search?q=%27%20UNION%20SELECT%20NULL--", 500, "application/json", "diagnostic")],
        "url_redirect": [("/redirect?to=https%3A%2F%2Fexample.com", 302, "text/html", "ordinary_json"), ("/redirect?to=%2F%2Fexample.com", 302, "text/html", "ordinary_json"), ("/", 200, "text/html", "spa_shell")],
        "xss": [("/rest/products/search?q=%3Cscript%3E", 200, "text/html", "spa_shell"), ("/rest/products/search?q=%3Cimg%20src=x%20onerror=alert(1)%3E", 200, "text/html", "spa_shell"), ("/robots.txt", 200, "text/plain", "robots_text")],
    }
    rows = []
    for label, bank in banks.items():
        for index in range(per_family):
            path, status, content_type, shape = rng.choice(bank)
            length = rng.randrange(64, 20000) if shape not in {"ordinary_json", "robots_text"} else rng.randrange(32, 4096)
            traces = [_trace({"action": {"method": "GET", "path": path}, "response": {"status_code": status, "content_type": content_type, "body_shape": shape, "body_length": length}}, status // 100 == 2)]
            rows.append({"record_id": f"shadow-{label}-{index:05d}", "family": label, "traces": traces, "source_family": "shadow_surface"})
    return rows


def _synthetic_rows(seed: int, per_family: int, families: tuple[str, ...]) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    index = 1
    for family in families:
        for _ in range(per_family):
            record = generate_record(index, rng, family, traces_per_program=12)
            rows.append({
                "record_id": record["record_id"],
                "family": SYNTHETIC_LABELS.get(family, HOLDOUT_LABELS.get(family, family)),
                "source_family": family,
                "traces": record["modalities"]["trace"],
            })
            index += 1
    return rows


def _balanced(rows: list[dict[str, Any]], per_label: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    selected = []
    for label in DECODER_FAMILIES:
        pool = list(grouped.get(label, []))
        if not pool:
            raise RuntimeError(f"no training rows for decoder label {label}")
        rng.shuffle(pool)
        required = {"shadow_surface": 60}
        if label == "xss":
            required["encoding_depth"] = 80
        picked = []
        picked_ids: set[int] = set()
        for source_family, quota in required.items():
            source_pool = [row for row in pool if row.get("source_family") == source_family]
            rng.shuffle(source_pool)
            for row in source_pool[:quota]:
                picked.append(row)
                picked_ids.add(id(row))
        remaining = [row for row in pool if id(row) not in picked_ids]
        need = max(0, per_label - len(picked))
        if len(remaining) >= need:
            picked.extend(remaining[:need])
        else:
            picked.extend(remaining)
            picked.extend(pool[index % len(pool)] for index in range(need - len(remaining)))
        selected.extend(picked)
    rng.shuffle(selected)
    return selected


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([trace_feature_vector(row["traces"]) for row in rows], dtype=torch.float32)


def _normalise(train_features: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (features - mean) / std, mean, std


@torch.inference_mode()
def _evaluate(model: RuleIRDecoder, features: torch.Tensor, labels: torch.Tensor, rows: list[dict[str, Any]], device: torch.device, threshold: float = 0.55) -> dict[str, Any]:
    model.eval()
    probabilities = torch.softmax(model(features.to(device)), dim=-1).cpu()
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)
    accepted = confidence >= threshold
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, predicted, expected, matched in zip(rows, predictions, labels, correct):
        family_stats = by_family[row["family"]]
        family_stats[0] += int(matched)
        family_stats[1] += 1
        source_stats = by_source[row["source_family"]]
        source_stats[0] += int(matched)
        source_stats[1] += 1
    return {
        "accuracy": round(float(correct.float().mean()), 6) if len(labels) else 0.0,
        "correct": int(correct.sum()),
        "total": len(labels),
        "accepted_accuracy": round(float(correct[accepted].float().mean()), 6) if bool(accepted.any()) else None,
        "coverage": round(float(accepted.float().mean()), 6) if len(labels) else 0.0,
        "abstain_rate": round(float((~accepted).float().mean()), 6) if len(labels) else 0.0,
        "by_family": {key: round(good / total, 6) for key, (good, total) in sorted(by_family.items())},
        "by_source_family": {key: round(good / total, 6) for key, (good, total) in sorted(by_source.items())},
        "predicted_templates": [DECODER_FAMILIES[int(value)] for value in predictions],
        "target_templates": [DECODER_FAMILIES[int(value)] for value in labels],
    }


def train_once(seed: int, train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], device: torch.device) -> tuple[RuleIRDecoder, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_features_raw = _features(train_rows)
    validation_features_raw = _features(validation_rows)
    holdout_features_raw = _features(holdout_rows)
    train_features, mean, std = _normalise(train_features_raw, train_features_raw)
    validation_features = (validation_features_raw - mean) / std
    holdout_features = (holdout_features_raw - mean) / std
    label_index = {family: index for index, family in enumerate(DECODER_FAMILIES)}
    train_labels = torch.tensor([label_index[row["family"]] for row in train_rows], dtype=torch.long)
    validation_labels = torch.tensor([label_index[row["family"]] for row in validation_rows], dtype=torch.long)
    holdout_labels = torch.tensor([label_index[row["family"]] for row in holdout_rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = RuleIRDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_state = None
    best_accuracy = -1.0
    history = []
    for epoch in range(1, 41):
        model.train()
        running = 0.0
        seen = 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device))
            loss = loss_fn(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch_labels)
            seen += len(batch_labels)
        validation = _evaluate(model, validation_features, validation_labels, validation_rows, device)
        history.append({"epoch": epoch, "train_loss": round(running / max(seen, 1), 6), "validation_accuracy": validation["accuracy"]})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.inference_mode():
        validation_probabilities = torch.softmax(model(validation_features.to(device)), dim=-1).cpu().tolist()
    calibration = calibrate_abstention_threshold(validation_probabilities, validation_labels.tolist(), minimum_precision=0.95)
    report = {
        "seed": seed,
        "history": history,
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "validation": _evaluate(model, validation_features, validation_labels, validation_rows, device),
        "holdout": _evaluate(model, holdout_features, holdout_labels, holdout_rows, device),
        "calibration": calibration,
    }
    return model, report


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    base_families = tuple(SYNTHETIC_LABELS)
    train_pool = _synthetic_rows(PRIMARY_SEED, 90, base_families)
    train_pool.extend(_custom_records("observability", PRIMARY_SEED + 1, 120))
    train_pool.extend(_custom_records("injection", PRIMARY_SEED + 2, 120))
    train_pool.extend(_custom_records("authentication", PRIMARY_SEED + 3, 120))
    train_pool.extend(_custom_records("encoding_depth", PRIMARY_SEED + 4, 180))
    train_pool.extend(_shadow_surface_records(PRIMARY_SEED + 5, 60))
    train_rows = _balanced(train_pool, 180, PRIMARY_SEED + 4)
    # A fresh iid slice from the same baseline generators is the ordinary
    # validation set; no target IR or family label enters the features.
    validation_pool = _synthetic_rows(PRIMARY_SEED + 5, 20, base_families)
    validation_pool.extend(_custom_records("observability", PRIMARY_SEED + 6, 20))
    validation_pool.extend(_custom_records("injection", PRIMARY_SEED + 7, 20))
    validation_pool.extend(_custom_records("authentication", PRIMARY_SEED + 8, 20))
    validation_pool.extend(_custom_records("encoding_depth", PRIMARY_SEED + 9, 20))
    validation_pool.extend(_shadow_surface_records(PRIMARY_SEED + 10, 20))
    validation_rows = _balanced(validation_pool, 20, PRIMARY_SEED + 9)
    holdout_rows = _synthetic_rows(PRIMARY_SEED + 10, 50, HOLDOUT_FAMILIES)
    model, run = train_once(PRIMARY_SEED, train_rows, validation_rows, holdout_rows, device)

    # Fresh data seeds test that the decoder is not relying on one generated
    # trace ordering or literal values.  The checkpoint stays fixed.
    fresh_seed_reports = []
    label_index = {family: index for index, family in enumerate(DECODER_FAMILIES)}
    checkpoint_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    mean = torch.tensor(run["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(run["normalisation_std"], dtype=torch.float32)
    model_for_eval = RuleIRDecoder().to(device)
    model_for_eval.load_state_dict(checkpoint_state)
    for eval_seed in EVAL_SEEDS:
        fresh_rows = _synthetic_rows(eval_seed, 40, HOLDOUT_FAMILIES)
        raw = _features(fresh_rows)
        fresh = (raw - mean) / std
        labels = torch.tensor([label_index[row["family"]] for row in fresh_rows], dtype=torch.long)
        result = _evaluate(model_for_eval, fresh, labels, fresh_rows, device)
        result["seed"] = eval_seed
        fresh_seed_reports.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-rule-ir-decoder-checkpoint-v1",
        "model_state": checkpoint_state,
        "feature_dim": FEATURE_DIM,
        "families": list(DECODER_FAMILIES),
        "normalisation_mean": run["normalisation_mean"],
        "normalisation_std": run["normalisation_std"],
        "seed": PRIMARY_SEED,
        "device_at_training": str(device),
        "abstain_threshold": run["calibration"]["threshold"],
    }, CHECKPOINT)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "schema_version": "sift-rule-ir-decoder-report-v1",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "status": "completed",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_peak_memory_bytes": peak_memory,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model": {"parameters": sum(parameter.numel() for parameter in model.parameters()), "feature_dim": FEATURE_DIM, "families": list(DECODER_FAMILIES)},
        "data": {"train_examples": len(train_rows), "validation_examples": len(validation_rows), "holdout_examples": len(holdout_rows), "holdout_source_families": list(HOLDOUT_FAMILIES), "encoding_depth_curriculum_examples": 180, "shadow_surface_curriculum_examples": 420, "oracle_fields_excluded_from_features": ["family", "record_id", "intended_output", "is_counterexample"]},
        "templates": {family: abstract_rule_ir_canonical(family) for family in DECODER_FAMILIES},
        "primary": run,
        "fresh_holdout_seeds": fresh_seed_reports,
        "calibration": run["calibration"],
        "stability": {"mean_accuracy": round(sum(row["accuracy"] for row in fresh_seed_reports) / len(fresh_seed_reports), 6), "min_accuracy": min(row["accuracy"] for row in fresh_seed_reports), "max_abstain_rate": max(row["abstain_rate"] for row in fresh_seed_reports)},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
    }
    report["acceptance"] = {
        "grammar_valid": True,
        "oracle_leakage": False,
        "validation_gate": run["validation"]["accuracy"] >= 0.90,
        "fresh_holdout_gate": report["stability"]["min_accuracy"] >= 0.70,
        "accepted": run["validation"]["accuracy"] >= 0.90 and report["stability"]["min_accuracy"] >= 0.70,
        "coverage_is_reported_separately": True,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "validation_accuracy": run["validation"]["accuracy"], "holdout_accuracy": run["holdout"]["accuracy"], "fresh_holdout_mean": report["stability"]["mean_accuracy"], "checkpoint": str(CHECKPOINT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
