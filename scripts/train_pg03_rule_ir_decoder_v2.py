"""Train the stronger, ambiguity-aware PG-03 Rule IR decoder baseline.

This is deliberately a separate experiment from the v1 checkpoint.  It adds
two-view surface/context encoding, class-balanced supervised contrastive
training and a minimum-support gate.  The gate is a safety mechanism: a
candidate family seen only once in the training split cannot emit a Rule IR
without more evidence.
"""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CATALOG_DECODER_V2_SCHEMA,
    CatalogRuleIRDecoderV2,
    catalog_feature_vector,
    catalog_visible_trace,
)
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM, validate_abstract_rule_ir  # noqa: E402


PROTOCOL_ID = "sift-pg03-rule-ir-decoder-v2"
CATALOG_PATH = ROOT / "research" / "payload_replay_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg03-rule-ir-decoder-v2"
REPORT_PATH = ROOT / "research" / "pg03_rule_ir_decoder_v2.json"
MARKDOWN_PATH = ROOT / "research" / "pg03_rule_ir_decoder_v2.md"
PROTOCOL_PATH = ROOT / "research" / "pg03_rule_ir_decoder_v2_protocol.json"
SEED = 20260874
EPOCHS = 220
AUGMENT_REPEATS = 5
NOISE_SCALE = 0.012
CONTRASTIVE_WEIGHT = 0.20
CONTRASTIVE_TEMPERATURE = 0.15
ABSTAIN_THRESHOLD = 0.25
MARGIN_THRESHOLD = 0.08
MIN_FAMILY_SUPPORT = 2


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)


def _normalise(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _novelty_threshold(train_features: torch.Tensor) -> float:
    if len(train_features) <= 1:
        return 8.0
    distances: list[float] = []
    for index in range(len(train_features)):
        others = torch.cat([train_features[:index], train_features[index + 1:]], dim=0)
        distances.append(float(torch.linalg.vector_norm(others - train_features[index], dim=1).min()))
    return max(8.0, round(max(distances) + 2.0, 6))


def _assert_visible_projection(rows: list[dict[str, Any]]) -> None:
    forbidden = {"family", "source_id", "semantic", "rule_ir", "evaluator", "intended_output", "is_counterexample"}
    for row in rows:
        flattened = json.dumps(catalog_visible_trace(row), ensure_ascii=False, sort_keys=True).casefold()
        assert not any(token in flattened for token in forbidden), f"label leaked into visible projection: {row['sample_id']}"


def _supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pull same-family views together without forcing ambiguous families apart."""

    if len(embeddings) < 2:
        return embeddings.sum() * 0.0
    normalized = F.normalize(embeddings, dim=-1)
    logits = normalized @ normalized.T / CONTRASTIVE_TEMPERATURE
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~diagonal
    if not bool(positive.any()):
        return embeddings.sum() * 0.0
    log_denominator = torch.logsumexp(logits.masked_fill(diagonal, float("-inf")), dim=1, keepdim=True)
    log_probability = logits - log_denominator
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_row = -(log_probability.masked_fill(~positive, 0.0).sum(dim=1) / positive_count)
    return per_row[positive.any(dim=1)].mean()


def _train_model(train_rows: list[dict[str, Any]], *, seed: int, device: torch.device) -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    raw = _features(train_rows)
    base_features, mean, std = _normalise(raw)
    labels = torch.tensor([CATALOG_DECODER_FAMILIES.index(row["semantic"]["family"]) for row in train_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(CATALOG_DECODER_FAMILIES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])

    # Multiple noisy views make the embedding learn stable structure rather
    # than memorising one source row.  The original rows remain in the batch.
    augmented = base_features.repeat_interleave(AUGMENT_REPEATS, dim=0)
    augmented = augmented + torch.randn_like(augmented) * NOISE_SCALE
    augmented_labels = labels.repeat_interleave(AUGMENT_REPEATS)
    train_features = torch.cat((base_features, augmented), dim=0)
    train_labels = torch.cat((labels, augmented_labels), dim=0)

    model = CatalogRuleIRDecoderV2(branch_dim=128, embedding_dim=96, dropout=0.08).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.02)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 32):
            batch_index = order[start:start + 32]
            batch_features = train_features[batch_index].to(device)
            batch_labels = train_labels[batch_index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            embeddings = model.encode(batch_features)
            loss = loss_fn(logits, batch_labels) + CONTRASTIVE_WEIGHT * _supervised_contrastive_loss(embeddings, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_index)

        model.eval()
        with torch.inference_mode():
            train_logits = model(base_features.to(device)).cpu()
            prediction = train_logits.argmax(dim=-1)
            accuracy = float(prediction.eq(labels).float().mean())
            epoch_loss = total_loss / max(len(train_features), 1)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 6), "train_accuracy": round(accuracy, 6)})
        if accuracy > best_accuracy or (accuracy == best_accuracy and epoch_loss < best_loss):
            best_accuracy = accuracy
            best_loss = epoch_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "novelty_threshold": _novelty_threshold(base_features),
        "train_accuracy": best_accuracy,
        "history": history,
        "family_support": {family: int(counts[index]) for index, family in enumerate(CATALOG_DECODER_FAMILIES)},
    }


def _split(rows: list[dict[str, Any]]) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    train_source_a = [row for row in rows if row["source_id"].endswith("-a")]
    test_source_b = [row for row in rows if row["source_id"].endswith("-b")]
    return {
        "source_split_same_family": (train_source_a, test_source_b),
        "family_holdout_structural_transfer": (
            [row for row in train_source_a if row["semantic"]["family"] in {"xss", "injection", "access_control"}],
            [row for row in test_source_b if row["semantic"]["family"] in {"logic", "url_redirect"}],
        ),
        "family_holdout_unseen_surface": (
            [row for row in train_source_a if row["semantic"]["family"] in {"xss", "injection"}],
            [row for row in test_source_b if row["semantic"]["family"] in {"access_control", "logic", "url_redirect"}],
        ),
    }


@torch.inference_mode()
def _evaluate(
    model: CatalogRuleIRDecoderV2,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_mean: list[float],
    train_std: list[float],
    novelty_threshold: float,
    family_support: dict[str, int],
    *,
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    mean = torch.tensor(train_mean)
    std = torch.tensor(train_std)
    train_features = (_features(train_rows) - mean) / std
    test_features = (_features(test_rows) - mean) / std
    decoded = model.decode(
        test_features.to(device),
        abstain_threshold=ABSTAIN_THRESHOLD,
        margin_threshold=MARGIN_THRESHOLD,
    )
    distances = torch.cdist(test_features, train_features).min(dim=1).values.tolist() if len(train_features) else [float("inf")] * len(test_rows)
    predictions: list[dict[str, Any]] = []
    for row, output, distance in zip(test_rows, decoded, distances):
        prediction = dict(output)
        novelty = float(distance) > novelty_threshold
        predicted = prediction.get("family")
        if novelty:
            prediction["family"] = None
            prediction["rule_ir"] = None
            prediction["abstained"] = True
            prediction["abstain_reason"] = "novel_surface_distance"
        elif predicted is not None and family_support.get(predicted, 0) < MIN_FAMILY_SUPPORT:
            prediction["family"] = None
            prediction["rule_ir"] = None
            prediction["abstained"] = True
            prediction["abstain_reason"] = "insufficient_family_support"
        expected = row["semantic"]["family"]
        predicted = prediction.get("family")
        exit_found = predicted == expected and prediction.get("rule_ir") is not None
        false_positive = predicted is not None and predicted != expected
        if prediction.get("rule_ir") is not None:
            validate_abstract_rule_ir(prediction["rule_ir"])
        predictions.append({
            "sample_id": row["sample_id"],
            "expected_family": expected,
            "predicted_family": predicted,
            "candidate_family": prediction.get("candidate_family"),
            "confidence": prediction.get("confidence", 0.0),
            "margin": prediction.get("margin", 0.0),
            "distance": round(float(distance), 6),
            "novel_surface": novelty,
            "abstain_reason": prediction.get("abstain_reason"),
            "exit_found": exit_found,
            "false_positive": false_positive,
            "abstained": bool(prediction.get("abstained")),
            "rule_ir_emitted": prediction.get("rule_ir") is not None,
        })
    total = len(predictions)
    return {
        "split": split,
        "total": total,
        "exit_found_rate": round(sum(row["exit_found"] for row in predictions) / total, 6) if total else 0.0,
        "false_positive_rate": round(sum(row["false_positive"] for row in predictions) / total, 6) if total else 0.0,
        "abstain_rate": round(sum(row["abstained"] for row in predictions) / total, 6) if total else 0.0,
        "rule_ir_emission_rate": round(sum(row["rule_ir_emitted"] for row in predictions) / total, 6) if total else 0.0,
        "predictions": predictions,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-03 Catalog Rule IR Decoder V2",
        "",
        "V2 使用 surface/context 双塔、噪声增强、监督式对比损失和最小族支持门；仍只输出 grammar-checked Rule IR。`exit_found` 不是漏洞 evaluator 确认。",
        "",
        "| split | exit found | false positive | abstain | Rule IR emitted |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["evaluations"]:
        lines.append(
            f"| {row['split']} | {row['exit_found_rate']:.2f} | {row['false_positive_rate']:.2f} | "
            f"{row['abstain_rate']:.2f} | {row['rule_ir_emission_rate']:.2f} |"
        )
    lines.extend([
        "",
        "V2 的增强重点是稳定表示和 fail-closed 决策，而不是在不可辨识的同形响应上硬猜。若训练 split 中某族少于 2 条样本，强制 abstain。",
        "",
        "边界：仍是同一 in-repo 本地 ASGI adapter 的来源/族/表面隔离；要证明独立靶场泛化，还需接入第二个授权本地实现。",
        "",
        f"原始 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(CATALOG_PATH)
    rows = flatten_catalog(load_catalog(CATALOG_PATH))
    _assert_visible_projection(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluations: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    checkpoints: list[str] = []
    for index, (split_name, (train_rows, test_rows)) in enumerate(_split(rows).items()):
        model, fit = _train_model(train_rows, seed=SEED + index, device=device)
        result = _evaluate(
            model,
            train_rows,
            test_rows,
            fit["normalisation_mean"],
            fit["normalisation_std"],
            fit["novelty_threshold"],
            fit["family_support"],
            device=device,
            split=split_name,
        )
        evaluations.append(result)
        training[split_name] = {
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_families": sorted({row["semantic"]["family"] for row in train_rows}),
            "test_families": sorted({row["semantic"]["family"] for row in test_rows}),
            "train_accuracy": round(float(fit["train_accuracy"]), 6),
            "family_support": fit["family_support"],
            "novelty_threshold": fit["novelty_threshold"],
            "history_tail": fit["history"][-5:],
        }
        checkpoint_dir = OUTPUT_DIR / split_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "decoder.pt"
        torch.save({
            "schema_version": CATALOG_DECODER_V2_SCHEMA,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "normalisation_mean": fit["normalisation_mean"],
            "normalisation_std": fit["normalisation_std"],
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "min_family_support": MIN_FAMILY_SUPPORT,
            "novelty_threshold": fit["novelty_threshold"],
            "seed": SEED + index,
            "device_at_training": str(device),
        }, checkpoint_path)
        checkpoints.append(str(checkpoint_path.relative_to(ROOT)))

    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg03-rule-ir-decoder-v2-report-v1",
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": load_catalog(CATALOG_PATH)["catalog_sha256"],
            "sample_count": len(rows),
        },
        "model": {
            "class": "CatalogRuleIRDecoderV2",
            "parameters": sum(parameter.numel() for parameter in CatalogRuleIRDecoderV2().parameters()),
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "augmentation_repeats": AUGMENT_REPEATS,
            "noise_scale": NOISE_SCALE,
            "contrastive_weight": CONTRASTIVE_WEIGHT,
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "min_family_support": MIN_FAMILY_SUPPORT,
            "novelty_gate": True,
        },
        "target_scope": {
            "base_url": "http://127.0.0.1:3100",
            "independent_target_implementation": False,
            "holdout_kind": "source, family and observable-surface isolation inside the in-repo local adapter",
        },
        "training": training,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "visible_projection_labels": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "model": report["model"],
        "evaluations": [{key: value for key, value in row.items() if key != "predictions"} for row in evaluations],
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
