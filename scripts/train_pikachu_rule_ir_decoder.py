"""Tiny GPU-aware Rule IR decoder pilot for the Pikachu staged catalog.

This is intentionally a transfer pilot, not a claim of complete
vulnerability coverage: stage-1 safe canaries train the visible surface
representation and the gated encoded variant is held out.  The decoder may
abstain; it never emits free-form exploit text.
"""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CatalogRuleIRDecoderV2,
    abstract_catalog_rule_ir,
    catalog_feature_vector,
    catalog_visible_trace,
)
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.pikachu_replay_collector import PIKACHU_BASE_URL  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM, validate_abstract_rule_ir  # noqa: E402


PROTOCOL_ID = "pg-pk-01-rule-ir-decoder-pilot-v1"
CATALOG_PATH = ROOT / "research" / "pikachu_payload_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg-pk-01-rule-decoder"
CHECKPOINT_PATH = OUTPUT_DIR / "decoder.pt"
REPORT_PATH = ROOT / "research" / "pikachu_rule_ir_decoder_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_rule_ir_decoder_v1.md"
SEED = 20260802
EPOCHS = 180
AUGMENT_REPEATS = 6
ABSTAIN_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.10


def _is_stage1(row: dict[str, Any]) -> bool:
    # Catalog validation intentionally drops evaluator-side lab metadata; the
    # sample id still carries the non-semantic stage marker for this pilot.
    return "refinement" not in str(row.get("sample_id", ""))


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)


def _normalise(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _train(train_rows: list[dict[str, Any]], *, device: torch.device) -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    raw = _features(train_rows)
    base, mean, std = _normalise(raw)
    labels = torch.tensor([CATALOG_DECODER_FAMILIES.index(row["semantic"]["family"]) for row in train_rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(CATALOG_DECODER_FAMILIES)).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    augmented = base.repeat_interleave(AUGMENT_REPEATS, dim=0)
    augmented = augmented + torch.randn_like(augmented) * 0.01
    train_features = torch.cat((base, augmented), dim=0).to(device)
    train_labels = torch.cat((labels, labels.repeat_interleave(AUGMENT_REPEATS)), dim=0).to(device)
    model = CatalogRuleIRDecoderV2(branch_dim=96, embedding_dim=64, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.02)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_features), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_features[order])
        loss = loss_fn(logits, train_labels[order])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_logits = model(base.to(device)).cpu()
            accuracy = float(train_logits.argmax(dim=-1).eq(labels).float().mean())
            epoch_loss = float(loss.detach().cpu())
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
        "train_accuracy": best_accuracy,
        "family_support": {
            family: int(counts[index]) for index, family in enumerate(CATALOG_DECODER_FAMILIES) if counts[index] > 0
        },
        "history_tail": history[-5:],
    }


@torch.inference_mode()
def _evaluate(
    model: CatalogRuleIRDecoderV2,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    fit: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    mean = torch.tensor(fit["normalisation_mean"])
    std = torch.tensor(fit["normalisation_std"])
    train_features = (_features(train_rows) - mean) / std
    test_features = (_features(test_rows) - mean) / std
    outputs = model.decode(
        test_features.to(device),
        abstain_threshold=ABSTAIN_THRESHOLD,
        margin_threshold=MARGIN_THRESHOLD,
    )
    distances = torch.cdist(test_features, train_features).min(dim=1).values.tolist()
    predictions: list[dict[str, Any]] = []
    for row, output, distance in zip(test_rows, outputs, distances):
        prediction = dict(output)
        # A held-out encoding variant is allowed to be novel.  Fail closed if
        # it is far outside the training surface instead of guessing a family.
        novel = float(distance) > 8.0
        if novel:
            prediction["family"] = None
            prediction["rule_ir"] = None
            prediction["abstained"] = True
            prediction["abstain_reason"] = "novel_visible_surface"
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
            "abstained": bool(prediction.get("abstained")),
            "abstain_reason": prediction.get("abstain_reason"),
            "exit_found": exit_found,
            "false_positive": false_positive,
            "rule_ir_emitted": prediction.get("rule_ir") is not None,
        })
    total = len(predictions)
    return {
        "split": "stage1_to_stage2_encoded_variant",
        "total": total,
        "exit_found_rate": round(sum(row["exit_found"] for row in predictions) / total, 6) if total else 0.0,
        "false_positive_rate": round(sum(row["false_positive"] for row in predictions) / total, 6) if total else 0.0,
        "abstain_rate": round(sum(row["abstained"] for row in predictions) / total, 6) if total else 0.0,
        "rule_ir_emission_rate": round(sum(row["rule_ir_emitted"] for row in predictions) / total, 6) if total else 0.0,
        "predictions": predictions,
    }


def _markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    return "\n".join([
        "# Pikachu PG-PK-01 Rule IR 解码器小试验",
        "",
        "训练只使用阶段 1 的安全 canary，阶段 2 的编码变体完全留作测试。模型看到的是 action/probe/response shape，不读取 family、evaluator 或漏洞确认标签；输出仍被限制为 grammar-checked Rule IR，低置信度或新表面必须 abstain。",
        "",
        f"设备：`{report['model']['device']}`；训练样本：{report['training']['train_count']}；测试样本：{evaluation['total']}。",
        "",
        "| split | 找到出口 | 误报 | abstain | Rule IR 输出 |",
        "|---|---:|---:|---:|---:|",
        f"| {evaluation['split']} | {evaluation['exit_found_rate']:.2f} | {evaluation['false_positive_rate']:.2f} | {evaluation['abstain_rate']:.2f} | {evaluation['rule_ir_emission_rate']:.2f} |",
        "",
        "这里的“找到出口”仅指抽象族/规则模板迁移成功，不是漏洞 evaluator 确认。数据只有 7 条，结果用于检验训练管线和 fail-closed 行为，不能外推到 Pikachu 全部漏洞。",
        "",
        f"Checkpoint：`{report['checkpoint_path']}`",
        f"完整 JSON：`{report['report_path']}`",
        "",
    ])


def main() -> None:
    started = time.perf_counter()
    rows = flatten_catalog(load_catalog(CATALOG_PATH))
    visible = json.dumps([catalog_visible_trace(row) for row in rows], ensure_ascii=False).casefold()
    assert "evaluator" not in visible and "intended_output" not in visible
    train_rows = [row for row in rows if _is_stage1(row)]
    test_rows = [row for row in rows if not _is_stage1(row)]
    if not train_rows or not test_rows:
        raise RuntimeError("Pikachu staged catalog must contain both train and holdout rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fit = _train(train_rows, device=device)
    evaluation = _evaluate(model, train_rows, test_rows, fit, device=device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "sift-pikachu-rule-ir-decoder-pilot-v1",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_dim": FEATURE_DIM,
        "families": list(CATALOG_DECODER_FAMILIES),
        "normalisation_mean": fit["normalisation_mean"],
        "normalisation_std": fit["normalisation_std"],
        "abstain_threshold": ABSTAIN_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "seed": SEED,
        "device_at_training": str(device),
    }, CHECKPOINT_PATH)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-rule-ir-decoder-pilot-report-v1",
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": load_catalog(CATALOG_PATH)["catalog_sha256"],
            "sample_count": len(rows),
        },
        "model": {
            "class": "CatalogRuleIRDecoderV2",
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "free_form_payload_generation": False,
        },
        "training": {
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_surfaces": sorted({row["semantic"]["surface"] for row in train_rows}),
            "test_surfaces": sorted({row["semantic"]["surface"] for row in test_rows}),
            **fit,
        },
        "evaluation": evaluation,
        "target_scope": {
            "base_url": PIKACHU_BASE_URL,
            "independent_target": False,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
        },
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "model": report["model"],
        "training": report["training"],
        "evaluation": {key: value for key, value in evaluation.items() if key != "predictions"},
        "predictions": evaluation["predictions"],
        "checkpoint": report["checkpoint_path"],
        "report": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
