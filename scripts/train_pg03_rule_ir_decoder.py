"""Train and evaluate the small PG-03 Catalog Rule IR decoder."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CATALOG_DECODER_SCHEMA,
    CatalogRuleIRDecoder,
    catalog_feature_vector,
    catalog_visible_trace,
)
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "sift-pg03-rule-ir-decoder-v1"
CATALOG_PATH = ROOT / "research" / "payload_replay_catalog_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg03-rule-ir-decoder"
REPORT_PATH = ROOT / "research" / "pg03_rule_ir_decoder_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg03_rule_ir_decoder_v1.md"
SEED = 20260873
ABSTAIN_THRESHOLD = 0.25
MARGIN_THRESHOLD = 0.08
EPOCHS = 160


def _features(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)


def _normalise(train_raw: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_raw.mean(dim=0)
    std = train_raw.std(dim=0, unbiased=False).clamp_min(1e-4)
    return (raw - mean) / std, mean, std


def _novelty_threshold(train_features: torch.Tensor) -> float:
    if len(train_features) <= 1:
        return 8.0
    distances: list[float] = []
    for index in range(len(train_features)):
        others = torch.cat([train_features[:index], train_features[index + 1:]], dim=0)
        distances.append(float(torch.linalg.vector_norm(others - train_features[index], dim=1).min()))
    # A conservative bound keeps exact/source-shift views in-distribution but
    # rejects surfaces absent from the training source split.
    return max(8.0, round(max(distances) + 2.0, 6))


def _assert_visible_projection(rows: list[dict[str, Any]]) -> None:
    forbidden = {"family", "source_id", "semantic", "rule_ir", "evaluator", "intended_output", "is_counterexample"}
    for row in rows:
        trace = catalog_visible_trace(row)
        flattened = json.dumps(trace, ensure_ascii=False, sort_keys=True).casefold()
        assert not any(token in flattened for token in forbidden), f"label leaked into visible projection: {row['sample_id']}"


def _train_model(train_rows: list[dict[str, Any]], *, seed: int, device: torch.device) -> tuple[CatalogRuleIRDecoder, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw = _features(train_rows)
    features, mean, std = _normalise(raw, raw)
    labels = torch.tensor([CATALOG_DECODER_FAMILIES.index(row["semantic"]["family"]) for row in train_rows], dtype=torch.long)
    model = CatalogRuleIRDecoder(hidden_dim=96, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01)
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), 16):
            batch_index = order[start:start + 16]
            batch_features = features[batch_index].to(device)
            batch_labels = labels[batch_index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_index)
        model.eval()
        with torch.inference_mode():
            prediction = model(features.to(device)).argmax(dim=-1).cpu()
        accuracy = float(prediction.eq(labels).float().mean())
        history.append({"epoch": epoch, "loss": round(total_loss / max(len(features), 1), 6), "train_accuracy": round(accuracy, 6)})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "normalisation_mean": mean.tolist(),
        "normalisation_std": std.tolist(),
        "novelty_threshold": _novelty_threshold(features),
        "train_accuracy": best_accuracy,
        "history": history,
    }


@torch.inference_mode()
def _evaluate(
    model: CatalogRuleIRDecoder,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_mean: list[float],
    train_std: list[float],
    novelty_threshold: float,
    *,
    device: torch.device,
    split: str,
    abstain_threshold: float = ABSTAIN_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> dict[str, Any]:
    train_features = (_features(train_rows) - torch.tensor(train_mean)) / torch.tensor(train_std)
    raw = _features(test_rows)
    features = (raw - torch.tensor(train_mean)) / torch.tensor(train_std)
    decoded = model.decode(
        features.to(device),
        abstain_threshold=abstain_threshold,
        margin_threshold=margin_threshold,
    )
    distances = torch.cdist(features, train_features).min(dim=1).values.tolist() if len(train_features) else [float("inf")] * len(test_rows)
    predictions: list[dict[str, Any]] = []
    for row, output, distance in zip(test_rows, decoded, distances):
        novelty = float(distance) > novelty_threshold
        prediction = dict(output)
        if novelty:
            prediction["family"] = None
            prediction["rule_ir"] = None
            prediction["abstained"] = True
            prediction["abstain_reason"] = "novel_surface_distance"
        expected = row["semantic"]["family"]
        predicted = prediction.get("family")
        exit_found = predicted == expected and prediction.get("rule_ir") is not None
        false_positive = predicted is not None and predicted != expected
        predictions.append({
            "sample_id": row["sample_id"],
            "expected_family": expected,
            "predicted_family": predicted,
            "candidate_family": prediction.get("candidate_family"),
            "confidence": prediction.get("confidence", 0.0),
            "margin": prediction.get("margin", 0.0),
            "distance": round(float(distance), 6),
            "novel_surface": novelty,
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
        "abstain_threshold": abstain_threshold,
        "margin_threshold": margin_threshold,
        "predictions": predictions,
    }


def _split(rows: list[dict[str, Any]]) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    train_source_a = [row for row in rows if row["source_id"].endswith("-a")]
    test_source_b = [row for row in rows if row["source_id"].endswith("-b")]
    transfer_train_families = {"xss", "injection", "access_control"}
    unseen_train_families = {"xss", "injection"}
    return {
        "source_split_same_family": (train_source_a, test_source_b),
        "family_holdout_structural_transfer": (
            [row for row in train_source_a if row["semantic"]["family"] in transfer_train_families],
            [row for row in test_source_b if row["semantic"]["family"] in {"logic", "url_redirect"}],
        ),
        "family_holdout_unseen_surface": (
            [row for row in train_source_a if row["semantic"]["family"] in unseen_train_families],
            [row for row in test_source_b if row["semantic"]["family"] in {"access_control", "logic", "url_redirect"}],
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-03 Catalog Rule IR Decoder",
        "",
        "模型只接收安全 probe 和受限响应形状，不接收 family/source/evaluator 标签。`exit_found` 是 Rule IR 家族候选与靶场语义一致，不是漏洞 evaluator 确认。",
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
        "解释：族外结构相同但语义不可区分时，模型的误报会暴露数据/可观测性不足；完全未见的 surface 应 abstain。该模型是小型基线，不等于 GPT/MoE。",
        f"运行点：confidence ≥ {report['model']['abstain_threshold']:.2f} 且 top-2 margin ≥ {report['model']['margin_threshold']:.2f}；novelty gate 独立生效。",
        "边界：本轮是同一 in-repo 本地 ASGI adapter 内的来源/族/表面隔离，不把它宣称成独立第三方靶场泛化。",
        "",
        f"原始 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"PG-03 catalog not found: {CATALOG_PATH}")
    rows = flatten_catalog(load_catalog(CATALOG_PATH))
    _assert_visible_projection(rows)
    splits = _split(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluations: list[dict[str, Any]] = []
    training_reports: dict[str, Any] = {}
    checkpoint_paths: list[str] = []
    for index, (split_name, (train_rows, test_rows)) in enumerate(splits.items()):
        model, training = _train_model(train_rows, seed=SEED + index, device=device)
        result = _evaluate(
            model,
            train_rows,
            test_rows,
            training["normalisation_mean"],
            training["normalisation_std"],
            training["novelty_threshold"],
            device=device,
            split=split_name,
        )
        training_reports[split_name] = {
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "train_families": sorted({row["semantic"]["family"] for row in train_rows}),
            "test_families": sorted({row["semantic"]["family"] for row in test_rows}),
            "train_accuracy": round(float(training["train_accuracy"]), 6),
            "novelty_threshold": training["novelty_threshold"],
            "history_tail": training["history"][-5:],
        }
        evaluations.append(result)
        checkpoint_dir = OUTPUT_DIR / split_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "decoder.pt"
        torch.save({
            "schema_version": CATALOG_DECODER_SCHEMA,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "normalisation_mean": training["normalisation_mean"],
            "normalisation_std": training["normalisation_std"],
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "novelty_threshold": training["novelty_threshold"],
            "seed": SEED + index,
            "device_at_training": str(device),
        }, checkpoint_path)
        checkpoint_paths.append(str(checkpoint_path.relative_to(ROOT)))

    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg03-rule-ir-decoder-report-v1",
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": load_catalog(CATALOG_PATH)["catalog_sha256"],
            "sample_count": len(rows),
        },
        "model": {
            "class": "CatalogRuleIRDecoder",
            "parameters": sum(parameter.numel() for parameter in CatalogRuleIRDecoder().parameters()),
            "feature_dim": FEATURE_DIM,
            "families": list(CATALOG_DECODER_FAMILIES),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "margin_threshold": MARGIN_THRESHOLD,
            "novelty_gate": True,
            "threshold_rationale": "exploratory confidence floor above 1/5 random baseline; margin and novelty gates remain mandatory",
        },
        "target_scope": {
            "base_url": "http://127.0.0.1:3100",
            "independent_target_implementation": False,
            "holdout_kind": "source, family and observable-surface isolation inside the in-repo local adapter",
            "not_claimed": "third-party or separately implemented target generalization",
        },
        "training": training_reports,
        "evaluations": evaluations,
        "checkpoints": checkpoint_paths,
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "visible_projection_labels": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": "research/pg03_rule_ir_decoder_protocol_v1.json",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "model": report["model"],
        "training": report["training"],
        "evaluations": report["evaluations"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
