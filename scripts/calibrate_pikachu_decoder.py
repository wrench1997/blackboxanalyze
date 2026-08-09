"""Calibrate the frozen Rule IR decoder against local negative controls.

This does not retrain the classifier.  It fits a scalar temperature on a
held-out calibration subset, then compares a model-only acceptance gate with a
family-specific bounded-oracle gate on unseen encoding variants.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import (  # noqa: E402
    CATALOG_DECODER_FAMILIES,
    CatalogRuleIRDecoderV2,
    catalog_feature_vector,
)
from app.confidence_calibration import (  # noqa: E402
    accept_with_evidence,
    binary_brier_score,
    evidence_fused_confidence,
    expected_calibration_error,
    family_oracle_support,
    fit_temperature,
    multiclass_nll,
    temperature_scale,
)
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "pg-pk-04-confidence-calibration-v1"
CATALOG_PATH = ROOT / "research" / "pikachu_counterfactual_catalog_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
REPORT_PATH = ROOT / "research" / "pikachu_confidence_calibration_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_confidence_calibration_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pikachu_confidence_calibration_protocol_v1.json"
MODEL_ONLY_THRESHOLD = 0.45
MODEL_ONLY_MARGIN = 0.10
EVIDENCE_THRESHOLD = 0.70
ORACLE_SUPPORT_THRESHOLD = 0.50


def _load_model() -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError("decoder checkpoint feature dimension does not match runtime")
    state = checkpoint["model_state"]
    branch_dim = int(state["surface_tower.0.weight"].shape[0])
    embedding_dim = int(state["projector.0.weight"].shape[0])
    model = CatalogRuleIRDecoderV2(branch_dim=branch_dim, embedding_dim=embedding_dim, dropout=0.0)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _model_outputs(model: CatalogRuleIRDecoderV2, rows: list[dict[str, Any]], checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    raw = torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32)
    features = (raw - mean) / std.clamp_min(1e-4)
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1).tolist()
    outputs: list[dict[str, Any]] = []
    for row, probability_row in zip(rows, probabilities):
        order = sorted(range(len(probability_row)), key=lambda index: probability_row[index], reverse=True)
        candidate_index = order[0]
        second_index = order[1]
        outputs.append({
            "row": row,
            "probabilities": probability_row,
            "candidate_family": CATALOG_DECODER_FAMILIES[candidate_index],
            "candidate_index": candidate_index,
            "model_probability": float(probability_row[candidate_index]),
            "margin": float(probability_row[candidate_index] - probability_row[second_index]),
        })
    return outputs


def _is_negative(row: dict[str, Any]) -> bool:
    return bool(row.get("counterfactual"))


def _is_plain_or_url_negative(row: dict[str, Any], variant: str) -> bool:
    return str(row.get("probe_artifact", {}).get("encoding", "")).endswith(variant)


def _gate_metrics(outputs: list[dict[str, Any]], *, probabilities: list[float], gate: str) -> dict[str, Any]:
    accepted_rows: list[dict[str, Any]] = []
    for output, probability in zip(outputs, probabilities):
        row = output["row"]
        support = family_oracle_support(output["candidate_family"], dict(row.get("oracle_projection") or {}))
        if gate == "model_only":
            accepted = probability >= MODEL_ONLY_THRESHOLD and output["margin"] >= MODEL_ONLY_MARGIN
            score = probability
        elif gate == "family_oracle":
            score = evidence_fused_confidence(probability, support)
            accepted = accept_with_evidence(
                calibrated_confidence=score,
                oracle_support=support,
                confidence_threshold=EVIDENCE_THRESHOLD,
                evidence_threshold=ORACLE_SUPPORT_THRESHOLD,
            )
        else:
            raise ValueError(f"unknown calibration gate {gate}")
        expected_signal = bool(row.get("rule_ir_result"))
        accepted_rows.append({
            "sample_id": row["sample_id"],
            "family": row["semantic"]["family"],
            "surface": row["semantic"]["surface"],
            "counterfactual": _is_negative(row),
            "candidate_family": output["candidate_family"],
            "model_probability": round(float(probability), 6),
            "oracle_support": round(float(support), 6),
            "score": round(float(score), 6),
            "accepted": bool(accepted),
            "expected_signal": expected_signal,
            "false_positive": bool(accepted and not expected_signal),
            "exit_found": bool(accepted and expected_signal),
            "abstained": not bool(accepted),
        })
    total = len(accepted_rows)
    return {
        "gate": gate,
        "total": total,
        "accepted": sum(row["accepted"] for row in accepted_rows),
        "exit_found_rate": sum(row["exit_found"] for row in accepted_rows) / total if total else 0.0,
        "false_positive_rate": sum(row["false_positive"] for row in accepted_rows) / total if total else 0.0,
        "abstain_rate": sum(row["abstained"] for row in accepted_rows) / total if total else 0.0,
        "predictions": accepted_rows,
    }


def _class_calibration(
    outputs: list[dict[str, Any]],
    raw_probabilities: list[list[float]],
    scaled_probabilities: list[list[float]],
) -> dict[str, Any]:
    labels = [CATALOG_DECODER_FAMILIES.index(output["row"]["semantic"]["family"]) for output in outputs]
    raw_confidences = [output["model_probability"] for output in outputs]
    scaled_confidences = [max(row) for row in scaled_probabilities]
    correctness = [output["candidate_family"] == output["row"]["semantic"]["family"] for output in outputs]
    return {
        "sample_count": len(outputs),
        "raw_ece": expected_calibration_error(raw_confidences, correctness),
        "scaled_ece": expected_calibration_error(scaled_confidences, correctness),
        "raw_brier": binary_brier_score(raw_confidences, correctness),
        "scaled_brier": binary_brier_score(scaled_confidences, correctness),
        "raw_accuracy": sum(correctness) / len(correctness) if correctness else 0.0,
        "raw_nll": multiclass_nll(raw_probabilities, labels),
        "scaled_nll": multiclass_nll(scaled_probabilities, labels),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-04 置信度校准",
        "",
        "冻结联合留出 checkpoint，只在 calibration subset 拟合一个温度，再把未见编码变体和反事实控制留作测试。Rule IR 只有在模型置信度与对应族的 bounded oracle 同时支持时才发射。",
        "",
        f"温度：`{report['temperature']['temperature']:.3f}`；class ECE {report['class_calibration']['raw_ece']:.3f} → {report['class_calibration']['scaled_ece']:.3f}；Brier {report['class_calibration']['raw_brier']:.3f} → {report['class_calibration']['scaled_brier']:.3f}。",
        "",
        "| 测试集 | gate | exit | false positive | abstain |",
        "|---|---|---:|---:|---:|",
    ]
    for suite, values in report["gates"].items():
        for gate, row in values.items():
            lines.append(f"| `{suite}` | `{gate}` | {row['exit_found_rate']:.2f} | {row['false_positive_rate']:.2f} | {row['abstain_rate']:.2f} |")
    lines.extend([
        "",
        "family_oracle gate 的 abstain 是有意的：没有族特异 bounded evidence 时，即使分类器猜中 family，也不发射 Rule IR。",
        "该实验没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。",
        "",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    rows = flatten_catalog(load_catalog(CATALOG_PATH))
    model, checkpoint = _load_model()
    positive = [row for row in rows if not _is_negative(row)]
    negative = [row for row in rows if _is_negative(row)]
    calibration_rows = [
        row for row in positive
        if str(row.get("pair", {}).get("variant")) in {"plain", "url_percent"}
    ] + [row for row in negative if _is_plain_or_url_negative(row, "plain")]
    test_rows = [
        row for row in positive
        if str(row.get("pair", {}).get("variant")) in {"html_entity", "double_html_entity"}
    ] + [row for row in negative if _is_plain_or_url_negative(row, "url_percent")]
    if not calibration_rows or not test_rows:
        raise RuntimeError("calibration split is empty")
    calibration_outputs = _model_outputs(model, calibration_rows, checkpoint)
    test_outputs = _model_outputs(model, test_rows, checkpoint)
    calibration_probabilities = [output["probabilities"] for output in calibration_outputs]
    calibration_labels = [CATALOG_DECODER_FAMILIES.index(output["row"]["semantic"]["family"]) for output in calibration_outputs]
    temperature = fit_temperature(calibration_probabilities, calibration_labels)
    test_scaled = temperature_scale([output["probabilities"] for output in test_outputs], temperature["temperature"])
    test_scaled_confidence = [max(row) for row in test_scaled]
    all_raw = [output["probabilities"] for output in calibration_outputs]
    all_scaled = temperature_scale(all_raw, temperature["temperature"])
    class_calibration = _class_calibration(calibration_outputs, all_raw, all_scaled)
    gates = {
        "encoding_holdout_plus_negative": {
            "model_only": _gate_metrics(test_outputs, probabilities=test_scaled_confidence, gate="model_only"),
            "family_oracle": _gate_metrics(test_outputs, probabilities=test_scaled_confidence, gate="family_oracle"),
        },
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-confidence-calibration-report-v1",
        "catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
        "catalog_sha256": load_catalog(CATALOG_PATH)["catalog_sha256"],
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "split": {
            "calibration_count": len(calibration_rows),
            "test_count": len(test_rows),
            "calibration_positive_count": sum(not _is_negative(row) for row in calibration_rows),
            "calibration_negative_count": sum(_is_negative(row) for row in calibration_rows),
            "test_positive_count": sum(not _is_negative(row) for row in test_rows),
            "test_negative_count": sum(_is_negative(row) for row in test_rows),
            "positive_train_variants": ["plain", "url_percent"],
            "positive_test_variants": ["html_entity", "double_html_entity"],
            "negative_train_variant": "counterfactual_marker_substitution_plain",
            "negative_test_variant": "counterfactual_marker_substitution_url_percent",
        },
        "temperature": temperature,
        "class_calibration": class_calibration,
        "gates": gates,
        "thresholds": {
            "model_only_probability": MODEL_ONLY_THRESHOLD,
            "model_only_margin": MODEL_ONLY_MARGIN,
            "evidence_fused_confidence": EVIDENCE_THRESHOLD,
            "oracle_support": ORACLE_SUPPORT_THRESHOLD,
        },
        "family_specific_oracle": True,
        "evaluator_confirmation_count": 0,
        "safety": {
            "local_only": True,
            "external_network": False,
            "script_execution": False,
            "database_touched": False,
            "raw_body_stored": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "temperature": report["temperature"],
        "class_calibration": report["class_calibration"],
        "gates": {suite: {gate: {key: value for key, value in row.items() if key != "predictions"} for gate, row in values.items()} for suite, values in gates.items()},
        "report": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
