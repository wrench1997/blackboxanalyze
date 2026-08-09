"""Evaluate the frozen decoder on a second local application, fail-closed."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.catalog_rule_decoder import CATALOG_DECODER_FAMILIES, CatalogRuleIRDecoderV2, catalog_feature_vector, catalog_visible_trace  # noqa: E402
from app.confidence_calibration import accept_with_evidence, evidence_fused_confidence, family_oracle_support  # noqa: E402
from app.juice_shop_adapter import DEFAULT_BASE_URL, PINNED_IMAGE, TARGET_CONTAINER  # noqa: E402
from app.juice_shop_shadow_collector import JuiceShopShadowCollector, default_juice_shop_shadow_specs  # noqa: E402
from app.ood_gate import fit_ood_reference, nearest_reference_distances, ood_flags  # noqa: E402
from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402
from app.rule_ir_decoder import FEATURE_DIM  # noqa: E402


PROTOCOL_ID = "pg-pk-05-cross-app-shadow-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
REPORT_PATH = ROOT / "research" / "juice_shop_cross_app_shadow_v1.json"
MARKDOWN_PATH = ROOT / "research" / "juice_shop_cross_app_shadow_v1.md"
PROTOCOL_PATH = ROOT / "research" / "juice_shop_cross_app_shadow_protocol_v1.json"


def _docker_instance_id() -> str:
    try:
        return subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", TARGET_CONTAINER],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unattested"


def _load_model() -> tuple[CatalogRuleIRDecoderV2, dict[str, Any]]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        raise ValueError("decoder checkpoint feature dimension mismatch")
    state = checkpoint["model_state"]
    model = CatalogRuleIRDecoderV2(
        branch_dim=int(state["surface_tower.0.weight"].shape[0]),
        embedding_dim=int(state["projector.0.weight"].shape[0]),
        dropout=0.0,
    )
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def _predict(
    model: CatalogRuleIRDecoderV2,
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    ood_fit: dict[str, Any],
    ood_reference: torch.Tensor,
) -> list[dict[str, Any]]:
    raw = torch.tensor([catalog_feature_vector(row) for row in rows], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (raw - mean) / std
    ood_distances = nearest_reference_distances(features, ood_reference)
    ood = ood_flags(ood_distances, ood_fit)
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1).tolist()
    predictions: list[dict[str, Any]] = []
    for row, values in zip(rows, probabilities):
        ordered = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
        candidate_index = ordered[0]
        second_index = ordered[1]
        family = CATALOG_DECODER_FAMILIES[candidate_index]
        model_probability = float(values[candidate_index])
        margin = float(values[candidate_index] - values[second_index])
        support = family_oracle_support(family, dict(row.get("oracle_projection") or {}))
        fused = evidence_fused_confidence(model_probability, support)
        predictions.append({
            "sample_id": row["sample_id"],
            "surface": row["semantic"]["surface"],
            "candidate_family": family,
            "model_probability": round(model_probability, 6),
            "margin": round(margin, 6),
            "oracle_support": round(float(support), 6),
            "ood_distance": round(float(ood_distances[len(predictions)]), 6),
            "ood": bool(ood[len(predictions)]),
            "evidence_fused_confidence": round(float(fused), 6),
            "model_only_accepted": bool(model_probability >= 0.45 and margin >= 0.10),
            "oracle_gate_accepted": accept_with_evidence(
                calibrated_confidence=fused,
                oracle_support=support,
                confidence_threshold=0.70,
                evidence_threshold=0.50,
            ) and not bool(ood[len(predictions)]),
            "expected_action": "abstain_unsupported_surface",
            "rule_ir_emitted": False,
        })
    return predictions


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-05 Juice Shop 跨应用 shadow",
        "",
        "这是与 Pikachu 完全分开的本地应用表面留出：只发送 5 个 allow-listed GET canary，模型看不到应用/路径标签，且 Juice Shop 轨道没有已授权的族特异 oracle，因此正确行为是 abstain。",
        "",
        f"样本：{report['sample_count']}；模型-only 接受：{report['model_only_accepted_count']}；oracle gate 接受：{report['oracle_gate_accepted_count']}；oracle gate abstain：{report['oracle_gate_abstain_rate']:.2%}。",
        "",
        "| surface | candidate family | model p | OOD | oracle support | model-only | oracle gate |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for row in report["predictions"]:
        lines.append(
            f"| `{row['surface']}` | `{row['candidate_family']}` | {row['model_probability']:.3f} | {'yes' if row['ood'] else 'no'} | {row['oracle_support']:.3f} | "
            f"{'accept' if row['model_only_accepted'] else 'abstain'} | {'accept' if row['oracle_gate_accepted'] else 'abstain'} |"
        )
    lines.extend([
        "",
        "model-only 的接受不是漏洞判断；本轮没有族特异 oracle，因此任何 Rule IR 发射都属于不合格猜测。",
        "没有访问 `/api/Challenges`、`/snippets` 或其他 evaluator 路径；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    instance_id = _docker_instance_id()
    records = __import__("asyncio").run(
        JuiceShopShadowCollector(target_instance_id=instance_id).collect_many(default_juice_shop_shadow_specs())
    )
    visible = json.dumps([catalog_visible_trace(row) for row in records], ensure_ascii=False).casefold()
    forbidden = ("evaluator", "challenge", "family", "source_id", "rule_ir")
    if any(token in visible for token in forbidden):
        raise RuntimeError("cross-app visible projection leaked evaluator or provenance tokens")
    model, checkpoint = _load_model()
    training_rows = flatten_catalog(load_catalog(ROOT / "research" / "pikachu_paired_catalog_v1.json"))
    training_rows = [
        row for row in training_rows
        if row["pair"]["variant"] in {"plain", "url_percent"}
        and row["pair"]["surface_role"] in {"reflected_get", "sqli_str", "sqli_search"}
    ]
    raw_reference = torch.tensor([catalog_feature_vector(row) for row in training_rows], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    ood_reference = (raw_reference - mean) / std
    ood_fit = fit_ood_reference(ood_reference)
    predictions = _predict(model, checkpoint, records, ood_fit=ood_fit, ood_reference=ood_reference)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-juice-shop-cross-app-shadow-report-v1",
        "target": {
            "base_url": DEFAULT_BASE_URL,
            "container": TARGET_CONTAINER,
            "target_instance_id": instance_id,
            "container_image": PINNED_IMAGE,
            "external_network": False,
            "loopback_only": True,
            "fresh_target": False,
        },
        "training_boundary": {
            "training_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "training_target": "Pikachu only",
            "cross_app_surface_seen_during_training": False,
            "visible_projection_labels_hidden": True,
        },
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "sample_count": len(records),
        "model_only_accepted_count": sum(row["model_only_accepted"] for row in predictions),
        "oracle_gate_accepted_count": sum(row["oracle_gate_accepted"] for row in predictions),
        "oracle_gate_abstain_rate": sum(not row["oracle_gate_accepted"] for row in predictions) / len(predictions) if predictions else 1.0,
        "ood_gate_abstain_count": sum(row["ood"] for row in predictions),
        "ood_gate_abstain_rate": sum(row["ood"] for row in predictions) / len(predictions) if predictions else 1.0,
        "unsupported_surface_count": sum(row["oracle_support"] < 0.50 for row in predictions),
        "ood_fit": ood_fit,
        "predictions": predictions,
        "safety": {
            "raw_body_stored": False,
            "evaluator_state_visible": False,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
        },
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "sample_count": report["sample_count"],
        "model_only_accepted_count": report["model_only_accepted_count"],
        "oracle_gate_accepted_count": report["oracle_gate_accepted_count"],
        "oracle_gate_abstain_rate": report["oracle_gate_abstain_rate"],
        "ood_gate_abstain_count": report["ood_gate_abstain_count"],
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
