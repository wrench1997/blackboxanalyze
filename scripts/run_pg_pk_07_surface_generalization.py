"""Run PG-PK-07 on several unseen, inert response surfaces."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_ROOT = ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.cross_app_surface_fixture import (  # noqa: E402
    SURFACE_FIXTURE_BASE_URL,
    SURFACE_FIXTURE_ORACLE,
    SurfaceFixtureCollector,
    default_surface_fixture_specs,
    make_surface_fixture_server,
    surface_fixture_source_sha256,
)
from app.catalog_rule_decoder import catalog_feature_vector  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.oracle_revalidation import revalidate_positive_pair  # noqa: E402
from app.rule_ir_decoder import DECODER_FAMILIES  # noqa: E402
from app.surface_discriminator import SurfaceDiscriminator  # noqa: E402
from app.surface_role_discriminator import SurfaceRoleDiscriminator, surface_shape_feature_vector  # noqa: E402
from app.surface_role_gate import assess_surface_role_gate  # noqa: E402
from run_pg_pk_06_positive_oracle import _load_model, _predict, _training_reference  # noqa: E402


PROTOCOL_ID = "pg-pk-07-surface-generalization-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_07_surface_generalization_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_07_surface_generalization_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_07_surface_generalization_protocol_v1.json"
SURFACE_CHECKPOINT = ROOT / "artifacts" / "surface-discriminator-loop-12-20260931" / "surface_discriminator.pt"
SURFACE_ROLE_CHECKPOINT = ROOT / "artifacts" / "surface-role-discriminator-pg-pk-08" / "surface_role_discriminator.pt"


def _surface_discriminator_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not SURFACE_CHECKPOINT.exists():
        return []
    checkpoint = torch.load(SURFACE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != 256:
        raise ValueError("surface discriminator checkpoint feature dimension mismatch")
    model = SurfaceDiscriminator().eval()
    model.load_state_dict(checkpoint["model_state"])
    raw = torch.tensor([catalog_feature_vector(row) for row in records], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    features = (raw - mean) / std
    with torch.inference_mode():
        probabilities = torch.softmax(model(features), dim=-1).tolist()
    threshold = float(checkpoint.get("abstain_threshold", 0.888045))
    output: list[dict[str, Any]] = []
    for record, values in zip(records, probabilities):
        index = max(range(len(values)), key=lambda item: values[item])
        probability = float(values[index])
        output.append({
            "sample_id": record["sample_id"],
            "candidate_family": DECODER_FAMILIES[index],
            "probability": round(probability, 6),
            "abstained": probability < threshold,
            "checkpoint_threshold": threshold,
        })
    return output


def _surface_role_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not SURFACE_ROLE_CHECKPOINT.exists():
        return []
    checkpoint = torch.load(SURFACE_ROLE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SurfaceRoleDiscriminator().eval()
    model.load_state_dict(checkpoint["model_state"])
    raw = torch.tensor([surface_shape_feature_vector(row) for row in records], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    decoded = model.decode((raw - mean) / std, abstain_threshold=float(checkpoint.get("abstain_threshold", 0.80718)))
    return [dict(row, sample_id=record["sample_id"]) for record, row in zip(records, decoded)]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-07 多表面反事实与族外泛化",
        "",
        "本轮在未见过的本地 fixture 上把同一 inert marker 放到 HTML 属性、HTML 文本、JSON、响应头和空白控制五类表面；只有属性表面属于本实验的正向 oracle。",
        "",
        f"样本：{report['sample_count']}；严格 OOD abstain：{report['strict_ood_abstain_count']}；模型-only 接受：{report['model_only_accepted_count']}；非属性表面模型-only 误报候选：{report['non_attribute_model_only_candidate_count']}；oracle 复核样本：{report['oracle_revalidated_sample_count']}。",
        f"辅助 surface discriminator 在非属性样本上拒绝：{report['non_attribute_secondary_surface_rejection_count']}；它只作诊断，不具备正向放行权。",
        f"同一 discriminator 对属性正例 abstain：{report['positive_secondary_surface_abstain_count']}；因此当前结论是‘能筛掉一部分表面捷径，但尚不能做正向 gate’，需要后续做跨应用校准。",
        f"PG-PK-08 surface-role head 属性正例接受：{report['surface_role_positive_accept_count']}/2；非属性误接受：{report['surface_role_non_attribute_accept_count']}；该 head 仍只作诊断。",
        f"surface-role promotion gate：`{report['surface_role_gate']['status']}`；原因：{', '.join(report['surface_role_gate'].get('reasons', [])) or 'none'}。",
        "",
        "| surface role | samples | model-only accepts | oracle signal rows | revalidated pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, value in report["surface_summary"].items():
        lines.append(
            f"| `{role}` | {value['sample_count']} | {value['model_only_accepted_count']} | {value['oracle_signal_count']} | {value['oracle_revalidated_pair_count']} |"
        )
    lines.extend([
        "",
        f"单 fixture 晋升试探：`{report['memory_promotion_probe']['status']}`；原因：{', '.join(report['memory_promotion_probe']['reasons']) or 'none'}。",
        "属性 oracle 不是把 OOD 阈值放宽，而是要求同一 surface 的 plain/url-percent pair、模型族一致、源码哈希、证据哈希和属性信号全部通过；文本/JSON/响应头即使回显 marker 也保持负例。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    server = make_surface_fixture_server()
    thread = threading.Thread(target=server.serve_forever, name="sift-pg-pk-07-fixture", daemon=True)
    thread.start()
    try:
        source_hash = surface_fixture_source_sha256()
        records = asyncio.run(
            SurfaceFixtureCollector(target_instance_id=f"surface-pid-{threading.get_ident()}", source_hash=source_hash).collect_many(
                default_surface_fixture_specs()
            )
        )
        from app.catalog_rule_decoder import catalog_visible_trace  # noqa: E402

        visible = json.dumps([catalog_visible_trace(row) for row in records], ensure_ascii=False).casefold()
        forbidden = ("evaluator", "challenge", "family", "source_id", "rule_ir", "marker_in_attribute", "expected_oracle")
        if any(token in visible for token in forbidden):
            raise RuntimeError("surface fixture visible projection leaked evaluator/oracle labels")
        model, checkpoint = _load_model()
        reference, ood_fit = _training_reference(checkpoint)
        predictions = _predict(model, checkpoint, records, ood_fit=ood_fit, ood_reference=reference)
        secondary_predictions = _surface_discriminator_predictions(records)
        role_predictions = _surface_role_predictions(records)
        for prediction, secondary in zip(predictions, secondary_predictions):
            prediction["surface_discriminator_family"] = secondary["candidate_family"]
            prediction["surface_discriminator_probability"] = secondary["probability"]
            prediction["surface_discriminator_abstained"] = secondary["abstained"]
        for prediction, role_prediction in zip(predictions, role_predictions):
            prediction["surface_role_candidate"] = role_prediction["candidate_role"]
            prediction["surface_role_confidence"] = role_prediction["confidence"]
            prediction["surface_role_abstained"] = role_prediction["abstained"]
        validation_precision = None
        training_acceptance = None
        role_training_report = SURFACE_ROLE_CHECKPOINT.parent / "report.json"
        if role_training_report.exists():
            try:
                training_report = json.loads(role_training_report.read_text(encoding="utf-8"))
                validation_precision = float(training_report.get("calibration", {}).get("precision"))
                training_acceptance = bool(training_report.get("acceptance", {}).get("passed", False))
            except (OSError, TypeError, ValueError):
                validation_precision = None
        if role_predictions:
            surface_role_gate = assess_surface_role_gate(
                role_predictions,
                [str(record["semantic"]["surface_role"]) for record in records],
                validation_precision=validation_precision,
                training_acceptance=training_acceptance,
            )
        else:
            surface_role_gate = {
                "status": "diagnostic_only",
                "enabled": False,
                "reasons": ["checkpoint_missing"],
                "metrics": {"positive_recall": 0.0, "non_attribute_accept_rate": 0.0, "validation_precision": validation_precision, "training_acceptance": training_acceptance},
            }
        by_id = {row["sample_id"]: row for row in predictions}

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("pair"):
                grouped[str(record["pair"]["pair_id"])].append(record)
        pair_results: list[dict[str, Any]] = []
        for pair_id, pair_records in sorted(grouped.items()):
            role = str((pair_records[0].get("pair") or {}).get("surface_role", ""))
            candidate_records = [
                dict(record, candidate_family=by_id[record["sample_id"]]["candidate_family"])
                for record in pair_records
            ]
            result = revalidate_positive_pair(
                candidate_records,
                expected_family="xss",
                oracle_name=SURFACE_FIXTURE_ORACLE,
                authorized_source_hash=source_hash,
                required_surface_role=role,
                required_sink_kind="html_attribute",
            )
            result["pair_id"] = pair_id
            result["surface_role"] = role
            pair_results.append(result)
            if result["accepted"]:
                for record in pair_records:
                    by_id[record["sample_id"]]["oracle_revalidated"] = True

        surface_summary: dict[str, dict[str, int]] = defaultdict(lambda: {
            "sample_count": 0,
            "model_only_accepted_count": 0,
            "oracle_signal_count": 0,
            "oracle_revalidated_pair_count": 0,
        })
        for record, prediction in zip(records, predictions):
            role = str(record["semantic"]["surface_role"])
            value = surface_summary[role]
            value["sample_count"] += 1
            value["model_only_accepted_count"] += int(prediction["model_only_accepted"])
            value["oracle_signal_count"] += int(bool(record["rule_ir_result"]))
        for result in pair_results:
            surface_summary[result["surface_role"]]["oracle_revalidated_pair_count"] += int(result["accepted"])

        positive_result = next((result for result in pair_results if result.get("surface_role") == "reflected_attribute"), None)
        ledger: list[dict[str, Any]] = []
        if positive_result and positive_result.get("accepted"):
            hashes = positive_result.get("evidence_hashes") or []
            for index, seed in enumerate((20260871, 20260879)):
                ledger.append({
                    "dataset_id": "surface-fixture-pg-pk-07",
                    "sampling_seed": seed,
                    "target_instance_id": f"surface-pid-{threading.get_ident()}",
                    "rule_key": "xss::reflected_attribute",
                    "accepted": True,
                    "oracle_revalidated": True,
                    "false_positive": False,
                    "evidence_hash": hashes[index % len(hashes)] if hashes else "",
                    "local_only": True,
                })
        promotion_probe = assess_memory_promotion("xss::reflected_attribute", ledger)

        non_attribute_candidates = sum(
            int(row["model_only_accepted"] and record["semantic"]["surface_role"] != "reflected_attribute")
            for record, row in zip(records, predictions)
        )
        non_attribute_secondary_rejections = sum(
            int(
                record["semantic"]["surface_role"] != "reflected_attribute"
                and (
                    row.get("surface_discriminator_family") != "xss"
                    or bool(row.get("surface_discriminator_abstained", True))
                )
            )
            for record, row in zip(records, predictions)
        )
        positive_secondary_abstains = sum(
            int(
                record["semantic"]["surface_role"] == "reflected_attribute"
                and bool(row.get("surface_discriminator_abstained", True))
            )
            for record, row in zip(records, predictions)
        )
        role_positive_accepts = sum(
            int(
                record["semantic"]["surface_role"] == "reflected_attribute"
                and row.get("surface_role_candidate") == "reflected_attribute"
                and not bool(row.get("surface_role_abstained", True))
            )
            for record, row in zip(records, predictions)
        )
        role_non_attribute_accepts = sum(
            int(
                record["semantic"]["surface_role"] != "reflected_attribute"
                and row.get("surface_role_candidate") == "reflected_attribute"
                and not bool(row.get("surface_role_abstained", True))
            )
            for record, row in zip(records, predictions)
        )
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pg-pk-07-surface-generalization-report-v1",
            "target": {
                "base_url": SURFACE_FIXTURE_BASE_URL,
                "target_instance_id": f"surface-pid-{threading.get_ident()}",
                "fixture_source_sha256": source_hash,
                "external_network": False,
                "loopback_only": True,
                "fresh_target": True,
            },
            "training_boundary": {
                "training_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "training_target": "Pikachu only",
                "surface_fixture_seen_during_training": False,
                "visible_projection_labels_hidden": True,
            },
            "sample_count": len(records),
            "strict_ood_abstain_count": sum(row["ood"] for row in predictions),
            "model_only_accepted_count": sum(row["model_only_accepted"] for row in predictions),
            "non_attribute_model_only_candidate_count": non_attribute_candidates,
            "non_attribute_secondary_surface_rejection_count": non_attribute_secondary_rejections,
            "positive_secondary_surface_abstain_count": positive_secondary_abstains,
            "surface_role_positive_accept_count": role_positive_accepts,
            "surface_role_non_attribute_accept_count": role_non_attribute_accepts,
            "surface_role_positive_recall": role_positive_accepts / 2 if records else 0.0,
            "oracle_revalidated_sample_count": sum(row["oracle_revalidated"] for row in predictions),
            "oracle_revalidated_pair_count": sum(result["accepted"] for result in pair_results),
            "surface_summary": dict(sorted(surface_summary.items())),
            "pair_results": pair_results,
            "predictions": predictions,
            "memory_promotion_probe": promotion_probe,
            "ood_fit": ood_fit,
            "surface_discriminator": {
                "checkpoint": str(SURFACE_CHECKPOINT.relative_to(ROOT)) if SURFACE_CHECKPOINT.exists() else None,
                "used_as_positive_gate": False,
                "non_attribute_rejection_count": non_attribute_secondary_rejections,
                "positive_abstain_count": positive_secondary_abstains,
            },
            "surface_role_discriminator": {
                "checkpoint": str(SURFACE_ROLE_CHECKPOINT.relative_to(ROOT)) if SURFACE_ROLE_CHECKPOINT.exists() else None,
                "used_as_positive_gate": False,
                "positive_accept_count": role_positive_accepts,
                "non_attribute_accept_count": role_non_attribute_accepts,
                "positive_recall": role_positive_accepts / 2 if records else 0.0,
            },
            "surface_role_gate": surface_role_gate,
            "safety": {
                "raw_body_stored": False,
                "evaluator_state_visible": False,
                "external_network": False,
                "script_execution": False,
                "database_write": False,
                "mutations": False,
            },
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "sample_count": report["sample_count"],
            "strict_ood_abstain_count": report["strict_ood_abstain_count"],
            "model_only_accepted_count": report["model_only_accepted_count"],
            "non_attribute_model_only_candidate_count": report["non_attribute_model_only_candidate_count"],
            "non_attribute_secondary_surface_rejection_count": report["non_attribute_secondary_surface_rejection_count"],
            "positive_secondary_surface_abstain_count": report["positive_secondary_surface_abstain_count"],
            "surface_role_positive_accept_count": report["surface_role_positive_accept_count"],
            "surface_role_non_attribute_accept_count": report["surface_role_non_attribute_accept_count"],
            "surface_role_gate_status": surface_role_gate["status"],
            "oracle_revalidated_sample_count": report["oracle_revalidated_sample_count"],
            "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"],
            "memory_promotion_status": promotion_probe["status"],
            "report": report["report_path"],
            "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
        }, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
