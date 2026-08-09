"""Run the abstract, non-executing SQL differential track."""

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

from app.catalog_rule_decoder import catalog_visible_trace  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.sql_differential_fixture import (  # noqa: E402
    SQL_FIXTURE_BASE_URL,
    SQL_FIXTURE_ORACLE,
    SqlDifferentialCollector,
    default_sql_fixture_specs,
    make_sql_fixture_server,
    sql_fixture_source_sha256,
)
from app.sql_oracle_revalidation import revalidate_sql_pair  # noqa: E402
from app.sql_channel_decoder import SqlChannelDecoder, sql_channel_feature_vector  # noqa: E402
from run_pg_pk_06_positive_oracle import _load_model, _predict, _training_reference  # noqa: E402


PROTOCOL_ID = "pg-pk-09-sql-differential-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg-pk-02-pair-invariance" / "joint_holdout" / "pair_encoding_invariant" / "decoder.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_09_sql_differential_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_09_sql_differential_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_09_sql_differential_protocol_v1.json"
SQL_DECODER_CHECKPOINT = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09" / "sql_channel_decoder.pt"


def _sql_decoder_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not SQL_DECODER_CHECKPOINT.exists():
        return [{"candidate_family": "control", "confidence": 0.0, "abstained": True} for _ in records]
    checkpoint = torch.load(SQL_DECODER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SqlChannelDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    raw = torch.tensor([sql_channel_feature_vector(row) for row in records], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model.decode((raw - mean) / std, abstain_threshold=float(checkpoint.get("abstain_threshold", 0.80)))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-09 抽象 SQL differential",
        "",
        "本轮只发送白名单抽象 fragment class；服务端不执行 SQL、不访问数据库、不进行真实 sleep。错误、盲差分、行形状、超时和本地 side-channel 都由 AST/响应形状 oracle 给出 bounded evidence。",
        "",
        f"样本：{report['sample_count']}；严格 OOD abstain：{report['strict_ood_abstain_count']}；全局 decoder model-only 接受：{report['model_only_accepted_count']}；SQL channel decoder injection 接受：{report['sql_decoder_injection_accepted_count']}；SQL oracle 复核样本：{report['oracle_revalidated_sample_count']}（pair：{report['oracle_revalidated_pair_count']}）。",
        "",
        "| modality | pair count | revalidated |",
        "|---|---:|---:|",
    ]
    for modality, value in report["modality_summary"].items():
        lines.append(f"| `{modality}` | {value['pair_count']} | {value['revalidated_pair_count']} |")
    lines.extend([
        "",
        f"plain control revalidated：{report['plain_control_revalidated_count']}；单 fixture 长期记忆晋升：`{report['memory_promotion_probe']['status']}`。",
        "这里的 revalidation 证明的是抽象通道/Rule IR 出口，不是对真实 SQL 服务发起攻击；任何真实目标仍必须保持本地授权、严格 OOD 和 abstain。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    server = make_sql_fixture_server()
    thread = threading.Thread(target=server.serve_forever, name="sift-pg-pk-09-fixture", daemon=True)
    thread.start()
    try:
        source_hash = sql_fixture_source_sha256()
        records = asyncio.run(
            SqlDifferentialCollector(target_instance_id=f"sql-pid-{threading.get_ident()}", source_hash=source_hash).collect_many(
                default_sql_fixture_specs()
            )
        )
        visible = json.dumps([catalog_visible_trace(row) for row in records], ensure_ascii=False).casefold()
        forbidden = ("evaluator", "challenge", "family", "source_id", "rule_ir", "candidate_ast_sha256", "parameterized_ast")
        if any(token in visible for token in forbidden):
            raise RuntimeError("SQL visible projection leaked evaluator or AST labels")
        model, checkpoint = _load_model()
        reference, ood_fit = _training_reference(checkpoint)
        predictions = _predict(model, checkpoint, records, ood_fit=ood_fit, ood_reference=reference)
        sql_predictions = _sql_decoder_predictions(records)
        for prediction, sql_prediction in zip(predictions, sql_predictions):
            prediction["sql_decoder_candidate_family"] = sql_prediction["candidate_family"]
            prediction["sql_decoder_confidence"] = sql_prediction["confidence"]
            prediction["sql_decoder_abstained"] = sql_prediction["abstained"]
        by_id = {row["sample_id"]: row for row in predictions}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("pair"):
                grouped[str(record["pair"]["pair_id"])].append(record)
        pair_results: list[dict[str, Any]] = []
        for pair_id, pair_records in sorted(grouped.items()):
            candidate_records = [dict(record, candidate_family=by_id[record["sample_id"]]["sql_decoder_candidate_family"]) for record in pair_records]
            result = revalidate_sql_pair(
                candidate_records,
                authorized_source_hash=source_hash,
                oracle_name=SQL_FIXTURE_ORACLE,
            )
            result["pair_id"] = pair_id
            result["decoder"] = "sql_channel_decoder"
            pair_results.append(result)
            if result["accepted"]:
                for record in pair_records:
                    by_id[record["sample_id"]]["oracle_revalidated"] = True

        modality_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"pair_count": 0, "revalidated_pair_count": 0})
        for result in pair_results:
            for modality in result.get("modalities", []) or ["unknown"]:
                modality_summary[modality]["pair_count"] += 1
                modality_summary[modality]["revalidated_pair_count"] += int(result["accepted"])
        accepted_pairs = {str(result.get("pair_id", "")) for result in pair_results if result.get("accepted")}
        ledger = []
        for record in records:
            evidence = record.get("evidence") or {}
            reset = evidence.get("reset") or {}
            pair_id = str((record.get("pair") or {}).get("pair_id", ""))
            prediction = by_id[record["sample_id"]]
            model_candidate = bool(
                prediction.get("sql_decoder_candidate_family") == "injection"
                and not prediction.get("sql_decoder_abstained", True)
            )
            accepted = pair_id in accepted_pairs
            # Use the pair variant as a deterministic two-seed replay split;
            # this records both encodings without fabricating a third source.
            seed = 20260891 if str((record.get("pair") or {}).get("variant", "plain")) == "plain" else 20260897
            ledger.append({
                "dataset_id": "sql_differential_fixture_v1",
                "sampling_seed": seed,
                "target_instance_id": str(reset.get("target_instance_id", f"sql-pid-{threading.get_ident()}")),
                "rule_key": "injection::synthetic_sql_channel",
                "accepted": accepted,
                "oracle_revalidated": accepted,
                "false_positive": bool(model_candidate and not accepted),
                "evidence_hash": str(evidence.get("evidence_hash", "")),
                "source_hash": source_hash,
                "local_only": True,
            })
        promotion_probe = assess_memory_promotion("injection::synthetic_sql_channel", ledger)
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pg-pk-09-sql-differential-report-v1",
            "target": {"base_url": SQL_FIXTURE_BASE_URL, "target_instance_id": f"sql-pid-{threading.get_ident()}", "fixture_source_sha256": source_hash, "external_network": False, "loopback_only": True, "fresh_target": True},
            "training_boundary": {"training_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "sql_decoder_checkpoint": str(SQL_DECODER_CHECKPOINT.relative_to(ROOT)) if SQL_DECODER_CHECKPOINT.exists() else None, "training_target": "Pikachu only plus abstract local SQL fixture head", "fixture_seen_during_training": False, "visible_projection_labels_hidden": True},
            "sample_count": len(records),
            "strict_ood_abstain_count": sum(row["ood"] for row in predictions),
            "model_only_accepted_count": sum(row["model_only_accepted"] for row in predictions),
            "sql_decoder_injection_accepted_count": sum(int(row["sql_decoder_candidate_family"] == "injection" and not row["sql_decoder_abstained"]) for row in predictions),
            "oracle_revalidated_sample_count": sum(row["oracle_revalidated"] for row in predictions),
            "oracle_revalidated_pair_count": sum(result["accepted"] for result in pair_results),
            "plain_control_revalidated_count": 0,
            "modality_summary": dict(sorted(modality_summary.items())),
            "pair_results": pair_results,
            "predictions": predictions,
            "promotion_ledger": ledger,
            "memory_promotion_probe": promotion_probe,
            "ood_fit": ood_fit,
            "safety": {"raw_body_stored": False, "evaluator_state_visible": False, "external_network": False, "script_execution": False, "database_write": False, "real_sleep_performed": False},
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({"protocol_id": PROTOCOL_ID, "sample_count": report["sample_count"], "strict_ood_abstain_count": report["strict_ood_abstain_count"], "model_only_accepted_count": report["model_only_accepted_count"], "sql_decoder_injection_accepted_count": report["sql_decoder_injection_accepted_count"], "oracle_revalidated_sample_count": report["oracle_revalidated_sample_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "memory_promotion_status": promotion_probe["status"], "report": report["report_path"], "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
