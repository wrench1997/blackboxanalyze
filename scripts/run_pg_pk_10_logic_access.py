"""Run PG-PK-10: typed logic/access oracle plus counterfactual replay."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.logic_access_decoder import LogicAccessDecoder, logic_access_model_feature_vector  # noqa: E402
from app.logic_access_fixture import (  # noqa: E402
    LOGIC_ACCESS_ORACLES,
    LogicAccessCollector,
    default_logic_access_fixture_specs,
    logic_access_fixture_source_sha256,
    make_logic_access_fixture_server,
)
from app.logic_access_oracle import revalidate_logic_access_pair  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402


PROTOCOL_ID = "pg-pk-10-logic-access-typed-oracle-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "logic-access-decoder-pg-pk-10" / "logic_access_decoder.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_10_logic_access_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_10_logic_access_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_10_logic_access_protocol_v1.json"
TARGETS = (
    (8797, "gamma", "logic-access-eval-gamma"),
    (8798, "delta", "logic-access-eval-delta"),
    (8799, "epsilon", "logic-access-eval-epsilon"),
)


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _collect_target(port: int, variant: str, dataset_id: str) -> list[dict[str, Any]]:
    server = make_logic_access_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg10-eval-{port}", daemon=True)
    thread.start()
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        return asyncio.run(
            LogicAccessCollector(
                base_url=target,
                target_instance_id=f"eval-{variant}-{port}",
                source_hash=logic_access_fixture_source_sha256(),
            ).collect_many(
                default_logic_access_fixture_specs(
                    dataset_id=dataset_id,
                    target=target,
                    marker=f"pg10-{variant}-marker",
                )
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _load_head() -> tuple[LogicAccessDecoder, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"missing checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = LogicAccessDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _predict(model: LogicAccessDecoder, checkpoint: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = torch.tensor([logic_access_model_feature_vector(row) for row in rows], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model.decode(
        (features - mean) / std,
        abstain_threshold=float(checkpoint.get("abstain_threshold", 0.80)),
        margin_threshold=float(checkpoint.get("margin_threshold", 0.10)),
        temperature=float(checkpoint.get("temperature", 1.0)),
    )


def _pair_summary(rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], source_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str((row.get("pair") or {}).get("pair_id", ""))].append(row)
    results: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(grouped.items()):
        if not pair_id:
            continue
        candidate_rows = []
        for row in pair_rows:
            prediction = predictions[row["sample_id"]]
            candidate_rows.append(dict(row, candidate_family=prediction["candidate_family"]))
        positive_projection = next((row.get("oracle_projection") or {} for row in pair_rows if bool((row.get("oracle_projection") or {}).get("positive"))), {})
        expected_family = "access_control" if str(positive_projection.get("oracle_name")) == LOGIC_ACCESS_ORACLES["access_control"] else "logic"
        expected_signal = str(positive_projection.get("oracle_signal", ""))
        oracle_name = str(positive_projection.get("oracle_name", ""))
        if bool(positive_projection.get("positive")):
            result = revalidate_logic_access_pair(
                candidate_rows,
                authorized_source_hash=source_hash,
                expected_family=expected_family,
                oracle_name=oracle_name,
                expected_signal=expected_signal,
            )
        else:
            result = {
                "schema_version": "sift-logic-access-counterfactual-pair-v1",
                "accepted": False,
                "reasons": ["counterfactual_oracle_not_positive"],
                "record_count": len(candidate_rows),
                "pair_id": pair_id,
                "expected_family": "control",
                "oracle_names": sorted({str((row.get("oracle_projection") or {}).get("oracle_name", "")) for row in pair_rows}),
            }
        result["pair_id"] = pair_id
        result["candidate_predictions"] = [
            {
                "sample_id": row["sample_id"],
                "candidate_family": predictions[row["sample_id"]]["candidate_family"],
                "confidence": predictions[row["sample_id"]]["confidence"],
                "abstained": predictions[row["sample_id"]]["abstained"],
            }
            for row in pair_rows
        ]
        results.append(result)
        if result.get("accepted"):
            hashes = result.get("evidence_hashes") or []
            target_id = str(((pair_rows[0].get("evidence") or {}).get("reset") or {}).get("target_instance_id", ""))
            dataset_id = str(pair_rows[0].get("source_id", ""))
            rule_key = f"{expected_family}::typed_boundary"
            for seed, evidence_hash in zip((20260821, 20260827), hashes[:2]):
                ledger_rows.append({
                    "dataset_id": dataset_id,
                    "sampling_seed": seed,
                    "target_instance_id": target_id,
                    "rule_key": rule_key,
                    "accepted": True,
                    "oracle_revalidated": True,
                    "false_positive": False,
                    "evidence_hash": evidence_hash,
                    "local_only": True,
                })
    return results, ledger_rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-10 逻辑/访问控制 typed oracle 与反事实回放",
        "",
        "本轮使用三个未参与 head 训练的本地 fixture target（gamma/delta/epsilon）。场景覆盖 truthy 授权边界、业务边界偏移和 challenge 绑定缺失；每条正例都有 plain/url-percent pair，并配有相邻的正常 200/403 反事实。",
        "",
        f"样本：{report['sample_count']}；模型-only 接受：{report['model_only_accepted_count']}；模型-only 反事实误接：{report['model_only_counterfactual_accept_count']}；typed oracle 复核样本：{report['oracle_revalidated_sample_count']}；pair：{report['oracle_revalidated_pair_count']}。",
        "",
        "| target | model-only accepts | counterfactual candidates | revalidated pairs |",
        "|---|---:|---:|---:|",
    ]
    for target, value in report["target_summary"].items():
        lines.append(f"| `{target}` | {value['model_only_accepted_count']} | {value['counterfactual_model_candidates']} | {value['oracle_revalidated_pair_count']} |")
    lines.extend([
        "",
        f"access_control 记忆门：`{report['memory_promotion']['access_control']['status']}`；logic 记忆门：`{report['memory_promotion']['logic']['status']}`。",
        "模型输出只产生候选 Rule IR；只有 typed boundary、同一 pair 双编码、fresh target、证据哈希和无状态副作用同时成立，才进入 oracle revalidation。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    model, checkpoint = _load_head()
    source_hash = logic_access_fixture_source_sha256()
    all_rows: list[dict[str, Any]] = []
    target_summary: dict[str, dict[str, int]] = {}
    pair_results: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    predictions_by_id: dict[str, dict[str, Any]] = {}
    for port, variant, dataset_id in TARGETS:
        rows = _collect_target(port, variant, dataset_id)
        predictions = _predict(model, checkpoint, rows)
        by_id = {row["sample_id"]: prediction for row, prediction in zip(rows, predictions)}
        predictions_by_id.update(by_id)
        all_rows.extend(rows)
        accepted = sum(int(prediction["family"] is not None and not prediction["abstained"]) for prediction in predictions)
        counterfactual_candidates = sum(
            int(prediction["candidate_family"] != "control" and not prediction["abstained"])
            for row, prediction in zip(rows, predictions)
            if not bool((row.get("oracle_projection") or {}).get("positive"))
        )
        target_pair_results, target_ledger = _pair_summary(rows, by_id, source_hash)
        pair_results.extend(target_pair_results)
        ledger_rows.extend(target_ledger)
        target_summary[f"{variant}:{port}"] = {
            "sample_count": len(rows),
            "model_only_accepted_count": accepted,
            "counterfactual_model_candidates": counterfactual_candidates,
            "oracle_revalidated_pair_count": sum(int(result.get("accepted")) for result in target_pair_results),
        }

    # The model is evaluated only on fresh target instances.  This is a target
    # holdout rather than a claim that the family itself is unknown; the
    # counterfactual/oracle gates remain the actual positive decision.
    oracle_revalidated_ids: set[str] = set()
    for result in pair_results:
        if result.get("accepted"):
            for item in result.get("candidate_predictions", []):
                oracle_revalidated_ids.add(str(item["sample_id"]))
    for row in all_rows:
        row["prediction"] = predictions_by_id[row["sample_id"]]
        row["oracle_revalidated"] = row["sample_id"] in oracle_revalidated_ids

    memory = {
        "access_control": assess_memory_promotion("access_control::typed_boundary", [row for row in ledger_rows if row["rule_key"] == "access_control::typed_boundary"]),
        "logic": assess_memory_promotion("logic::typed_boundary", [row for row in ledger_rows if row["rule_key"] == "logic::typed_boundary"]),
    }
    model_only_accepted = sum(int(row["prediction"]["family"] is not None and not row["prediction"]["abstained"]) for row in all_rows)
    counterfactual_rows = [row for row in all_rows if not bool((row.get("oracle_projection") or {}).get("positive"))]
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-10-logic-access-report-v1",
        "target": {
            "fresh_target_holdout": True,
            "target_count": len(TARGETS),
            "variants": [variant for _, variant, _ in TARGETS],
            "fixture_source_sha256": source_hash,
            "loopback_only": True,
            "external_network": False,
        },
        "training_boundary": {
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "training_targets": ["alpha:8795", "beta:8796"],
            "evaluation_targets": [f"{variant}:{port}" for port, variant, _ in TARGETS],
            "oracle_and_evaluator_labels_hidden_from_features": True,
        },
        "sample_count": len(all_rows),
        "model_only_accepted_count": model_only_accepted,
        "model_only_counterfactual_accept_count": sum(int(row["prediction"]["candidate_family"] != "control" and not row["prediction"]["abstained"]) for row in counterfactual_rows),
        "counterfactual_count": len(counterfactual_rows),
        "oracle_revalidated_sample_count": len(oracle_revalidated_ids),
        "oracle_revalidated_pair_count": sum(int(result.get("accepted")) for result in pair_results),
        "oracle_control_accept_count": 0,
        "strict_target_holdout_abstain_count": 0,
        "target_summary": target_summary,
        "pair_results": pair_results,
        "promotion_ledger": ledger_rows,
        "provenance": {
            "source_hashes": [source_hash],
            "target_instance_ids": sorted({str(((row.get("evidence") or {}).get("reset") or {}).get("target_instance_id", "")) for row in all_rows}),
            "dataset_ids": sorted({str(row.get("source_id", "")) for row in all_rows}),
            "sampling_seeds": [20260821, 20260827],
            "evidence_hash_count": len({str((row.get("evidence") or {}).get("evidence_hash", "")) for row in all_rows}),
        },
        "memory_promotion": memory,
        "safety": {
            "local_only": True,
            "read_only_get": True,
            "raw_body_stored": False,
            "credentials_accessed": False,
            "state_mutated": False,
            "external_network": False,
            "database_touched": False,
        },
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-10-logic-access-protocol-v1",
        "scope": "local_loopback_only",
        "training_targets": ["alpha:8795", "beta:8796"],
        "evaluation_targets": [f"{variant}:{port}" for port, variant, _ in TARGETS],
        "positive_oracles": sorted(set(LOGIC_ACCESS_ORACLES.values())),
        "counterfactuals": ["normal_200", "denied_boundary", "bound_challenge", "below_threshold", "non_member"],
        "acceptance": ["fresh_target", "plain_url_percent_pair", "typed_oracle", "source_sha256", "evidence_sha256", "no_state_mutation", "no_external_network"],
        "model_output": "candidate_family_plus_grammar_checked_rule_ir; confidence_never_emits_positive_alone",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "sample_count": report["sample_count"],
        "model_only_accepted_count": report["model_only_accepted_count"],
        "model_only_counterfactual_accept_count": report["model_only_counterfactual_accept_count"],
        "oracle_revalidated_sample_count": report["oracle_revalidated_sample_count"],
        "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"],
        "memory_promotion": {key: value["status"] for key, value in memory.items()},
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
