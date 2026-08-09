"""Collect real local no-signal controls and merge them with the paired catalog."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import load_catalog, write_catalog  # noqa: E402
from app.pikachu_replay_collector import (  # noqa: E402
    PIKACHU_BASE_URL,
    PIKACHU_COLLECTOR_SCHEMA,
    PIKACHU_IMAGE_DIGEST,
    PikachuReplayCollector,
    default_pikachu_counterfactual_specs,
)


PROTOCOL_ID = "pg-pk-04-counterfactual-negative-controls-v1"
PAIRED_CATALOG_PATH = ROOT / "research" / "pikachu_paired_catalog_v1.json"
CATALOG_PATH = ROOT / "research" / "pikachu_counterfactual_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pikachu_counterfactual_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_counterfactual_v1.md"
SOURCE_ID = "pikachu-counterfactual-pg05"


def _provenance() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_type": "authorized_local_container",
        "origin": "app/pikachu_replay_collector.py",
        "license": "local_container",
        "authorization": "workspace_local_only",
        "scope": [PIKACHU_BASE_URL],
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "authorized_for": ["training", "local_replay", "holdout_evaluation"],
        "external_network": False,
        "evaluator_state_visible": False,
        "container_image_digest": PIKACHU_IMAGE_DIGEST,
    }


def _matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        projection = record["oracle_projection"]
        rows.append({
            "sample_id": record["sample_id"],
            "family": record["semantic"]["family"],
            "surface": record["semantic"]["surface"],
            "marker_reflected": bool(projection.get("marker_reflected")),
            "marker_in_attribute": bool(projection.get("marker_in_attribute")),
            "sql_error_shape": bool(projection.get("sql_error_shape")),
            "body_length_delta_abs": int(projection.get("body_length_delta_abs", 0)),
            "rule_ir_result": bool(record["rule_ir_result"]),
            "counterfactual": record.get("counterfactual"),
            "evidence_hash": record["evidence"]["evidence_hash"],
            "payload_is_exploit": False,
        })
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-04 反事实负样本",
        "",
        "每条控制样本仍向本地 Pikachu 发送无害输入，但 oracle 期待的 marker 与实际输入不同；因此可以测量模型是否把‘输入被回显’误当成目标证据。",
        "",
        f"正向配对样本：{report['positive_count']}；反事实控制：{report['negative_count']}；控制样本 Rule IR 信号：{report['negative_rule_signal_count']}。",
        "",
        "| family | surface | marker_reflected | sql_error_shape | Rule IR signal |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["negative_matrix"]:
        lines.append(
            f"| `{row['family']}` | `{row['surface']}` | {int(row['marker_reflected'])} | {int(row['sql_error_shape'])} | {int(row['rule_ir_result'])} |"
        )
    lines.extend([
        "",
        "这些是校准/拒答负样本，不把 family 标签直接提供给模型；原始响应体、Cookie、凭据和 evaluator 状态均未保存。",
        "没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。",
        "",
        f"Catalog：`{report['catalog_path']}`",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


async def _run() -> dict[str, Any]:
    baseline = load_catalog(PAIRED_CATALOG_PATH)
    negative_records = await PikachuReplayCollector().collect_many(default_pikachu_counterfactual_specs())
    for record in negative_records:
        record["counterfactual"] = {
            "kind": "negative_control",
            "intervention": "marker_substitution",
        }
    if any(bool(record["rule_ir_result"]) for record in negative_records):
        raise RuntimeError("counterfactual negative control unexpectedly produced a Rule IR signal")
    catalog = write_catalog(CATALOG_PATH, {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pikachu-counterfactual-v1",
        "sources": [
            baseline["sources"][0],
            {"provenance": _provenance(), "samples": negative_records},
        ],
    })
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-counterfactual-report-v1",
        "target": {
            "base_url": PIKACHU_BASE_URL,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "fresh_reset_per_probe": False,
        },
        "positive_count": len(baseline["sources"][0]["samples"]),
        "negative_count": len(negative_records),
        "negative_rule_signal_count": sum(bool(record["rule_ir_result"]) for record in negative_records),
        "negative_family_counts": dict(Counter(record["semantic"]["family"] for record in negative_records)),
        "negative_surface_counts": dict(Counter(record["semantic"]["surface"] for record in negative_records)),
        "negative_matrix": _matrix(negative_records),
        "all_negative_non_exploit": all(bool(record["payload"]["safety"]["does_not_execute"]) for record in negative_records),
        "catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
        "catalog_sha256": catalog["catalog_sha256"],
        "collector_schema": PIKACHU_COLLECTOR_SCHEMA,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    report = asyncio.run(_run())
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "positive_count": report["positive_count"],
        "negative_count": report["negative_count"],
        "negative_rule_signal_count": report["negative_rule_signal_count"],
        "catalog": report["catalog_path"],
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
