"""Run one budgeted active-probe round on the local paired Pikachu catalog."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pikachu_active_controller import ACTIVE_CONTROLLER_SCHEMA, PikachuActiveController  # noqa: E402
from app.pikachu_replay_collector import PIKACHU_BASE_URL, PIKACHU_IMAGE_DIGEST, default_pikachu_paired_specs  # noqa: E402


PROTOCOL_ID = "pg-pk-03-active-safe-round-v1"
REPORT_PATH = ROOT / "research" / "pikachu_active_round_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_active_round_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pikachu_active_round_protocol_v1.json"
MAX_REQUESTS = 12


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    projection = record["oracle_projection"]
    pair = record.get("pair", {})
    return {
        "sample_id": record["sample_id"],
        "pair_id": pair.get("pair_id"),
        "surface_role": pair.get("surface_role"),
        "variant": pair.get("variant"),
        "probe_kind": record["payload"]["probe_kind"],
        "candidate_status": record.get("candidate_status"),
        "signals": [
            key for key, value in (
                ("marker_reflected", projection.get("marker_reflected")),
                ("marker_in_attribute", projection.get("marker_in_attribute")),
                ("sql_error_shape", projection.get("sql_error_shape")),
                ("external_redirect", projection.get("external_redirect")),
            ) if value
        ],
        "rule_ir_result": bool(record["rule_ir_result"]),
        "evidence_hash": record["evidence"]["evidence_hash"],
        "payload_is_exploit": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-03 主动安全探测轮",
        "",
        "控制器先为每个表面发送一个 plain canary，只有 bounded projection 出现候选信号时，才按 belief information gain 逐个追加编码变体；请求预算和 loopback scope 均由控制器强制。",
        "",
        f"请求数：{report['request_count']}/{report['max_requests']}；screen：{report['screen_count']}；refine：{report['refine_count']}；候选信号：{report['candidate_signal_count']}。",
        "",
        "| 阶段 | surface | variant | signals |",
        "|---|---|---|---|",
    ]
    for row in report["observations"]:
        lines.append(f"| {row['stage']} | `{row['surface_role']}` | `{row['variant']}` | {','.join(row['signals']) or 'none'} |")
    lines.extend([
        "",
        "这是主动选择探针的工程记录，不是漏洞确认；所有变体仍是无害编码/标识符，未执行脚本、SQL 语法、RCE、SSRF、XXE 或上传。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


async def _run() -> dict[str, Any]:
    result = await PikachuActiveController(max_requests=MAX_REQUESTS).run(default_pikachu_paired_specs())
    observations: list[dict[str, Any]] = []
    for trace, record in zip(result["selection_trace"], result["records"]):
        row = _summary(record)
        row["stage"] = trace["stage"]
        row["belief_information_gain"] = trace.get("belief_information_gain")
        observations.append(row)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-active-round-report-v1",
        "controller_schema": ACTIVE_CONTROLLER_SCHEMA,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "target": {
            "base_url": PIKACHU_BASE_URL,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
        },
        "max_requests": result["max_requests"],
        "request_count": result["request_count"],
        "screen_count": sum(trace["stage"] == "screen" for trace in result["selection_trace"]),
        "refine_count": sum(trace["stage"] == "refine" for trace in result["selection_trace"]),
        "candidate_signal_count": sum(bool(record["rule_ir_result"]) for record in result["records"]),
        "observations": observations,
        "selection_trace": result["selection_trace"],
        "belief": result["belief"],
        "safety": result["safety"],
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    report = asyncio.run(_run())
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "request_count": report["request_count"],
        "screen_count": report["screen_count"],
        "refine_count": report["refine_count"],
        "candidate_signal_count": report["candidate_signal_count"],
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
