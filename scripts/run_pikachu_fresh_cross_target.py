"""Replay the bounded active round on a fresh pinned local Pikachu instance.

The script creates only a fixed-name, loopback-only, ephemeral container from
the already-pinned image.  It compares bounded signal summaries with the
previous 8766 round; no raw response body is persisted.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pikachu_active_controller import PikachuActiveController  # noqa: E402
from app.pikachu_replay_collector import (  # noqa: E402
    PIKACHU_BASE_URL,
    PIKACHU_FRESH_BASE_URL,
    PIKACHU_IMAGE_DIGEST,
    PikachuReplayCollector,
    default_pikachu_paired_specs,
)


PROTOCOL_ID = "pg-pk-03-fresh-cross-target-v1"
CONTAINER_NAME = "sift-pikachu-fresh-8767"
BASELINE_REPORT_PATH = ROOT / "research" / "pikachu_active_round_v1.json"
REPORT_PATH = ROOT / "research" / "pikachu_fresh_cross_target_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_fresh_cross_target_v1.md"
FRESH_START_TIMEOUT_SECONDS = 120


def _run_docker(*args: str) -> str:
    completed = subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _container_exists() -> bool:
    return bool(_run_docker("ps", "-a", "--filter", f"name=^{CONTAINER_NAME}$", "--format", "{{.Names}}"))


def _start_fresh_container() -> str:
    if _container_exists():
        raise RuntimeError(f"refusing to reuse pre-existing container {CONTAINER_NAME}")
    _run_docker(
        "run",
        "--detach",
        "--rm",
        "--pull=never",
        "--name",
        CONTAINER_NAME,
        "--publish",
        "127.0.0.1:8767:8090",
        f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}",
        "bash",
        "-lc",
        "/app/run.sh; exec tail -f /dev/null",
    )
    try:
        deadline = time.monotonic() + FRESH_START_TIMEOUT_SECONDS
        last_error = "not attempted"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{PIKACHU_FRESH_BASE_URL}/", timeout=2.0, follow_redirects=False)
                if response.status_code < 500:
                    return _run_docker("inspect", "--format", "{{.Id}}", CONTAINER_NAME)
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(f"fresh Pikachu container did not become ready: {last_error}")
    except BaseException:
        # The container was created by this function and is fixed-name/ephemeral.
        _stop_fresh_container()
        raise


def _stop_fresh_container() -> None:
    if _container_exists():
        _run_docker("stop", "--time", "5", CONTAINER_NAME)


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    projection = record["oracle_projection"]
    pair = record.get("pair", {})
    return {
        "pair_id": pair.get("pair_id"),
        "surface_role": pair.get("surface_role"),
        "variant": pair.get("variant"),
        "signals": sorted(
            key for key, value in (
                ("marker_reflected", projection.get("marker_reflected")),
                ("marker_in_attribute", projection.get("marker_in_attribute")),
                ("sql_error_shape", projection.get("sql_error_shape")),
                ("external_redirect", projection.get("external_redirect")),
            ) if value
        ),
        "rule_ir_result": bool(record["rule_ir_result"]),
        "payload_is_exploit": False,
    }


def _key(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return row.get("pair_id"), row.get("surface_role"), row.get("variant")


def _markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    lines = [
        "# Pikachu PG-PK-03 fresh cross-target 回放",
        "",
        f"旧实例请求数：{report['baseline']['request_count']}；新实例请求数：{report['fresh']['request_count']}。",
        f"同一表面/编码的 bounded 结果一致率：{report['agreement']['observation_agreement_rate']:.2%}；一致条目：{report['agreement']['matching_observations']}/{report['agreement']['compared_observations']}。",
        "",
        "| surface | variant | 旧实例 signals | 新实例 signals | 一致 |",
        "|---|---|---|---|---|",
    ]
    for row in comparison:
        lines.append(
            f"| `{row['surface_role']}` | `{row['variant']}` | {','.join(row['baseline_signals']) or 'none'} | {','.join(row['fresh_signals']) or 'none'} | {'yes' if row['match'] else 'NO'} |"
        )
    lines.extend([
        "",
        "新实例由固定 SHA-256 镜像创建，端口仅绑定 127.0.0.1:8767；容器在脚本结束时停止并由 `--rm` 回收。",
        "这仍然是 bounded signal 的跨实例稳定性实验，不是漏洞确认；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。",
        "",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


async def _run_round(container_id: str) -> dict[str, Any]:
    collector = PikachuReplayCollector(
        base_url=PIKACHU_FRESH_BASE_URL,
        target_instance_id=container_id,
        fresh_target=True,
    )
    specs = default_pikachu_paired_specs()
    for spec in specs:
        spec["target"] = PIKACHU_FRESH_BASE_URL
    return await PikachuActiveController(collector, max_requests=12).run(specs)


def main() -> None:
    container_id = ""
    started = False
    try:
        container_id = _start_fresh_container()
        started = True
        result = asyncio.run(_run_round(container_id))
        baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        baseline_rows = {_key(row): row for row in baseline["observations"]}
        fresh_rows = {_key(_summary(record)): _summary(record) for record in result["records"]}
        comparison = []
        for key in sorted(set(baseline_rows) | set(fresh_rows), key=str):
            old = baseline_rows.get(key, {"signals": [], "rule_ir_result": False})
            new = fresh_rows.get(key, {"signals": [], "rule_ir_result": False})
            comparison.append({
                "pair_id": key[0],
                "surface_role": key[1],
                "variant": key[2],
                "baseline_signals": old["signals"],
                "fresh_signals": new["signals"],
                "baseline_rule_ir_result": bool(old["rule_ir_result"]),
                "fresh_rule_ir_result": bool(new["rule_ir_result"]),
                "match": sorted(old["signals"]) == sorted(new["signals"]) and bool(old["rule_ir_result"]) == bool(new["rule_ir_result"]),
            })
        matching = sum(row["match"] for row in comparison)
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pikachu-fresh-cross-target-report-v1",
            "target": {
                "fresh_base_url": PIKACHU_FRESH_BASE_URL,
                "container_name": CONTAINER_NAME,
                "container_id": container_id,
                "container_image_digest": PIKACHU_IMAGE_DIGEST,
                "external_network": False,
                "loopback_only": True,
                "fresh_target": True,
            },
            "baseline": {
                "base_url": PIKACHU_BASE_URL,
                "report_path": str(BASELINE_REPORT_PATH.relative_to(ROOT)),
                "request_count": baseline["request_count"],
            },
            "fresh": {
                "base_url": PIKACHU_FRESH_BASE_URL,
                "request_count": result["request_count"],
                "screen_count": sum(trace["stage"] == "screen" for trace in result["selection_trace"]),
                "refine_count": sum(trace["stage"] == "refine" for trace in result["selection_trace"]),
                "candidate_signal_count": sum(bool(record["rule_ir_result"]) for record in result["records"]),
            },
            "agreement": {
                "compared_observations": len(comparison),
                "matching_observations": matching,
                "observation_agreement_rate": matching / len(comparison) if comparison else 1.0,
            },
            "comparison": comparison,
            "safety": result["safety"],
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "container_id": container_id,
            "request_count": result["request_count"],
            "agreement_rate": report["agreement"]["observation_agreement_rate"],
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
        }, ensure_ascii=False, indent=2))
    finally:
        if started:
            _stop_fresh_container()


if __name__ == "__main__":
    main()
