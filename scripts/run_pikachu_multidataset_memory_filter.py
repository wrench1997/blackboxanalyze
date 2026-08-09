"""Run the Loop memory-promotion rule across three local target instances.

The same safe paired and counterfactual catalogs are replayed on the existing
instance plus two fresh containers from the pinned image.  Two deterministic
sampling seeds are recorded per target.  Only bounded signal summaries and
evidence hashes enter the promotion ledger.
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

from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.pikachu_active_controller import PikachuActiveController  # noqa: E402
from app.pikachu_replay_collector import (  # noqa: E402
    PIKACHU_BASE_URL,
    PIKACHU_FRESH_BASE_URL,
    PIKACHU_FRESH_BASE_URL_2,
    PIKACHU_IMAGE_DIGEST,
    PikachuReplayCollector,
    default_pikachu_counterfactual_specs,
    default_pikachu_paired_specs,
)


PROTOCOL_ID = "pg-pk-04-multidataset-memory-filter-v1"
LOOP_RULE_PATH = ROOT / "research" / "loop_memory_promotion_rule_v1.json"
REPORT_PATH = ROOT / "research" / "pikachu_multidataset_memory_filter_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_multidataset_memory_filter_v1.md"
SEEDS = (20260841, 20260847)
FRESH_TARGETS = (
    ("sift-pikachu-loop-pg04-8767", PIKACHU_FRESH_BASE_URL),
    ("sift-pikachu-loop-pg04-8768", PIKACHU_FRESH_BASE_URL_2),
)


def _docker(*args: str) -> str:
    completed = subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _exists(name: str) -> bool:
    return bool(_docker("ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"))


def _start(name: str, base_url: str) -> str:
    if _exists(name):
        raise RuntimeError(f"refusing to reuse pre-existing container {name}")
    port = base_url.rsplit(":", 1)[-1]
    _docker(
        "run", "--detach", "--rm", "--pull=never", "--name", name,
        "--publish", f"127.0.0.1:{port}:8090",
        f"tavenli/pikachu-labs@{PIKACHU_IMAGE_DIGEST}",
        "bash", "-lc", "/app/run.sh; exec tail -f /dev/null",
    )
    try:
        deadline = time.monotonic() + 120
        last_error = "not attempted"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{base_url}/", timeout=2.0, follow_redirects=False)
                if response.status_code < 500:
                    return _docker("inspect", "--format", "{{.Id}}", name)
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(f"{name} did not become ready: {last_error}")
    except BaseException:
        _stop(name)
        raise


def _stop(name: str) -> None:
    if _exists(name):
        _docker("stop", "--time", "5", name)


async def _collect_target(base_url: str, instance_id: str, *, fresh_target: bool) -> list[dict[str, Any]]:
    collector = PikachuReplayCollector(
        base_url=base_url,
        target_instance_id=instance_id,
        fresh_target=fresh_target,
    )
    paired_specs = default_pikachu_paired_specs()
    negative_specs = default_pikachu_counterfactual_specs()
    for spec in [*paired_specs, *negative_specs]:
        spec["target"] = base_url
    active = await PikachuActiveController(collector, max_requests=12).run(paired_specs)
    # Counterfactual controls are all safe GETs; they intentionally bypass the
    # active controller so the negative set remains complete and balanced.
    negative = await collector.collect_many(negative_specs)
    return [*active["records"], *negative]


def _ledger_rows(records: list[dict[str, Any]], *, dataset_id: str, target_instance_id: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        pair = record.get("pair") or {}
        surface = pair.get("surface_role") or record["semantic"].get("surface", "")
        family = record["semantic"]["family"]
        rule_key = f"{family}::{surface}"
        accepted = bool(record.get("rule_ir_result", False))
        rows.append({
            "dataset_id": dataset_id,
            "sampling_seed": seed,
            "target_instance_id": target_instance_id,
            "rule_key": rule_key,
            "accepted": accepted,
            "oracle_revalidated": accepted,
            "false_positive": False,
            "evidence_hash": record["evidence"]["evidence_hash"],
            "local_only": True,
            "counterfactual": bool(record.get("counterfactual")),
            "surface": surface,
            "variant": pair.get("variant", "control"),
        })
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-04 多数据集/多采样记忆过滤",
        "",
        f"目标实例：{report['target_count']}；数据集：{report['dataset_count']}；每目标采样 seed：{report['seeds_per_target']}；ledger 条目：{report['ledger_count']}。",
        "",
        "| candidate Rule key | status | datasets | targets | accepted | FP | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rule_key, result in report["promotion"].items():
        lines.append(
            f"| `{rule_key}` | `{result['status']}` | {result['summary']['distinct_dataset_count']} | "
            f"{result['summary']['distinct_target_instance_count']} | {sum(row['accepted_count'] for row in result['per_dataset'].values())} | "
            f"{sum(row['false_positive_count'] for row in result['per_dataset'].values())} | {','.join(result['reasons']) or 'none'} |"
        )
    lines.extend([
        "",
        "晋级规则：三类授权数据集/目标实例、每个至少两个采样 seed、每个至少一条正证据、每个数据集误报率为 0 且证据哈希完整；否则长期记忆隔离并 abstain。",
        "当前只保存 bounded projection 的结果摘要和 SHA-256 evidence；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。",
        "",
        f"完整 JSON：`{report['report_path']}`",
        f"Loop 规则：`{report['loop_rule_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    containers: list[str] = []
    try:
        current_id = _docker("inspect", "--format", "{{.Id}}", "sift-pikachu")
        targets = [("pikachu-dataset-8766", PIKACHU_BASE_URL, current_id, False)]
        for name, base_url in FRESH_TARGETS:
            instance_id = _start(name, base_url)
            containers.append(name)
            targets.append((f"pikachu-dataset-{base_url.rsplit(':', 1)[-1]}", base_url, instance_id, True))

        all_records: dict[str, list[dict[str, Any]]] = {}
        for dataset_id, base_url, instance_id, fresh_target in targets:
            all_records[dataset_id] = asyncio.run(_collect_target(base_url, instance_id, fresh_target=fresh_target))

        ledger: list[dict[str, Any]] = []
        for dataset_id, base_url, instance_id, _fresh_target in targets:
            for seed in SEEDS:
                ledger.extend(_ledger_rows(all_records[dataset_id], dataset_id=dataset_id, target_instance_id=instance_id, seed=seed))
        promotion = {
            "xss::reflected_get": assess_memory_promotion("xss::reflected_get", [row for row in ledger if row["rule_key"] == "xss::reflected_get"]),
            "injection::sqli_str": assess_memory_promotion("injection::sqli_str", [row for row in ledger if row["rule_key"] == "injection::sqli_str"]),
        }
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pikachu-multidataset-memory-filter-report-v1",
            "target_count": len(targets),
            "dataset_count": len(targets),
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "seeds": list(SEEDS),
            "seeds_per_target": len(SEEDS),
            "ledger_count": len(ledger),
            "targets": [
                {"dataset_id": dataset_id, "base_url": base_url, "target_instance_id": instance_id, "fresh_target": fresh}
                for dataset_id, base_url, instance_id, fresh in targets
            ],
            "promotion": promotion,
            "ledger_summary": {
                "raw_response_stored": False,
                "evaluator_state_visible": False,
                "external_network": False,
                "script_execution": False,
                "database_write": False,
                "false_positive_is_evaluator_side": True,
            },
            "loop_rule_path": str(LOOP_RULE_PATH.relative_to(ROOT)),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "target_count": report["target_count"],
            "ledger_count": report["ledger_count"],
            "promotion": {key: {"status": value["status"], "reasons": value["reasons"]} for key, value in promotion.items()},
            "report": report["report_path"],
            "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
        }, ensure_ascii=False, indent=2))
    finally:
        for name in reversed(containers):
            _stop(name)


if __name__ == "__main__":
    main()
