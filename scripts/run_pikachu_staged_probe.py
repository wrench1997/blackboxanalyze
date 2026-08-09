"""Run the local Pikachu staged-probe experiment.

Stage 0 inventories read-only pages.  Stage 1 sends inert canaries.  Stage 2
is gated by a visible response signal and sends exactly one encoded/abstract
variant at a time.  The script writes a source-attested Payload Catalog and a
bounded matrix; it never sends exploit syntax or executes a browser oracle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.belief_state import MultiStepBelief
from app.maze_engine import sha256_json
from app.payload_catalog import flatten_catalog, load_catalog, write_catalog
from app.pikachu_replay_collector import (
    PIKACHU_BASE_URL,
    PIKACHU_COLLECTOR_SCHEMA,
    PIKACHU_IMAGE_DIGEST,
    SAFE_INVENTORY_PATHS,
    PikachuReplayCollector,
    default_pikachu_probe_specs,
)


PROTOCOL_ID = "pg-pk-01-staged-local-probe-v1"
CATALOG_PATH = ROOT / "research" / "pikachu_payload_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pikachu_staged_probe_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_staged_probe_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pikachu_staged_probe_protocol_v1.json"
SOURCE_ID = "pikachu-local-container-pg-pk-01"
SOURCE_DATE = datetime.now(timezone.utc).isoformat()

ABSTAIN_REASONS = {
    "/vul/xss/xss_stored.php": "requires POST and a database write; no state-mutating probe in the safe track",
    "/vul/ssrf/ssrf_curl.php": "would require a callback or non-loopback request; external network is prohibited",
    "/vul/fileinclude/fi_local.php": "would exercise file disclosure/path resolution; no file oracle is enabled",
    "/vul/dir/dir_list.php": "read-only page observed, but no approved directory-traversal oracle is enabled",
    "/vul/infoleak/findabc.php": "form contains credential-like fields; credential probes are prohibited",
    "/vul/xxe/xxe_1.php": "XML entity resolution is disabled in the safe track",
    "/vul/rce/rce_ping.php": "command execution is disabled in the safe track",
    "/vul/unsafeupload/upload.php": "file upload/write is disabled in the safe track",
}


def _source_provenance() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_type": "authorized_local_container",
        "origin": "app/pikachu_replay_collector.py",
        "license": "local_container",
        "authorization": "workspace_local_only",
        "scope": [PIKACHU_BASE_URL],
        "captured_at": SOURCE_DATE,
        "authorized_for": ["training", "local_replay", "holdout_evaluation"],
        "external_network": False,
        "evaluator_state_visible": False,
        "container_image_digest": PIKACHU_IMAGE_DIGEST,
    }


async def _inventory() -> list[dict[str, Any]]:
    """Collect status/length/hash only, with no query parameters or cookies."""

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=PIKACHU_BASE_URL,
        timeout=5.0,
        follow_redirects=False,
        cookies={},
    ) as client:
        for path in sorted(SAFE_INVENTORY_PATHS):
            response = await client.get(path, headers={"accept": "text/html", "x-sift-probe": "pk-inventory"})
            body = response.content
            rows.append({
                "stage": "inventory",
                "path": path,
                "status_code": int(response.status_code),
                "content_type": str(response.headers.get("content-type", "")),
                "body_length": len(body),
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "executed": True,
                "next_action": "safe_canary" if path in {"/vul/xss/xss_reflected_get.php", "/vul/xss/xss_dom.php", "/vul/sqli/sqli_str.php", "/vul/sqli/sqli_search.php", "/vul/sqli/sqli_blind_b.php", "/vul/sqli/sqli_blind_t.php", "/vul/urlredirect/urlredirect.php"} else "abstain",
                "abstain_reason": ABSTAIN_REASONS.get(path),
            })
    return rows


def _refinement_spec(record: dict[str, Any], *, ordinal: int) -> dict[str, Any]:
    """Make one safe variant after a suspicious bounded projection."""

    payload = record["payload"]
    marker = f"pk-refine-{ordinal:02d}"
    params = dict(record["replay"]["params"])
    if record["family"] == "xss":
        encoded = f"&lt;span data-sift-marker=&#34;{marker}&#34;&gt;x&lt;/span&gt;"
        field = "text" if "text" in params else "message"
        params[field] = encoded
        return {
            "source_id": SOURCE_ID,
            "lab_id": f"{record['lab_id']}-refinement-{ordinal:02d}",
            "family": "xss",
            "surface": f"{record['semantic']['surface']}_encoded_variant",
            "path": payload["path"],
            "params": params,
            "probe_kind": "encoded_dom_markup",
            "marker": marker,
            "probe": encoded,
            "encoding": "html_entity_encode_depth_1_inert",
            "expected_signal": "encoding_boundary_only",
        }
    if record["family"] == "injection":
        # The abstract channel changes, but the transmitted value remains an
        # inert identifier.  No quote/operator/comment/sleep syntax is sent.
        field = "name"
        params[field] = marker
        return {
            "source_id": SOURCE_ID,
            "lab_id": f"{record['lab_id']}-refinement-{ordinal:02d}",
            "family": "injection",
            "surface": f"{record['semantic']['surface']}_abstract_variant",
            "path": payload["path"],
            "params": params,
            "probe_kind": "sql_channel_class",
            "marker": marker,
            "probe": "quoted_value",
            "encoding": "abstract_sql_fragment_class_only",
            "expected_signal": "bounded_differential_only",
        }
    raise ValueError("refinement is not defined for this family")


def _surface_likelihood(record: dict[str, Any]) -> dict[str, float]:
    """A label-free, deliberately weak likelihood baseline for belief updates."""

    projection = record["oracle_projection"]
    if projection.get("marker_reflected") or projection.get("marker_in_script_source"):
        return {"xss": 0.62, "injection": 0.08, "access_control": 0.08, "url_redirect": 0.08, "logic": 0.14}
    if projection.get("sql_error_shape") or projection.get("body_length_delta_abs", 0) >= 256:
        return {"xss": 0.08, "injection": 0.62, "access_control": 0.08, "url_redirect": 0.08, "logic": 0.14}
    if projection.get("external_redirect"):
        return {"xss": 0.06, "injection": 0.06, "access_control": 0.08, "url_redirect": 0.68, "logic": 0.12}
    return {family: 0.20 for family in ("xss", "injection", "access_control", "url_redirect", "logic")}


def _catalog(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pikachu-payload-grounding-v1",
        "sources": [{"provenance": _source_provenance(), "samples": records}],
    }


def _matrix_row(record: dict[str, Any], *, stage: str, belief_step: dict[str, Any] | None = None) -> dict[str, Any]:
    projection = record["oracle_projection"]
    signals = [
        name for name, enabled in (
            ("marker_reflected", projection.get("marker_reflected")),
            ("marker_in_attribute", projection.get("marker_in_attribute")),
            ("marker_in_script_source", projection.get("marker_in_script_source")),
            ("sql_error_shape", projection.get("sql_error_shape")),
            ("large_shape_delta", projection.get("body_length_delta_abs", 0) >= 256),
            ("external_redirect", projection.get("external_redirect")),
        ) if enabled
    ]
    return {
        "stage": stage,
        "sample_id": record["sample_id"],
        "lab_id": record["lab_id"],
        "family": record["family"],
        "surface": record["semantic"]["surface"],
        "path": record["payload"]["path"],
        "probe_kind": record["payload"]["probe_kind"],
        "candidate_status": record.get("candidate_status", "unknown"),
        "signals": signals,
        "rule_ir_result": bool(record["rule_ir_result"]),
        "next_action": "one_at_a_time_refinement" if record["rule_ir_result"] else "abstain_or_collect_more_benign_evidence",
        "evidence_hash": record["evidence"]["evidence_hash"],
        "belief_step": belief_step,
        "payload_is_exploit": False,
        "evaluator_confirmation": False,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-01 分阶段本地探针",
        "",
        "本实验把 AI 的动作拆成：只读 inventory → 无害 canary → 根据 bounded response signal 更新 belief → 逐个发送安全变体 → 无法证明时 abstain。请求严格限制为 `http://127.0.0.1:8766` 的 GET；不执行脚本，不发送 SQL 语法、命令、外部 URL、实体、上传或延时 payload。",
        "",
        f"阶段 1 样本：{report['capture']['stage1_count']}；阶段 2 样本：{report['capture']['stage2_count']}；可疑信号：{report['capture']['suspicious_count']}；明确 abstain：{report['capture']['abstain_count']}。",
        "",
        "| 阶段 | endpoint | probe | 结果 | 下一步 |",
        "|---|---|---|---|---|",
    ]
    for row in report["matrix"]:
        lines.append(
            f"| {row['stage']} | `{row['path']}` | `{row['probe_kind']}` | "
            f"{row['candidate_status']} ({','.join(row['signals']) or 'none'}) | {row['next_action']} |"
        )
    lines.extend([
        "",
        "## 读法",
        "",
        "`suspicious_surface_signal` 只表示响应表面出现了候选信号。例如 reflected canary 被回显，不等于浏览器执行，也不等于 evaluator 已确认漏洞；下一次变体仍然是无害编码边界探针。高风险族被记录为 abstain，不能把未执行当作通过。",
        "",
        f"Catalog：`{report['catalog']['path']}`（SHA-256 `{report['catalog']['catalog_sha256']}`）",
        f"协议：`{report['protocol_path']}`",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


async def _run() -> dict[str, Any]:
    inventory = await _inventory()
    collector = PikachuReplayCollector()
    stage1_specs = default_pikachu_probe_specs("pk-safe-probe-a1")
    stage1_records = await collector.collect_many(stage1_specs)
    belief = MultiStepBelief()
    matrix = []
    belief_steps = []
    for record in stage1_records:
        step = belief.observe(
            record["payload"]["path"],
            _surface_likelihood(record),
            evidence_hash=record["evidence"]["evidence_hash"],
        )
        belief_steps.append(step)
        matrix.append(_matrix_row(record, stage="stage_1_safe_canary", belief_step=step))

    suspicious = [record for record in stage1_records if record["rule_ir_result"]]
    stage2_specs = [_refinement_spec(record, ordinal=index + 1) for index, record in enumerate(suspicious)]
    stage2_records = await collector.collect_many(stage2_specs)
    for record in stage2_records:
        step = belief.observe(
            record["payload"]["path"],
            _surface_likelihood(record),
            evidence_hash=record["evidence"]["evidence_hash"],
        )
        belief_steps.append(step)
        matrix.append(_matrix_row(record, stage="stage_2_gated_refinement", belief_step=step))

    abstentions = [
        {
            "stage": "stage_2_abstain",
            "path": row["path"],
            "executed": False,
            "reason": row["abstain_reason"],
            "next_action": "requires_separate_authorized_oracle_and_reset_protocol",
        }
        for row in inventory if row.get("abstain_reason")
    ]
    records = stage1_records + stage2_records
    catalog = write_catalog(CATALOG_PATH, _catalog(records))
    normalized = flatten_catalog(load_catalog(CATALOG_PATH))
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-staged-probe-report-v1",
        "target": {
            "base_url": PIKACHU_BASE_URL,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "scope": "loopback_only",
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "fresh_reset_per_probe": False,
        },
        "method": {
            "stage_0": "read-only inventory",
            "stage_1": "inert canary",
            "stage_2": "one-at-a-time refinement gated by visible projection",
            "abstention": "explicit for hazardous or unsupported oracle families",
            "belief_update": "MultiStepBelief from label-free bounded projection",
        },
        "inventory": inventory,
        "matrix": matrix,
        "abstentions": abstentions,
        "belief": belief.snapshot(),
        "belief_steps": belief_steps,
        "capture": {
            "stage1_count": len(stage1_records),
            "stage2_count": len(stage2_records),
            "suspicious_count": len(suspicious),
            "abstain_count": len(abstentions),
            "catalog_sample_count": len(normalized),
            "all_samples_local": all(row["payload"]["target"] == PIKACHU_BASE_URL for row in normalized),
            "all_samples_non_exploit": all(
                bool(row.get("payload", {}).get("safety", {}).get("does_not_execute", False))
                for row in normalized
            ),
        },
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": catalog["catalog_sha256"],
            "source_count": len(catalog["sources"]),
            "sample_count": len(normalized),
        },
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "collector_schema": PIKACHU_COLLECTOR_SCHEMA,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    report = asyncio.run(_run())
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "capture": report["capture"],
        "belief": report["belief"],
        "catalog": report["catalog"],
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
