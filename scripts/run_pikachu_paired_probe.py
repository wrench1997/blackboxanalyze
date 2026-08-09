"""Collect multi-encoding, cross-surface Pikachu canary pairs.

Every request remains a read-only GET to the pinned loopback container.  The
pair metadata is evaluator-side training metadata; the decoder receives only
the visible action/probe/response projection and never the pair id or family.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import flatten_catalog, load_catalog, write_catalog  # noqa: E402
from app.pikachu_replay_collector import (  # noqa: E402
    PIKACHU_BASE_URL,
    PIKACHU_COLLECTOR_SCHEMA,
    PIKACHU_IMAGE_DIGEST,
    PikachuReplayCollector,
    default_pikachu_paired_specs,
)


PROTOCOL_ID = "pg-pk-02-paired-encoding-surface-v1"
CATALOG_PATH = ROOT / "research" / "pikachu_paired_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pikachu_pair_invariance_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pikachu_pair_invariance_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pikachu_pair_invariance_protocol_v1.json"
SOURCE_ID = "pikachu-pair-pg04"


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


def _catalog(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pikachu-paired-encoding-surface-v1",
        "sources": [{"provenance": _provenance(), "samples": records}],
    }


def _matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        projection = record["oracle_projection"]
        pair = record.get("pair", {})
        rows.append({
            "sample_id": record["sample_id"],
            "pair_id": pair.get("pair_id"),
            "variant": pair.get("variant"),
            "encoding_depth": pair.get("encoding_depth"),
            "surface_role": pair.get("surface_role"),
            "family": record["semantic"]["family"],
            "path": record["payload"]["path"],
            "probe_kind": record["payload"]["probe_kind"],
            "marker_reflected": bool(projection.get("marker_reflected")),
            "marker_in_attribute": bool(projection.get("marker_in_attribute")),
            "sql_error_shape": bool(projection.get("sql_error_shape")),
            "body_length_delta_abs": int(projection.get("body_length_delta_abs", 0)),
            "rule_ir_result": bool(record["rule_ir_result"]),
            "evidence_hash": record["evidence"]["evidence_hash"],
            "payload_is_exploit": False,
            "evaluator_confirmation": False,
        })
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pikachu PG-PK-02 编码/表面配对实验",
        "",
        "本轮把同一抽象族拆成 plain、URL percent、HTML entity、double HTML entity 四种无害表示，并跨多个本地页面表面配对。pair id 只用于训练的一致性损失和评估分组，不进入模型可见输入。",
        "",
        f"样本：{report['capture']['sample_count']}；配对组：{report['capture']['pair_count']}；表面：{report['capture']['surface_count']}；编码变体：{report['capture']['variant_count']}。",
        "",
        "| pair | surface | variants | rule signal count |",
        "|---|---|---|---:|",
    ]
    for key, row in sorted(report["pair_surface_matrix"].items()):
        lines.append(f"| `{row['pair_id']}` | `{row['surface_role']}` | {','.join(row['variants'])} | {row['rule_signal_count']} |")
    lines.extend([
        "",
        "注意：reflection 只表示 HTTP 响应回显 canary；没有浏览器执行 oracle，也没有 SQL/RCE/SSRF/XXE exploit 确认。",
        "",
        f"Catalog：`{report['catalog']['path']}`",
        f"训练协议：`{report['protocol_path']}`",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


async def _run() -> dict[str, Any]:
    specs = default_pikachu_paired_specs()
    records = await PikachuReplayCollector().collect_many(specs)
    catalog = write_catalog(CATALOG_PATH, _catalog(records))
    normalized = flatten_catalog(load_catalog(CATALOG_PATH))
    pair_surface: dict[str, dict[str, Any]] = {}
    for row in normalized:
        pair = row["pair"]
        key = f"{pair['pair_id']}::{pair['surface_role']}"
        entry = pair_surface.setdefault(key, {
            "pair_id": pair["pair_id"],
            "surface_role": pair["surface_role"],
            "variants": [],
            "rule_signal_count": 0,
        })
        if pair["variant"] not in entry["variants"]:
            entry["variants"].append(pair["variant"])
        entry["rule_signal_count"] += int(bool(row.get("rule_ir_result", False)))
    pair_ids = {row["pair"]["pair_id"] for row in normalized}
    variants = {row["pair"]["variant"] for row in normalized}
    surfaces = {row["pair"]["surface_role"] for row in normalized}
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pikachu-pair-invariance-report-v1",
        "target": {
            "base_url": PIKACHU_BASE_URL,
            "container_image_digest": PIKACHU_IMAGE_DIGEST,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "fresh_reset_per_probe": False,
        },
        "pair_design": {
            "variants": ["plain", "url_percent", "html_entity", "double_html_entity"],
            "pair_metadata_hidden_from_decoder": True,
            "cross_surface": True,
            "same_family": True,
        },
        "capture": {
            "sample_count": len(normalized),
            "pair_count": len(pair_ids),
            "surface_count": len(surfaces),
            "variant_count": len(variants),
            "all_samples_local": all(row["payload"]["target"] == PIKACHU_BASE_URL for row in normalized),
            "all_samples_non_exploit": all(bool(row["payload"]["safety"]["does_not_execute"]) for row in normalized),
        },
        "variant_counts": dict(Counter(row["pair"]["variant"] for row in normalized)),
        "surface_counts": dict(Counter(row["pair"]["surface_role"] for row in normalized)),
        "pair_surface_matrix": pair_surface,
        "matrix": _matrix(normalized),
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_sha256": catalog["catalog_sha256"],
            "source_count": len(catalog["sources"]),
            "sample_count": len(normalized),
        },
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
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
        "variant_counts": report["variant_counts"],
        "surface_counts": report["surface_counts"],
        "catalog": report["catalog"],
        "report": report["report_path"],
        "markdown": str(MARKDOWN_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
