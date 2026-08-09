"""Collect PG-03 local replay records and materialize Catalog/Rule IR data."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app
from app.maze_engine import validate_evidence
from app.payload_catalog import flatten_catalog, load_catalog, policy_candidate, write_catalog
from app.payload_grounding import SourceGroundedMemory
from app.replay_collector import LocalReplayCollector
from app.rule_ir import evaluate as evaluate_rule_ir


PROTOCOL_ID = "sift-pg03-local-replay-v1"
SOURCE_DATE = "2026-08-02"


FAMILY_CONFIG = {
    "xss": {
        "index": 1,
        "surface": "dom_sink",
        "expected_oracle": "controlled_detached_dom_v1",
        "expected_signal": "browser_sink_observed+dom_change",
        "path": "/api/maze/replay/dom",
    },
    "injection": {
        "index": 2,
        "surface": "sql_ast_boundary",
        "expected_oracle": "synthetic_sql_ast_differential_v1",
        "expected_signal": "controlled_differential+interpreter_boundary",
        "path": "/api/maze/replay/sql",
    },
    "access_control": {
        "index": 3,
        "surface": "lab_registry_access_surface",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "status_code_200",
        "path": "/api/maze/labs",
    },
    "url_redirect": {
        "index": 4,
        "surface": "lab_registry_redirect_surface",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "status_code_200",
        "path": "/api/maze/labs",
    },
    "logic": {
        "index": 5,
        "surface": "lab_registry_logic_surface",
        "expected_oracle": "synthetic_rule_surface_v1",
        "expected_signal": "status_code_200",
        "path": "/api/maze/labs",
    },
}


def _marker(index: int, suffix: str, variant: str) -> str:
    # Marker content must not encode the family index; otherwise a decoder
    # could memorize a provenance label instead of learning the response.
    return f"pg03-probe-{suffix}-{variant}"


def build_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for family, config in FAMILY_CONFIG.items():
        for suffix in ("a", "b"):
            source_id = f"pg03-source-{config['index']:02d}-{suffix}"
            marker = _marker(config["index"], suffix, "base")
            if family == "xss":
                inert = f'<span data-sift-marker="{marker}">{marker}</span>'
                specs.append({
                    "source_id": source_id,
                    "lab_id": "maze_dom_sink_replay",
                    "family": family,
                    "surface": config["surface"],
                    "expected_oracle": config["expected_oracle"],
                    "expected_signal": config["expected_signal"],
                    "path": config["path"],
                    "probe_kind": "inert_dom_markup",
                    "probe": inert,
                    "encoding": "none_inert_markup",
                    "marker": marker,
                    "params": {"value": inert, "marker": marker},
                    "expected": {"browser_sink_observed": True, "dom_change": True},
                })
                encoded = f'&amp;lt;span data-sift-marker="{marker}"&amp;gt;{marker}&amp;lt;/span&amp;gt;'
                specs.append({
                    "source_id": source_id,
                    "lab_id": "maze_dom_double_decode_replay",
                    "family": family,
                    "surface": "dom_double_decode",
                    "expected_oracle": config["expected_oracle"],
                    "expected_signal": "browser_sink_observed+dom_change+double_decode",
                    "path": config["path"],
                    "probe_kind": "encoded_dom_markup",
                    "probe": encoded,
                    "encoding": "html_entity_encode_depth_2",
                    "marker": marker,
                    "params": {
                        "value": encoded,
                        "marker": marker,
                        "transforms": "html_entity_decode,html_entity_decode",
                    },
                    "expected": {"browser_sink_observed": True, "dom_change": True, "requires_decode_depth": 2},
                })
            elif family == "injection":
                for fragment in ("operator_like", "syntax_error", "blind_boolean", "time_delay", "local_side_channel"):
                    specs.append({
                        "source_id": source_id,
                        "lab_id": f"maze_sql_{fragment}_replay",
                        "family": family,
                        "surface": config["surface"],
                        "expected_oracle": config["expected_oracle"],
                        "expected_signal": config["expected_signal"],
                        "path": config["path"],
                        "probe_kind": "sql_channel_class",
                        "probe": fragment,
                        "encoding": "abstract_sql_fragment_class",
                        "marker": marker,
                        "params": {"fragment_class": fragment},
                        "expected": {"requires_recheck": True, "channel": fragment},
                    })
            else:
                specs.append({
                    "source_id": source_id,
                    "lab_id": f"maze_{family}_registry_replay",
                    "family": family,
                    "surface": config["surface"],
                    "expected_oracle": config["expected_oracle"],
                    "expected_signal": config["expected_signal"],
                    "path": config["path"],
                    "probe_kind": "http_canary",
                    "probe": marker,
                    "encoding": "identifier_canary",
                    "marker": marker,
                    "params": {},
                    "expected": {"status_code": 200, "requires_recheck": True},
                })
    return specs


def _source_provenance(source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_type": "in_repo_synthetic",
        "origin": "app/replay_collector.py",
        "license": "in_repo_synthetic",
        "authorization": "workspace_local_only",
        "scope": ["http://127.0.0.1:3100"],
        "captured_at": SOURCE_DATE,
        "authorized_for": ["training", "local_replay", "holdout_evaluation"],
        "external_network": False,
        "evaluator_state_visible": False,
    }


def _materialize_catalog(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
    return {
        "schema_version": "sift-authorized-payload-catalog-v1",
        "catalog_id": "pg03-local-replay-catalog-v1",
        "sources": [
            {"provenance": _source_provenance(source_id), "samples": samples}
            for source_id, samples in sorted(by_source.items())
        ],
    }


def _train_memory(records: list[dict[str, Any]], *, seed: int = 20260802) -> SourceGroundedMemory:
    memory = SourceGroundedMemory(seed=seed)
    for record in records:
        # The collector already evaluated Rule IR against the bounded response
        # projection.  This is an observable local signal, not evaluator state.
        validate_evidence(record["evidence"])
        memory.observe(
            policy_candidate(record),
            status="observable_success" if record["rule_ir_result"] else "dead_end",
            evidence=record["evidence"],
            evaluator_confirmed=False,
        )
    return memory


def _split_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    train_source_a = [row for row in records if row["source_id"].endswith("-a")]
    test_source_b = [row for row in records if row["source_id"].endswith("-b")]
    train_families = {"xss", "injection", "access_control"}
    train_transfer = [row for row in train_source_a if row["semantic"]["family"] in train_families]
    test_transfer = [row for row in test_source_b if row["semantic"]["family"] in {"url_redirect", "logic"}]
    train_unseen = [row for row in train_source_a if row["semantic"]["family"] in {"xss", "injection"}]
    test_unseen = [row for row in test_source_b if row["semantic"]["family"] in {"access_control", "url_redirect", "logic"}]

    def coverage(train: list[dict[str, Any]], test: list[dict[str, Any]], split: str) -> dict[str, Any]:
        memory = _train_memory(train, seed=20260802)
        supported = set(memory.supported_features())
        covered = [row for row in test if row["structural_feature"] in supported]
        checkpoint = memory.checkpoint()
        return {
            "split": split,
            "train_sample_count": len(train),
            "test_sample_count": len(test),
            "supported_features": sorted(supported),
            "test_feature_coverage": round(len(covered) / len(test), 4) if test else 0.0,
            "fail_closed_abstention_rate": round(1.0 - (len(covered) / len(test)), 4) if test else 0.0,
            "memory_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        }

    return {
        "source_split_same_family": coverage(train_source_a, test_source_b, "source_split_same_family"),
        "family_holdout_structural_transfer": coverage(train_transfer, test_transfer, "family_holdout_structural_transfer"),
        "family_holdout_unseen_surface": coverage(train_unseen, test_unseen, "family_holdout_unseen_surface"),
    }


def _consistency(records: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[bool] = []
    statuses: dict[str, int] = defaultdict(int)
    for record in records:
        envelope = {
            "response": record["response_projection"],
            "oracle_projection": record["oracle_projection"],
        }
        result = bool(evaluate_rule_ir(record["rule_ir"], envelope))
        results.append(result == bool(record["rule_ir_result"]))
        statuses[str(record["response_projection"]["status_code"])] += 1
    return {
        "rule_ir_replay_consistency": round(sum(results) / len(results), 4) if results else 0.0,
        "response_status_counts": dict(sorted(statuses.items())),
        "all_rule_ir_consistent": all(results),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-03 Local Replay Collector",
        "",
        "采集器仅通过 `http://127.0.0.1:3100` 的只读 replay adapter 运行 GET probe；每条记录先 fresh reset，响应只保存 bounded projection、长度/哈希和 Rule IR，不保存原始 body。",
        "",
        f"样本数：{report['capture']['sample_count']}；来源数：{report['capture']['source_count']}；Rule IR 一致性：{report['capture']['consistency']['rule_ir_replay_consistency']:.2f}",
        "",
        "| split | train 样本 | test 样本 | feature coverage | fail-closed abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["splits"].values():
        lines.append(
            f"| {row['split']} | {row['train_sample_count']} | {row['test_sample_count']} | "
            f"{row['test_feature_coverage']:.2f} | {row['fail_closed_abstention_rate']:.2f} |"
        )
    lines.extend([
        "",
        "边界：这是本地 ASGI 应用的真实路由响应回放，不是公网数据，也不是 evaluator 确认。下一阶段可将同一 collector 接到用户明确授权的本地容器靶场。",
        "",
        f"原始 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    specs = build_specs()
    collector = LocalReplayCollector(app)
    records = asyncio.run(collector.collect_many(specs))
    catalog_path = ROOT / "research" / "payload_replay_catalog_v1.json"
    catalog = write_catalog(catalog_path, _materialize_catalog(records))
    normalized_records = flatten_catalog(load_catalog(catalog_path))
    splits = _split_metrics(normalized_records)
    consistency = _consistency(normalized_records)
    artifact_dir = ROOT / "artifacts" / "payload-grounding" / PROTOCOL_ID
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[str] = []
    for split_name, split in splits.items():
        if split_name == "source_split_same_family":
            train = [row for row in normalized_records if row["source_id"].endswith("-a")]
        elif split_name == "family_holdout_structural_transfer":
            train = [row for row in normalized_records if row["source_id"].endswith("-a") and row["semantic"]["family"] in {"xss", "injection", "access_control"}]
        else:
            train = [row for row in normalized_records if row["source_id"].endswith("-a") and row["semantic"]["family"] in {"xss", "injection"}]
        checkpoint = _train_memory(train).save(artifact_dir / f"{split_name}.json")
        checkpoint_paths.append(str((artifact_dir / f"{split_name}.json").relative_to(ROOT)))
        split["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]

    report_path = ROOT / "research" / "payload_replay_collector_v1.json"
    markdown_path = ROOT / "research" / "payload_replay_collector_v1.md"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg03-local-replay-report-v1",
        "collector": {
            "schema_version": "sift-local-replay-collector-v1",
            "base_url": "http://127.0.0.1:3100",
            "transport": "in_process_asgi",
            "fresh_reset_per_sample": True,
            "allowed_methods": ["GET"],
            "raw_body_stored": False,
            "credentials_stored": False,
            "external_network": False,
            "script_execution": False,
            "database_touched": False,
            "real_sleep_performed": False,
        },
        "catalog": {
            "path": str(catalog_path.relative_to(ROOT)),
            "catalog_sha256": catalog["catalog_sha256"],
            "source_count": len(catalog["sources"]),
            "sample_count": len(normalized_records),
        },
        "capture": {
            "sample_count": len(normalized_records),
            "source_count": len(catalog["sources"]),
            "consistency": consistency,
        },
        "splits": splits,
        "checkpoints": checkpoint_paths,
        "spec_count": len(specs),
        "evaluator_confirmation_count": 0,
        "public_corpus_ingested": False,
        "report_path": str(report_path.relative_to(ROOT)),
        "protocol_path": "research/payload_replay_collector_protocol_v1.json",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "catalog": report["catalog"],
        "capture": report["capture"],
        "splits": report["splits"],
        "report": str(report_path.relative_to(ROOT)),
        "markdown": str(markdown_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
