"""Build a safe, metadata-only registry of local lab data sources.

This is a read-only inventory step for PG-24.  It deliberately does not copy
challenge paths, raw probes, response bodies, cookies, or evaluator labels
into the registry.  The registry is used to plan source/target-disjoint
training and evaluation; it is not itself a training corpus.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.payload_catalog import flatten_catalog, load_catalog  # noqa: E402


PROTOCOL_ID = "pg-pk-24-cross-lab-registry-v1"
REPORT_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_24_cross_lab_registry_protocol_v1.json"


def _read(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def _pikachu_entry() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    catalogs = ["pikachu_counterfactual_catalog_v1.json", "pikachu_payload_catalog_v1.json"]
    for name in catalogs:
        rows.extend(flatten_catalog(load_catalog(ROOT / "research" / name)))
    by_id = {str(row.get("sample_id")): row for row in rows}
    rows = list(by_id.values())
    sources = sorted({str(row.get("source_id", "")) for row in rows})
    families = sorted({str((row.get("semantic") or {}).get("family", "")) for row in rows})
    surfaces = sorted({str((row.get("semantic") or {}).get("surface", "")) for row in rows})
    oracles = sorted({str((row.get("semantic") or {}).get("expected_oracle", "")) for row in rows})
    return {
        "target_id": "pikachu",
        "app_family": "pikachu",
        "artifacts": catalogs,
        "source_ids": sources,
        "sample_count": len(rows),
        "family_set": families,
        "surface_count": len(surfaces),
        "oracle_set": oracles,
        "source_instance_count": len(sources),
        "training_role": "authorized_catalog_training_and_holdout",
        "training_eligible": True,
        "label_authority": "bounded_local_oracle_projection",
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "script_execution": False,
            "database_write": False,
            "raw_body_stored": False,
        },
    }


def _juice_entry() -> dict[str, Any]:
    shadow = _read("juice_shop_loop_12_shadow_replay_runs.json")
    cross = _read("juice_shop_cross_app_shadow_v1.json")
    hidden = _read("juice_shop_loop_12_hidden_matrix_runs_v6.json")
    runs = shadow.get("runs") or {}
    shadow_rows = []
    for value in runs.values():
        shadow_rows.extend(value.get("shadow_observations") or [])
    hidden_summary: dict[str, Any] = {}
    for name, value in (hidden.get("runs") or {}).items():
        family_hits = value.get("family_hits") or {}
        hidden_summary[name] = {
            "shadow_action_count": int(value.get("shadow_action_count", 0)),
            "shadow_row_count": len(value.get("shadow_rows") or {}),
            "hidden_family_hit_count": int(value.get("hidden_family_hit_count", 0)),
            "family_names_observed": sorted(str(key) for key in family_hits.keys()),
            "batch_not_independent_episodes": bool(value.get("batch_not_independent_episodes", False)),
        }
    target = cross.get("target") or {}
    return {
        "target_id": "juice_shop_loop12",
        "app_family": "juice_shop",
        "artifacts": [
            "juice_shop_loop_12_catalog_v3.json",
            "juice_shop_loop_12_shadow_replay_runs.json",
            "juice_shop_loop_12_hidden_matrix_runs_v6.json",
            "juice_shop_cross_app_shadow_v1.json",
        ],
        "container_image": target.get("container_image"),
        "sample_count": int(cross.get("sample_count", 0)),
        "shadow_observation_count": len(shadow_rows),
        "ood_sample_count": int(cross.get("ood_gate_abstain_count", 0)),
        "ood_abstain_rate": float(cross.get("ood_gate_abstain_rate", 0.0)),
        "hidden_matrix_runs": hidden_summary,
        "training_role": "evaluation_only_until_canonical_safe_catalog_is_collected",
        "training_eligible": False,
        "label_authority": "engineering_shadow_or_hidden_evaluator_diagnostic",
        "safety": cross.get("safety") or {},
    }


def _sql_entry() -> dict[str, Any]:
    report = _read("pg_pk_09_sql_differential_v1.json")
    return {
        "target_id": "sql_differential_fixture",
        "app_family": "synthetic_sql_fixture",
        "artifacts": ["pg_pk_09_sql_differential_v1.json"],
        "sample_count": int(report.get("sample_count", 0)),
        "oracle_revalidated_sample_count": int(report.get("oracle_revalidated_sample_count", 0)),
        "oracle_revalidated_pair_count": int(report.get("oracle_revalidated_pair_count", 0)),
        "modalities": sorted((report.get("modality_summary") or {}).keys()),
        "fixture_source_hash": (report.get("target") or {}).get("fixture_source_sha256"),
        "training_role": "evaluation_only_until_source_catalog_is_attested",
        "training_eligible": False,
        "label_authority": "synthetic_sql_oracle_diagnostic",
        "safety": report.get("safety") or {},
    }


def _logic_entry() -> dict[str, Any]:
    report = _read("pg_pk_10_logic_access_v1.json")
    target = report.get("target") or {}
    return {
        "target_id": "logic_access_fixture",
        "app_family": "synthetic_logic_access_fixture",
        "artifacts": ["pg_pk_10_logic_access_v1.json"],
        "sample_count": int(report.get("sample_count", 0)),
        "target_count": int(target.get("target_count", 0)),
        "oracle_revalidated_sample_count": int(report.get("oracle_revalidated_sample_count", 0)),
        "counterfactual_count": int(report.get("counterfactual_count", 0)),
        "fixture_source_hash": target.get("fixture_source_sha256"),
        "training_role": "evaluation_only_until_source_catalog_is_attested",
        "training_eligible": False,
        "label_authority": "typed_logic_oracle_diagnostic",
        "safety": report.get("safety") or {},
    }


def _heterogeneous_entry() -> dict[str, Any]:
    report = _read("pg_pk_12_heterogeneous_surface_v1.json")
    target = report.get("target") or {}
    return {
        "target_id": "heterogeneous_surface_fixture",
        "app_family": "heterogeneous_surface_fixture",
        "artifacts": ["pg_pk_12_heterogeneous_surface_v1.json"],
        "sample_count": int(report.get("sample_count", 0)),
        "target_count": int(target.get("target_count", 0)),
        "seed_count": int(target.get("seed_count", 0)),
        "oracle_revalidated_pair_count": int(report.get("oracle_revalidated_pair_count", 0)),
        "fixture_source_hash": target.get("fixture_source_sha256"),
        "promotion_status": str((report.get("promotion") or {}).get("status", "unknown")),
        "training_role": "evaluation_only_until_pair_catalog_is_attested",
        "training_eligible": False,
        "label_authority": "bounded_heterogeneous_oracle_diagnostic",
        "safety": report.get("safety") or {},
    }


def _maze_entry() -> dict[str, Any]:
    report = _read("maze_lab_holdout_v1.json")
    rows = report.get("rows") or []
    modalities = sorted({str(row.get("modality", "unknown")) for row in rows})
    families = sorted({str(row.get("family", "unknown")) for row in rows})
    return {
        "target_id": "rule_maze",
        "app_family": "abstract_rule_maze",
        "artifacts": ["maze_lab_holdout_v1.json"],
        "lab_count": int((report.get("summary") or {}).get("lab_count", len(rows))),
        "holdout_lab_count": int((report.get("summary") or {}).get("holdout_lab_count", 0)),
        "family_set": families,
        "modalities": modalities,
        "holdout_observable_success": int((report.get("summary") or {}).get("holdout_observable_success", 0)),
        "holdout_evaluator_confirmed": int((report.get("summary") or {}).get("holdout_evaluator_confirmed", 0)),
        "training_role": "protocol_and_oracle_evaluation_only",
        "training_eligible": False,
        "label_authority": "abstract_maze_observation_without_evaluator_confirmation",
        "safety": {
            "external_network": bool(report.get("external_network", True)),
            "browser_script_executed": bool(report.get("browser_script_executed", True)),
            "database_executed": bool(report.get("database_executed", True)),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-24 跨靶场数据注册表",
        "",
        "该表只登记本地实验数据源的边界和质量，不复制原始 payload、响应正文、challenge key 或 evaluator 标签。Pikachu 是当前唯一满足 Catalog 训练契约的来源；其他靶场先作为隔离评估源，待补齐授权 probe、oracle、fresh reset 和 source hash 后才能进入训练。",
        "",
        "| target | app family | samples/labs | instances | training role | eligible |",
        "|---|---|---:|---:|---|---:|",
    ]
    for item in report["targets"]:
        count = item.get("sample_count", item.get("lab_count", 0))
        instances = item.get("source_instance_count", item.get("target_count", 0))
        lines.append(
            f"| `{item['target_id']}` | `{item['app_family']}` | {count} | {instances} | "
            f"{item['training_role']} | {'yes' if item['training_eligible'] else 'no'} |"
        )
    lines.extend([
        "",
        "## 训练扩展条件",
        "",
        "1. 每个靶场至少有独立 source hash、container/image digest、fresh reset 记录和 loopback 范围。",
        "2. 每个样本保存 safe probe、编码、bounded oracle projection、Rule IR、evidence hash；不保存原始正文或凭据。",
        "3. 按靶场实例和来源隔离 train/validation/test；同一模板不同标签不能算新来源。",
        "4. 未满足条件的 Juice Shop、SQL、logic、maze 数据只能做 OOD/abstain 测试。",
        "",
        f"完整 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    targets = [_pikachu_entry(), _juice_entry(), _sql_entry(), _logic_entry(), _heterogeneous_entry(), _maze_entry()]
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-cross-lab-registry-v1",
        "read_only": True,
        "raw_probe_strings_stored": False,
        "evaluator_labels_stored": False,
        "targets": targets,
        "training_eligible_target_count": sum(bool(item["training_eligible"]) for item in targets),
        "evaluation_only_target_count": sum(not bool(item["training_eligible"]) for item in targets),
        "next_collection": [
            "juice_shop_safe_catalog_v1",
            "sql_fixture_source_attestation_v1",
            "logic_access_source_attestation_v1",
            "heterogeneous_surface_pair_catalog_v1",
        ],
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "targets": [
            {
                "target_id": item["target_id"],
                "sample_or_lab_count": item.get("sample_count", item.get("lab_count", 0)),
                "training_eligible": item["training_eligible"],
                "training_role": item["training_role"],
            }
            for item in targets
        ],
        "report": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

