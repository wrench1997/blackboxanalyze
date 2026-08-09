"""Run PG-PK-12 heterogeneous surface + encoding/seed holdout."""

from __future__ import annotations

import asyncio
import json
import random
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

from app.heterogeneous_surface_fixture import (  # noqa: E402
    HETERO_SURFACE_BASE_URL,
    HETERO_SURFACE_ORACLE,
    HeterogeneousSurfaceCollector,
    default_heterogeneous_surface_specs,
    heterogeneous_surface_source_sha256,
    make_heterogeneous_surface_fixture_server,
)
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.oracle_revalidation import revalidate_positive_pair  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402


PROTOCOL_ID = "pg-pk-12-heterogeneous-surface-holdout-v1"
CHECKPOINT_PATH = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_12_heterogeneous_surface_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_12_heterogeneous_surface_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_12_heterogeneous_surface_protocol_v1.json"
TARGETS = ((8800, "alpha", "hetero-eval-alpha"), (8801, "beta", "hetero-eval-beta"), (8802, "gamma", "hetero-eval-gamma"))
SEEDS = (20260831, 20260837, 20260843)
# The three loopback ports are target/variant instances of one fixture source,
# not three independent datasets.  Keeping this identity separate from the
# target label prevents the long-term memory gate from counting variants as
# independent evidence.
PROMOTION_DATASET_ID = "heterogeneous_surface_fixture_v1"


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _run_target(port: int, variant: str, dataset_id: str, bridge: SharedRouterBridge, source_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    server = make_heterogeneous_surface_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg12-{port}", daemon=True)
    thread.start()
    all_rows: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    stats = {
        "sample_count": 0,
        "model_only_xss_candidates": 0,
        "non_attribute_model_only_candidates": 0,
        "positive_abstains": 0,
        "shared_ood": 0,
        "oracle_scheduled_pairs": 0,
        "oracle_revalidated_pairs": 0,
        "oracle_revalidated_without_router_pairs": 0,
    }
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        for seed in SEEDS:
            specs = default_heterogeneous_surface_specs(dataset_id=f"{dataset_id}-seed-{seed}", target=target, marker=f"pg12-{variant}-{seed}")
            ordered = random.Random(seed).sample(specs, len(specs))
            collector = HeterogeneousSurfaceCollector(base_url=target, target_instance_id=f"{dataset_id}-{variant}-seed-{seed}", source_hash=source_hash)
            records = asyncio.run(collector.collect_many(ordered))
            predictions: dict[str, dict[str, Any]] = {}
            for row in records:
                prediction = bridge.inspect(row)
                predictions[row["sample_id"]] = prediction
                row["shared_router"] = prediction
                positive_surface = str((row.get("semantic") or {}).get("surface_role")) == "html_attribute"
                accepted_xss = bool(prediction.get("candidate_family") == "xss" and not prediction.get("abstained", True) and not prediction.get("ood", False))
                stats["model_only_xss_candidates"] += int(accepted_xss)
                stats["non_attribute_model_only_candidates"] += int(accepted_xss and not positive_surface)
                stats["positive_abstains"] += int(positive_surface and not accepted_xss)
                stats["shared_ood"] += int(bool(prediction.get("ood", False)))
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                grouped[str((row.get("pair") or {}).get("pair_id", ""))].append(row)
            for pair_id, pair_rows in sorted(grouped.items()):
                role = str((pair_rows[0].get("pair") or {}).get("surface_role", ""))
                router_candidate = all(
                    predictions[row["sample_id"]].get("candidate_family") == "xss"
                    and not predictions[row["sample_id"]].get("abstained", True)
                    and not predictions[row["sample_id"]].get("ood", False)
                    for row in pair_rows
                )
                # This is a family-specific active lane, not a model positive:
                # when the shared router abstains, the scheduler may still run
                # the safe XSS sink oracle.  Only that typed oracle can accept
                # a pair; the route is recorded separately for recall.
                candidate_rows = [dict(row, candidate_family="xss") for row in pair_rows]
                stats["oracle_scheduled_pairs"] += 1
                result = revalidate_positive_pair(
                    candidate_rows,
                    expected_family="xss",
                    oracle_name=HETERO_SURFACE_ORACLE,
                    authorized_source_hash=source_hash,
                    required_surface_role=role,
                    required_sink_kind="html_attribute",
                )
                result.update({
                    "pair_id": pair_id,
                    "surface_role": role,
                    "dataset_id": dataset_id,
                    "promotion_dataset_id": PROMOTION_DATASET_ID,
                    "seed": seed,
                    "shared_router_candidate": router_candidate,
                    "active_fallback": not router_candidate,
                })
                pair_results.append(result)
                stats["oracle_revalidated_pairs"] += int(result.get("accepted", False))
                stats["oracle_revalidated_without_router_pairs"] += int(result.get("accepted", False) and not router_candidate)
                if result.get("accepted"):
                    for row in pair_rows:
                        evidence = row.get("evidence") or {}
                        all_rows.append({
                            "dataset_id": PROMOTION_DATASET_ID,
                            "target_label": dataset_id,
                            "sampling_seed": seed,
                            "target_instance_id": str((evidence.get("reset") or {}).get("target_instance_id", "")),
                            "rule_key": "xss::html_attribute_reflection",
                            "accepted": True,
                            "oracle_revalidated": True,
                            "false_positive": False,
                            "evidence_hash": str(evidence.get("evidence_hash", "")),
                            "source_hash": source_hash,
                            "local_only": True,
                        })
                else:
                    # Keep one bounded negative observation per pair row in
                    # the same promotion ledger.  A shared-router candidate
                    # on a counterfactual surface is explicitly a false
                    # positive; an abstain is a clean negative, not a model
                    # success.  No response body or raw probe is retained.
                    for row in pair_rows:
                        evidence = row.get("evidence") or {}
                        prediction = predictions[row["sample_id"]]
                        model_candidate = bool(
                            prediction.get("candidate_family") == "xss"
                            and not prediction.get("abstained", True)
                            and not prediction.get("ood", False)
                        )
                        all_rows.append({
                            "dataset_id": PROMOTION_DATASET_ID,
                            "target_label": dataset_id,
                            "sampling_seed": seed,
                            "target_instance_id": str((evidence.get("reset") or {}).get("target_instance_id", "")),
                            "rule_key": "xss::html_attribute_reflection",
                            "accepted": False,
                            "oracle_revalidated": False,
                            "false_positive": model_candidate,
                            "evidence_hash": str(evidence.get("evidence_hash", "")),
                            "source_hash": source_hash,
                            "local_only": True,
                        })
            stats["sample_count"] += len(records)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return all_rows, pair_results, stats


def main() -> None:
    source_hash = heterogeneous_surface_source_sha256()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"shared router checkpoint missing: {CHECKPOINT_PATH}")
    bridge = SharedRouterBridge(CHECKPOINT_PATH, strict_ood=True)
    ledger_rows: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    target_summary: dict[str, dict[str, int]] = {}
    for port, variant, dataset_id in TARGETS:
        accepted_rows, target_pairs, stats = _run_target(port, variant, dataset_id, bridge, source_hash)
        ledger_rows.extend(accepted_rows)
        pair_results.extend(target_pairs)
        target_summary[f"{variant}:{port}"] = stats
    promotion = assess_memory_promotion("xss::html_attribute_reflection", ledger_rows)
    nonattribute_pairs = sum(int(result.get("surface_role") != "html_attribute" and not result.get("accepted")) for result in pair_results)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-12-heterogeneous-surface-report-v1",
        "status": "promote" if promotion["status"] == "promote" else "diagnostic_only",
        "target": {"target_count": len(TARGETS), "variants": [variant for _, variant, _ in TARGETS], "seed_count": len(SEEDS), "seeds": list(SEEDS), "fixture_source_sha256": source_hash, "loopback_only": True, "external_network": False},
        "training_boundary": {
            "shared_router_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "fixture_seen_during_training": False,
            "strict_ood": True,
            "positive_authority": False,
            "ood_surface_invariant_feature_count": int(bridge.ood_feature_mask.sum()) if bridge.ood_feature_mask is not None else None,
            "surface_specific_ood_dimensions_excluded": True,
        },
        "sample_count": sum(value["sample_count"] for value in target_summary.values()),
        "model_only_xss_candidate_count": sum(value["model_only_xss_candidates"] for value in target_summary.values()),
        "non_attribute_model_only_candidate_count": sum(value["non_attribute_model_only_candidates"] for value in target_summary.values()),
        "positive_shared_router_abstain_count": sum(value["positive_abstains"] for value in target_summary.values()),
        "shared_router_ood_count": sum(value["shared_ood"] for value in target_summary.values()),
        "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
        "oracle_scheduled_pair_count": sum(int(value["oracle_scheduled_pairs"]) for value in target_summary.values()),
        "oracle_revalidated_without_router_pair_count": sum(int(value["oracle_revalidated_without_router_pairs"]) for value in target_summary.values()),
        "router_gated_pair_count": sum(int(bool(result.get("shared_router_candidate", False))) for result in pair_results),
        "fallback_oracle_pair_count": sum(int(result.get("accepted", False) and result.get("active_fallback", False)) for result in pair_results),
        "non_attribute_rejected_pair_count": nonattribute_pairs,
        "target_summary": target_summary,
        "pair_results": pair_results,
        "promotion_ledger": ledger_rows,
        "promotion": promotion,
        "provenance": {"source_hashes": [source_hash], "dataset_ids": [PROMOTION_DATASET_ID], "target_labels": sorted(target_summary), "target_instances": len(TARGETS) * len(SEEDS), "sampling_seeds": list(SEEDS), "evidence_hash_count": len({row["evidence_hash"] for row in ledger_rows})},
        "safety": {"local_only": True, "read_only_get": True, "raw_body_stored": False, "script_execution": False, "database_touched": False, "state_mutated": False, "positive_authority": False},
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-12 异构表面 + 编码 + seed 留出\n\n"
        f"样本：{report['sample_count']}；target：{len(TARGETS)}；seed：{len(SEEDS)}；共享路由 XSS 候选：{report['model_only_xss_candidate_count']}；非属性误报候选：{report['non_attribute_model_only_candidate_count']}；正向 abstain：{report['positive_shared_router_abstain_count']}。\n\n"
        f"typed oracle pair：{report['oracle_revalidated_pair_count']}；其中路由 abstain 后由族特异 fallback 找回：{report['oracle_revalidated_without_router_pair_count']}；反事实表面拒绝：{report['non_attribute_rejected_pair_count']}；长期记忆门：`{promotion['status']}`。\n\n"
        "共享路由只提供候选/主动 prior；abstain 不等于停止探测。XML、JSON、文本和响应头回显 marker 不可替代 HTML attribute sink oracle；同一 fixture 的 variant 不计作独立数据集。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "surfaces": ["html_attribute", "html_text", "json_value", "xml_text", "plain_text", "response_header"],
        "targets": [f"{variant}:{port}" for port, variant, _ in TARGETS],
        "seeds": list(SEEDS),
        "positive_gate": {"oracle": HETERO_SURFACE_ORACLE, "sink_kind": "html_attribute", "pair": ["plain", "url_percent"], "source_hash": True, "evidence_hash": True},
        "shared_router": {"strict_ood": True, "positive_authority": False, "abstain_required": True, "abstain_fallback": "family_specific_xss_typed_oracle"},
        "promotion_dataset_identity": {"dataset_id": PROMOTION_DATASET_ID, "variant_ports_are_target_instances": True, "requires_independent_fixture_source": True},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "sample_count": report["sample_count"], "non_attribute_model_only_candidate_count": report["non_attribute_model_only_candidate_count"], "positive_shared_router_abstain_count": report["positive_shared_router_abstain_count"], "shared_router_ood_count": report["shared_router_ood_count"], "oracle_scheduled_pair_count": report["oracle_scheduled_pair_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "oracle_revalidated_without_router_pair_count": report["oracle_revalidated_without_router_pair_count"], "promotion": promotion["status"], "report": report["report_path"], "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
