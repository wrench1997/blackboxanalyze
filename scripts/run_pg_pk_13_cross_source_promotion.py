"""PG-PK-13: cross-source surface transfer and audited memory promotion.

The run combines three independently implemented, authorized loopback
fixtures: PG-PK-12 v1, the structurally different heterogeneous v2 fixture,
and the older positive attribute fixture.  A shared-router abstain may still
schedule the XSS typed-oracle lane; only pair revalidation contributes to the
promotion ledger.
"""

from __future__ import annotations

import asyncio
import json
import random
import socket
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cross_app_positive_fixture import (  # noqa: E402
    FIXTURE_BASE_URL,
    FIXTURE_ORACLE,
    PositiveFixtureCollector,
    default_fixture_specs,
    fixture_source_sha256,
    make_server,
)
from app.heterogeneous_surface_fixture_v2 import (  # noqa: E402
    V2_ORACLE,
    HeterogeneousSurfaceV2Collector,
    default_heterogeneous_surface_v2_specs,
    heterogeneous_surface_v2_source_sha256,
    make_heterogeneous_surface_v2_fixture_server,
)
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.oracle_revalidation import revalidate_positive_pair  # noqa: E402
from app.promotion_runner import run_promotion_audit  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402


PROTOCOL_ID = "pg-pk-13-cross-source-promotion-v1"
RULE_KEY = "xss::html_attribute_reflection"
CHECKPOINT_PATH = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
PG12_REPORT_PATH = ROOT / "research" / "pg_pk_12_heterogeneous_surface_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_13_cross_source_promotion_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_13_cross_source_promotion_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_13_cross_source_promotion_protocol_v1.json"
PRE_FIX_FAILURE_PATH = ROOT / "research" / "pg_pk_13_cross_source_promotion_pre_fix_quarantine_v1.json"
V2_TARGETS = ((8803, "alpha", "heterogeneous_surface_fixture_v2"), (8804, "beta", "heterogeneous_surface_fixture_v2"), (8805, "gamma", "heterogeneous_surface_fixture_v2"))
V2_SEEDS = (20300901, 20300907, 20300913)
LEGACY_SEEDS = (20300919, 20300923)


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _normalise_v1_ledger() -> list[dict[str, Any]]:
    source = json.loads(PG12_REPORT_PATH.read_text(encoding="utf-8"))
    rows = []
    for original in source.get("promotion_ledger") or []:
        row = dict(original)
        # Keep the abstract rule key stable across fixture implementations;
        # the source hash remains the evidence boundary.
        row["rule_key"] = RULE_KEY
        row["dataset_id"] = "heterogeneous_surface_fixture_v1"
        rows.append(row)
    if not rows:
        raise RuntimeError("PG-PK-12 promotion ledger is missing; rerun PG-PK-12 first")
    return rows


def _load_v1_report() -> dict[str, Any]:
    return json.loads(PG12_REPORT_PATH.read_text(encoding="utf-8"))


def _ledger_row(
    row: dict[str, Any],
    *,
    dataset_id: str,
    target_label: str,
    seed: int,
    source_hash: str,
    accepted: bool,
    false_positive: bool = False,
) -> dict[str, Any]:
    evidence = row.get("evidence") or {}
    reset = evidence.get("reset") or {}
    return {
        "dataset_id": dataset_id,
        "target_label": target_label,
        "sampling_seed": seed,
        "target_instance_id": str(reset.get("target_instance_id", "")),
        "rule_key": RULE_KEY,
        "accepted": bool(accepted),
        "oracle_revalidated": bool(accepted),
        "false_positive": bool(false_positive),
        "evidence_hash": str(evidence.get("evidence_hash", "")),
        "source_hash": source_hash,
        "local_only": True,
    }


def _run_v2_target(port: int, variant: str, dataset_id: str, bridge: SharedRouterBridge, source_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    server = make_heterogeneous_surface_v2_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg13-v2-{port}", daemon=True)
    thread.start()
    ledger: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    stats = {"sample_count": 0, "shared_abstains": 0, "shared_ood": 0, "oracle_scheduled_pairs": 0, "oracle_revalidated_pairs": 0, "fallback_pairs": 0}
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        for seed in V2_SEEDS:
            specs = default_heterogeneous_surface_v2_specs(dataset_id=f"pg13-v2-{variant}-seed-{seed}", target=target, marker=f"pg13-v2-{variant}-{seed}")
            ordered = random.Random(seed).sample(specs, len(specs))
            rows = asyncio.run(HeterogeneousSurfaceV2Collector(base_url=target, target_instance_id=f"pg13-v2-{variant}-seed-{seed}", source_hash=source_hash).collect_many(ordered))
            predictions: dict[str, dict[str, Any]] = {}
            for row in rows:
                prediction = bridge.inspect(row)
                predictions[row["sample_id"]] = prediction
                stats["shared_abstains"] += int(bool(prediction.get("abstained", True)))
                stats["shared_ood"] += int(bool(prediction.get("ood", False)))
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[str((row.get("pair") or {}).get("pair_id", ""))].append(row)
            for pair_id, pair_rows in sorted(grouped.items()):
                router_candidate = all(not predictions[row["sample_id"]].get("abstained", True) and not predictions[row["sample_id"]].get("ood", False) and predictions[row["sample_id"]].get("candidate_family") == "xss" for row in pair_rows)
                stats["oracle_scheduled_pairs"] += 1
                result = revalidate_positive_pair(
                    [dict(row, candidate_family="xss") for row in pair_rows],
                    expected_family="xss",
                    oracle_name=V2_ORACLE,
                    authorized_source_hash=source_hash,
                    required_surface_role=str((pair_rows[0].get("pair") or {}).get("surface_role", "")),
                    required_sink_kind="html_attribute",
                )
                result.update({"pair_id": pair_id, "source": "v2", "variant": variant, "seed": seed, "shared_router_candidate": router_candidate, "active_fallback": not router_candidate})
                pairs.append(result)
                stats["oracle_revalidated_pairs"] += int(result.get("accepted", False))
                stats["fallback_pairs"] += int(result.get("accepted", False) and not router_candidate)
                if result.get("accepted"):
                    ledger.extend(_ledger_row(row, dataset_id=dataset_id, target_label=f"{variant}:{port}", seed=seed, source_hash=source_hash, accepted=True) for row in pair_rows)
                else:
                    ledger.extend(
                        _ledger_row(
                            row,
                            dataset_id=dataset_id,
                            target_label=f"{variant}:{port}",
                            seed=seed,
                            source_hash=source_hash,
                            accepted=False,
                            false_positive=bool(
                                predictions[row["sample_id"]].get("candidate_family") == "xss"
                                and not predictions[row["sample_id"]].get("abstained", True)
                                and not predictions[row["sample_id"]].get("ood", False)
                            ),
                        )
                        for row in pair_rows
                    )
            stats["sample_count"] += len(rows)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return ledger, pairs, stats


def _run_legacy_positive(bridge: SharedRouterBridge, source_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, name="pg13-legacy-positive", daemon=True)
    thread.start()
    ledger: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    stats = {"sample_count": 0, "shared_abstains": 0, "shared_ood": 0, "oracle_scheduled_pairs": 0, "oracle_revalidated_pairs": 0, "fallback_pairs": 0}
    try:
        _wait_ready(8790)
        for seed in LEGACY_SEEDS:
            specs = default_fixture_specs(marker_prefix=f"pg13-legacy-{seed}")
            ordered = random.Random(seed).sample(specs, len(specs))
            rows = asyncio.run(PositiveFixtureCollector(target_instance_id=f"pg13-legacy-seed-{seed}", source_hash=source_hash).collect_many(ordered))
            predictions = {row["sample_id"]: bridge.inspect(row) for row in rows}
            stats["shared_abstains"] += sum(int(bool(prediction.get("abstained", True))) for prediction in predictions.values())
            stats["shared_ood"] += sum(int(bool(prediction.get("ood", False))) for prediction in predictions.values())
            positive_rows = [row for row in rows if str((row.get("pair") or {}).get("pair_id", "")) == "fixture-pair-01"]
            stats["oracle_scheduled_pairs"] += 1
            router_candidate = all(not predictions[row["sample_id"]].get("abstained", True) and not predictions[row["sample_id"]].get("ood", False) and predictions[row["sample_id"]].get("candidate_family") == "xss" for row in positive_rows)
            result = revalidate_positive_pair(
                [dict(row, candidate_family="xss") for row in positive_rows],
                expected_family="xss",
                oracle_name=FIXTURE_ORACLE,
                authorized_source_hash=source_hash,
                required_surface_role="reflected_attribute",
                required_sink_kind="html_attribute",
            )
            result.update({"pair_id": "fixture-pair-01", "source": "legacy_positive", "seed": seed, "shared_router_candidate": router_candidate, "active_fallback": not router_candidate})
            pairs.append(result)
            stats["oracle_revalidated_pairs"] += int(result.get("accepted", False))
            stats["fallback_pairs"] += int(result.get("accepted", False) and not router_candidate)
            if result.get("accepted"):
                ledger.extend(_ledger_row(row, dataset_id="cross_app_positive_fixture_v1", target_label="legacy-positive:8790", seed=seed, source_hash=source_hash, accepted=True) for row in positive_rows)
            for row in rows:
                if row in positive_rows and result.get("accepted"):
                    continue
                prediction = predictions[row["sample_id"]]
                ledger.append(
                    _ledger_row(
                        row,
                        dataset_id="cross_app_positive_fixture_v1",
                        target_label="legacy-positive:8790",
                        seed=seed,
                        source_hash=source_hash,
                        accepted=False,
                        false_positive=bool(
                            prediction.get("candidate_family") == "xss"
                            and not prediction.get("abstained", True)
                            and not prediction.get("ood", False)
                        ),
                    )
                )
            stats["sample_count"] += len(rows)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    return ledger, pairs, stats


def main() -> None:
    if not CHECKPOINT_PATH.exists() or not PG12_REPORT_PATH.exists():
        raise FileNotFoundError("PG-PK-13 requires the shared checkpoint and a fresh PG-PK-12 report")
    bridge = SharedRouterBridge(CHECKPOINT_PATH, strict_ood=True)
    v1_report = _load_v1_report()
    v1_rows = _normalise_v1_ledger()
    v2_source_hash = heterogeneous_surface_v2_source_sha256()
    all_rows = list(v1_rows)
    pair_results: list[dict[str, Any]] = []
    target_summary: dict[str, dict[str, Any]] = {"heterogeneous_surface_fixture_v1": {"sample_count": int(v1_report.get("sample_count", 0)), "ledger_rows": len(v1_rows), "source_hash": str(v1_rows[0].get("source_hash", "")), "origin": "pg-pk-12"}}
    for port, variant, dataset_id in V2_TARGETS:
        rows, pairs, stats = _run_v2_target(port, variant, dataset_id, bridge, v2_source_hash)
        all_rows.extend(rows)
        pair_results.extend(pairs)
        target_summary[f"{dataset_id}:{variant}:{port}"] = stats
    legacy_source_hash = fixture_source_sha256()
    legacy_rows, legacy_pairs, legacy_stats = _run_legacy_positive(bridge, legacy_source_hash)
    all_rows.extend(legacy_rows)
    pair_results.extend(legacy_pairs)
    target_summary["cross_app_positive_fixture_v1:legacy-positive:8790"] = legacy_stats
    audit = run_promotion_audit(all_rows, rule_keys=[RULE_KEY])
    source_by_dataset = defaultdict(set)
    for row in all_rows:
        source_by_dataset[str(row.get("dataset_id", ""))].add(str(row.get("source_hash", "")))
    distinct_source_hashes = {value for values in source_by_dataset.values() for value in values if value}
    source_gate = {"required_distinct_source_hashes": 3, "observed_distinct_source_hashes": len(distinct_source_hashes), "passed": len(distinct_source_hashes) >= 3, "source_hashes_by_dataset": {key: sorted(value) for key, value in sorted(source_by_dataset.items())}}
    promotion = audit["promotion"][RULE_KEY]
    status = "promote" if audit["all_promoted"] and source_gate["passed"] else "quarantine"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-13-cross-source-promotion-report-v1",
        "status": status,
        "rule_key": RULE_KEY,
        "source_fixture_count": len(distinct_source_hashes),
        "target_summary": target_summary,
        "sample_count": sum(int(value.get("sample_count", 0)) for value in target_summary.values()),
        "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
        "fallback_oracle_pair_count": sum(int(result.get("accepted", False) and result.get("active_fallback", False)) for result in pair_results),
        "router_gated_pair_count": sum(int(result.get("accepted", False) and result.get("shared_router_candidate", False)) for result in pair_results),
        "promotion_ledger_row_count": len(all_rows),
        "counterfactual_ledger_row_count": sum(int(not row.get("accepted", False)) for row in all_rows),
        "false_positive_ledger_row_count": sum(int(row.get("false_positive", False)) for row in all_rows),
        "pair_results": pair_results,
        "promotion": promotion,
        "source_diversity_gate": source_gate,
        "promotion_audit": audit,
        "promotion_ledger": all_rows,
        "preserved_pre_fix_failure": str(PRE_FIX_FAILURE_PATH.relative_to(ROOT)) if PRE_FIX_FAILURE_PATH.exists() else None,
        "provenance": audit["provenance"],
        "safety": {"local_only": True, "loopback_only": True, "read_only_get": True, "external_network": False, "script_execution": False, "database_touched": False, "state_mutated": False, "raw_body_stored": False, "shared_router_positive_authority": False},
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-13 跨独立数据源表面迁移与 memory promotion\n\n"
        f"状态：`{status}`；独立 source hash：{len(distinct_source_hashes)}；样本：{report['sample_count']}；typed oracle pair：{report['oracle_revalidated_pair_count']}。\n\n"
        f"promotion ledger：{report['promotion_ledger_row_count']} 行（反事实：{report['counterfactual_ledger_row_count']}，false positive：{report['false_positive_ledger_row_count']}）；共享路由直接放行 pair：{report['router_gated_pair_count']}；abstain 后由 typed-oracle fallback 找回：{report['fallback_oracle_pair_count']}；source diversity gate：`{'pass' if source_gate['passed'] else 'quarantine'}`。\n\n"
        "只有跨独立 source hash、不同 target/seed、双编码 pair 和 typed sink oracle 同时通过，Rule IR 才允许进入长期 memory；共享 head 不拥有正向 authority。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "rule_key": RULE_KEY,
        "sources": ["pg-pk-12 heterogeneous v1", "pg-pk-13 heterogeneous v2", "cross-app positive v1"],
        "source_diversity_required": 3,
        "seeds": {"v2": list(V2_SEEDS), "legacy": list(LEGACY_SEEDS)},
        "positive_gate": "family_specific_typed_oracle_and_plain_url_percent_pair",
        "shared_router": {"strict_ood": True, "positive_authority": False, "abstain_fallback": True},
        "promotion": {"runner": "app/promotion_runner.py", "gate": "app/memory_promotion_gate.py", "source_hash_diversity": source_gate},
        "preserved_pre_fix_failure": str(PRE_FIX_FAILURE_PATH.relative_to(ROOT)) if PRE_FIX_FAILURE_PATH.exists() else None,
        "safety": report["safety"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": status, "source_fixture_count": len(distinct_source_hashes), "sample_count": report["sample_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "fallback_oracle_pair_count": report["fallback_oracle_pair_count"], "router_gated_pair_count": report["router_gated_pair_count"], "promotion": promotion["status"], "source_diversity_gate": source_gate, "report": report["report_path"], "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
