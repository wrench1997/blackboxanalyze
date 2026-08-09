"""PG-PK-15: third SQL transport/source for durable Rule IR promotion.

This episode is intentionally boring from an exploit perspective: every
request is a bounded GET to an in-repo loopback fixture, and the positive
decision is made only by the typed synthetic AST oracle.  The decoder and
shared router provide priors and may abstain; neither is allowed to promote a
memory entry.  The important test is whether the v1/v2 Rule IR survives a
third independent endpoint/response implementation without false positives.
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
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_probe import choose_active_probe  # noqa: E402
from app.belief_state import DECODER_FAMILIES, MultiStepBelief  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.pikachu_active_controller import _fuse_shared_route, _projection_likelihood  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402
from app.sql_channel_decoder import SqlChannelDecoder, sql_channel_feature_vector  # noqa: E402
from app.sql_differential_fixture_v3 import (  # noqa: E402
    SQL_V3_ORACLE,
    SqlV3Collector,
    default_sql_v3_specs,
    make_sql_v3_fixture_server,
    sql_v3_source_sha256,
)
from app.sql_oracle_revalidation import revalidate_sql_pair  # noqa: E402


PROTOCOL_ID = "pg-pk-15-sql-v3-cross-source-promotion-v1"
DECODER_CHECKPOINT = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09" / "sql_channel_decoder.pt"
SHARED_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_15_sql_v3_cross_source_promotion_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_15_sql_v3_cross_source_promotion_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_15_sql_v3_cross_source_promotion_protocol_v1.json"
PG14_REPORT_PATH = ROOT / "research" / "pg_pk_14_sql_v2_active_generalization_v1.json"
PRE_FIX_FAILURE_PATH = ROOT / "research" / "pg_pk_15_sql_v3_pre_fix_quarantine_v1.json"
TARGETS = ((8809, "alpha", "sql_v3_alpha"), (8810, "beta", "sql_v3_beta"), (8811, "gamma", "sql_v3_gamma"))
SEEDS = (20410101, 20410107, 20410113)
MAX_REQUESTS = 15
SQL_V3_DATASET_ID = "sql_differential_fixture_v3"


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _load_decoder() -> tuple[SqlChannelDecoder, torch.Tensor, torch.Tensor, float]:
    checkpoint = torch.load(DECODER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SqlChannelDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model, mean, std, float(checkpoint.get("abstain_threshold", 0.80))


def _decode(model: SqlChannelDecoder, mean: torch.Tensor, std: torch.Tensor, threshold: float, record: dict[str, Any]) -> dict[str, Any]:
    features = (torch.tensor([sql_channel_feature_vector(record)], dtype=torch.float32) - mean) / std
    return model.decode(features, abstain_threshold=threshold)[0]


def _decoder_likelihood(output: dict[str, Any]) -> dict[str, float]:
    values = {family: 1.0 for family in DECODER_FAMILIES}
    values["injection"] = 5.0 if output.get("candidate_family") == "injection" and not output.get("abstained") else 0.45
    return values


def _collect(collector: SqlV3Collector, spec: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(collector.collect(spec))


def _ledger_row(record: dict[str, Any], *, seed: int, accepted: bool, false_positive: bool = False, evaluation_status: str = "accepted") -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    reset = evidence.get("reset") or {}
    return {
        "dataset_id": SQL_V3_DATASET_ID,
        "sampling_seed": seed,
        "target_instance_id": str(reset.get("target_instance_id", "")),
        "rule_key": "injection::synthetic_sql_channel",
        "accepted": bool(accepted),
        "oracle_revalidated": bool(accepted),
        "false_positive": bool(false_positive),
        "evaluation_status": evaluation_status,
        "evidence_hash": str(evidence.get("evidence_hash", "")),
        "source_hash": str(reset.get("fixture_source_sha256", "")),
        "local_only": True,
    }


def _run_episode(
    *,
    port: int,
    variant: str,
    dataset_id: str,
    seed: int,
    model: SqlChannelDecoder,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float,
    shared_router: SharedRouterBridge,
    source_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    server = make_sql_v3_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg15-{port}-{seed}", daemon=True)
    thread.start()
    records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        specs = default_sql_v3_specs(dataset_id=f"{dataset_id}-seed-{seed}", target=target, marker=f"pg15-{variant}-{seed}")
        ordered = random.Random(seed).sample(specs, len(specs))
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in ordered:
            pair_id = str((spec.get("pair") or {}).get("pair_id", f"control::{spec.get('lab_id', '')}"))
            by_pair[pair_id].append(spec)
        for group in by_pair.values():
            group.sort(key=lambda item: 0 if (item.get("pair") or {}).get("variant") == "plain" else 1)
        screening = [group[0] for _, group in sorted(by_pair.items())]
        refinements: list[dict[str, Any]] = []
        belief = MultiStepBelief()
        collector = SqlV3Collector(base_url=target, target_instance_id=f"{dataset_id}-seed-{seed}", source_hash=source_hash)

        def observe(spec: dict[str, Any], stage: str) -> None:
            record = _collect(collector, spec)
            output = _decode(model, mean, std, threshold, record)
            shared_route = shared_router.inspect(record)
            likelihood = _fuse_shared_route(_decoder_likelihood(output), shared_route)
            likelihood.update({key: likelihood[key] * value for key, value in _projection_likelihood(record).items()})
            step = belief.observe(spec["path"], likelihood, evidence_hash=record["evidence"]["evidence_hash"])
            record["sql_decoder"] = output
            record["shared_router"] = shared_route
            records.append(record)
            pair = spec.get("pair") or {}
            trace.append({"stage": stage, "pair_id": pair.get("pair_id", ""), "variant": pair.get("variant", ""), "decoder_candidate": output.get("candidate_family"), "decoder_abstained": output.get("abstained", True), "posterior": step["posterior"], "shared_router": shared_route, "evidence_hash": record["evidence"]["evidence_hash"]})
            for candidate in by_pair.get(str(pair.get("pair_id", "")), [])[1:]:
                candidate_copy = dict(candidate)
                candidate_copy["rule_ir_decoder"] = {"probabilities": likelihood, "confidence": max(likelihood.values()) / sum(likelihood.values())}
                candidate_copy["surface_discriminator"] = {"probabilities": likelihood}
                candidate_copy["model_score"] = float(output.get("confidence", 0.0))
                refinements.append(candidate_copy)

        for spec in screening[:MAX_REQUESTS]:
            observe(spec, "screen")
        while refinements and len(records) < MAX_REQUESTS:
            selected = choose_active_probe([belief.choose_next_probe(refinements)])
            observe(selected, "refine")
            pair = selected.get("pair") or {}
            refinements = [row for row in refinements if not (str((row.get("pair") or {}).get("pair_id", "")) == str(pair.get("pair_id", "")) and str((row.get("pair") or {}).get("variant", "")) == str(pair.get("variant", "")))]

        grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("pair"):
                grouped_records[str(record["pair"]["pair_id"])].append(record)
        for pair_id, pair_records in sorted(grouped_records.items()):
            predictions = [row.get("sql_decoder") or {} for row in pair_records]
            model_pair = len(pair_records) == 2 and all(prediction.get("candidate_family") == "injection" and not prediction.get("abstained", True) for prediction in predictions)
            candidate_records = [dict(record, candidate_family="injection") for record in pair_records]
            result = revalidate_sql_pair(candidate_records, authorized_source_hash=source_hash, oracle_name=SQL_V3_ORACLE)
            result.update({"pair_id": pair_id, "target": f"{variant}:{port}", "seed": seed, "decoder_pair_candidate": model_pair, "active_fallback": not model_pair})
            pair_results.append(result)
            if result.get("accepted"):
                ledger.extend(_ledger_row(record, seed=seed, accepted=True) for record in pair_records)
            else:
                complete_pair = len(pair_records) == 2
                for record, prediction in zip(pair_records, predictions):
                    model_candidate = prediction.get("candidate_family") == "injection" and not prediction.get("abstained", True)
                    # An incomplete active pair is not a negative oracle
                    # result.  Keep it as an unvalidated observation instead
                    # of inflating the false-positive rate.
                    ledger.append(_ledger_row(record, seed=seed, accepted=False, false_positive=bool(complete_pair and model_candidate), evaluation_status="negative_control" if complete_pair else "incomplete_pair"))
        stats = {
            "target": f"{variant}:{port}",
            "seed": seed,
            "request_count": len(records),
            "static_request_count": len(specs),
            "complete_pair_count": sum(len(value) == 2 for value in grouped_records.values()),
            "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
            "decoder_pair_candidate_count": sum(int(result.get("decoder_pair_candidate", False)) for result in pair_results),
            "fallback_oracle_pair_count": sum(int(result.get("accepted", False) and result.get("active_fallback", False)) for result in pair_results),
            "decoder_abstain_count": sum(int((row.get("sql_decoder") or {}).get("abstained", True)) for row in records),
            "shared_router_abstain_count": sum(int(bool((row.get("shared_router") or {}).get("abstained", True))) for row in records),
            "shared_router_ood_count": sum(int(bool((row.get("shared_router") or {}).get("ood", False))) for row in records),
            "incomplete_pair_count": sum(len(value) != 2 for value in grouped_records.values()),
            "belief": belief.snapshot(),
            "trace": trace,
        }
        return ledger, stats, pair_results
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main() -> None:
    if not DECODER_CHECKPOINT.exists() or not SHARED_CHECKPOINT.exists() or not PG14_REPORT_PATH.exists():
        raise FileNotFoundError("PG-PK-15 requires SQL/shared checkpoints and the PG-PK-14 ledger")
    # Preserve the first failing run before the report is replaced by a fix.
    if REPORT_PATH.exists() and not PRE_FIX_FAILURE_PATH.exists():
        PRE_FIX_FAILURE_PATH.write_text(REPORT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    model, mean, std, threshold = _load_decoder()
    bridge = SharedRouterBridge(SHARED_CHECKPOINT, strict_ood=True)
    source_hash = sql_v3_source_sha256()
    ledger: list[dict[str, Any]] = []
    target_runs: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for port, variant, dataset_id in TARGETS:
        for seed in SEEDS:
            rows, stats, pairs = _run_episode(port=port, variant=variant, dataset_id=dataset_id, seed=seed, model=model, mean=mean, std=std, threshold=threshold, shared_router=bridge, source_hash=source_hash)
            ledger.extend(rows)
            target_runs.append(stats)
            pair_results.extend(pairs)
    local_promotion = assess_memory_promotion("injection::synthetic_sql_channel", ledger)
    v2_report = json.loads(PG14_REPORT_PATH.read_text(encoding="utf-8"))
    combined_ledger = [dict(row) for row in (v2_report.get("promotion_ledger") or [])] + ledger
    cross_source_promotion = assess_memory_promotion("injection::synthetic_sql_channel", combined_ledger)
    source_hashes = sorted({str(row.get("source_hash", "")) for row in combined_ledger if row.get("source_hash")})
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-15-sql-v3-cross-source-promotion-report-v1",
        "status": "promote" if cross_source_promotion["status"] == "promote" else "diagnostic_only",
        "target": {"target_count": len(TARGETS), "variants": [variant for _, variant, _ in TARGETS], "seed_count": len(SEEDS), "seeds": list(SEEDS), "fixture_source_sha256": source_hash, "loopback_only": True, "external_network": False},
        "training_boundary": {"sql_decoder_checkpoint": str(DECODER_CHECKPOINT.relative_to(ROOT)), "shared_router_checkpoint": str(SHARED_CHECKPOINT.relative_to(ROOT)), "fixture_seen_during_training": False, "positive_authority": False},
        "static_request_count": sum(int(run["static_request_count"]) for run in target_runs),
        "request_count": sum(int(run["request_count"]) for run in target_runs),
        "complete_pair_count": sum(int(run["complete_pair_count"]) for run in target_runs),
        "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
        "decoder_pair_candidate_count": sum(int(result.get("decoder_pair_candidate", False)) for result in pair_results),
        "fallback_oracle_pair_count": sum(int(result.get("accepted", False) and result.get("active_fallback", False)) for result in pair_results),
        "decoder_abstain_count": sum(int(run["decoder_abstain_count"]) for run in target_runs),
        "shared_router_abstain_count": sum(int(run["shared_router_abstain_count"]) for run in target_runs),
        "shared_router_ood_count": sum(int(run["shared_router_ood_count"]) for run in target_runs),
        "incomplete_pair_count": sum(int(run["incomplete_pair_count"]) for run in target_runs),
        "false_positive_ledger_row_count": sum(int(row.get("false_positive", False)) for row in combined_ledger),
        "target_runs": target_runs,
        "pair_results": pair_results,
        "promotion_ledger": combined_ledger,
        "local_source_promotion": local_promotion,
        "promotion": cross_source_promotion,
        "cross_source_promotion": cross_source_promotion,
        "preserved_pre_fix_failure": str(PRE_FIX_FAILURE_PATH.relative_to(ROOT)) if PRE_FIX_FAILURE_PATH.exists() else None,
        "provenance": {"dataset_ids": sorted({str(row.get("dataset_id", "")) for row in combined_ledger}), "source_hashes": source_hashes, "source_count": len(source_hashes), "target_instance_count": len({str(row.get("target_instance_id", "")) for row in combined_ledger}), "sampling_seeds": sorted({str(row.get("sampling_seed", "")) for row in combined_ledger}), "evidence_hash_count": len({str(row.get("evidence_hash", "")) for row in combined_ledger})},
        "safety": {"local_only": True, "read_only_get": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "state_mutated": False, "raw_body_stored": False, "shared_router_positive_authority": False},
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-15 SQL v3 第三 source 跨源晋升\n\n"
        f"请求：{report['request_count']}/{report['static_request_count']}；target：{len(TARGETS)}；seed：{len(SEEDS)}；complete pair：{report['complete_pair_count']}。\n\n"
        f"typed oracle pair：{report['oracle_revalidated_pair_count']}；decoder 直接 pair：{report['decoder_pair_candidate_count']}；abstain 后 fallback：{report['fallback_oracle_pair_count']}；decoder abstain：{report['decoder_abstain_count']}；incomplete pair：{report['incomplete_pair_count']}；false positive ledger：{report['false_positive_ledger_row_count']}。\n\n"
        f"v3 本地门：`{report['local_source_promotion']['status']}`；v1+v2+v3 跨 source 门：`{report['promotion']['status']}`；source hash：{report['provenance']['source_count']}。\n\n"
        "v3 改变 endpoint、参数名、响应协议和抽象 fragment 命名；server 不执行 SQL、不访问数据库、不进行真实 sleep。decoder/shared router 只能提供 active prior，typed AST oracle 才能作为正证据。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "surfaces": ["/search?q=...", "renamed_sql_channel_modes", "changed_json_plain_headers"],
        "targets": [f"{variant}:{port}" for port, variant, _ in TARGETS],
        "seeds": list(SEEDS),
        "budget": {"max_requests_per_target_seed": MAX_REQUESTS, "static_specs_per_target_seed": 15},
        "positive_gate": {"oracle": SQL_V3_ORACLE, "pair": ["plain", "url_percent"], "source_hash": True, "evidence_hash": True},
        "active_policy": {"decoder_abstain_fallback": True, "shared_router_positive_authority": False, "oracle_only_positive": True},
        "promotion": {"local_v3": report["local_source_promotion"], "cross_source_v1_v2_v3": report["promotion"]},
        "preserved_pre_fix_failure": report["preserved_pre_fix_failure"],
        "safety": report["safety"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "request_count": report["request_count"], "complete_pair_count": report["complete_pair_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "decoder_pair_candidate_count": report["decoder_pair_candidate_count"], "fallback_oracle_pair_count": report["fallback_oracle_pair_count"], "decoder_abstain_count": report["decoder_abstain_count"], "false_positive_ledger_row_count": report["false_positive_ledger_row_count"], "source_count": report["provenance"]["source_count"], "report": report["report_path"], "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
