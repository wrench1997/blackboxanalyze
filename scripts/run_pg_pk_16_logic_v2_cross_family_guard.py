"""PG-PK-16: logic/access surface holdout and cross-family guard.

The v2 maze is not used to teach the SQL head positive evidence.  It is a
family-negative, source-held-out surface: the SQL decoder and shared router
are observed for accidental ``injection`` routes while the logic decoder is
checked against the typed authorization/business/history oracle.  Every
request is a fresh, read-only loopback GET and every stored observation is a
bounded projection plus evidence hash.
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
from app.logic_access_decoder import LogicAccessDecoder, logic_access_model_feature_vector  # noqa: E402
from app.logic_access_fixture import LOGIC_ACCESS_ORACLES, logic_access_fixture_source_sha256  # noqa: E402
from app.logic_access_fixture_v2 import (  # noqa: E402
    LogicAccessV2Collector,
    default_logic_access_v2_specs,
    logic_access_v2_source_sha256,
    make_logic_access_v2_fixture_server,
)
from app.logic_access_oracle import revalidate_logic_access_pair  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.pikachu_active_controller import _fuse_shared_route, _projection_likelihood  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402
from app.sql_channel_decoder import SqlChannelDecoder, sql_channel_feature_vector  # noqa: E402


PROTOCOL_ID = "pg-pk-16-logic-v2-cross-family-guard-v1"
LOGIC_CHECKPOINT = ROOT / "artifacts" / "logic-access-decoder-pg-pk-10" / "logic_access_decoder.pt"
SQL_CHECKPOINT = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09" / "sql_channel_decoder.pt"
SHARED_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
V1_REPORT_PATH = ROOT / "research" / "pg_pk_10_logic_access_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_protocol_v1.json"
PRE_FIX_FAILURE_PATH = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_pre_fix_v1.json"
TARGETS = ((8812, "alpha", "logic_v2_alpha"), (8813, "beta", "logic_v2_beta"), (8814, "gamma", "logic_v2_gamma"))
SEEDS = (20510101, 20510107, 20510113)
MAX_REQUESTS = 20
V2_DATASET_ID = "logic_access_fixture_v2"


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _load_logic() -> tuple[LogicAccessDecoder, dict[str, Any]]:
    checkpoint = torch.load(LOGIC_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LogicAccessDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _load_sql() -> tuple[SqlChannelDecoder, torch.Tensor, torch.Tensor, float]:
    checkpoint = torch.load(SQL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SqlChannelDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model, mean, std, float(checkpoint.get("abstain_threshold", 0.80))


def _logic_decode_one(model: LogicAccessDecoder, checkpoint: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    features = torch.tensor([logic_access_model_feature_vector(record)], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model.decode((features - mean) / std, abstain_threshold=float(checkpoint.get("abstain_threshold", 0.80)), margin_threshold=float(checkpoint.get("margin_threshold", 0.10)), temperature=float(checkpoint.get("temperature", 1.0)))[0]


def _sql_decode(model: SqlChannelDecoder, mean: torch.Tensor, std: torch.Tensor, threshold: float, record: dict[str, Any]) -> dict[str, Any]:
    features = (torch.tensor([sql_channel_feature_vector(record)], dtype=torch.float32) - mean) / std
    return model.decode(features, abstain_threshold=threshold)[0]


def _logic_likelihood(output: dict[str, Any]) -> dict[str, float]:
    values = {family: 1.0 for family in DECODER_FAMILIES}
    candidate = str(output.get("candidate_family", ""))
    if candidate in values:
        values[candidate] = 5.0 if not output.get("abstained", True) else 0.45
    return values


def _collect(collector: LogicAccessV2Collector, spec: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(collector.collect(spec))


def _cross_family_row(record: dict[str, Any], *, sql_output: dict[str, Any], shared_route: dict[str, Any], seed: int) -> dict[str, Any]:
    projection = record.get("oracle_projection") or {}
    sql_candidate = sql_output.get("candidate_family") == "injection" and not sql_output.get("abstained", True)
    route_candidate = shared_route.get("candidate_route") == "injection" and not shared_route.get("abstained", True)
    return {
        "dataset_id": V2_DATASET_ID,
        "sampling_seed": seed,
        "target_instance_id": str(((record.get("evidence") or {}).get("reset") or {}).get("target_instance_id", "")),
        "family_under_test": str((record.get("semantic") or {}).get("family", "")),
        "oracle_positive_for_logic_family": bool(projection.get("positive")),
        "sql_candidate_injection": bool(sql_candidate),
        "sql_cross_family_false_positive": bool(sql_candidate),
        "shared_candidate_injection": bool(route_candidate),
        "evidence_hash": str((record.get("evidence") or {}).get("evidence_hash", "")),
        "source_hash": str(((record.get("evidence") or {}).get("reset") or {}).get("fixture_source_sha256", "")),
        "local_only": True,
    }


def _logic_ledger_row(record: dict[str, Any], *, seed: int, accepted: bool, false_positive: bool = False, status: str = "accepted") -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    reset = evidence.get("reset") or {}
    projection = record.get("oracle_projection") or {}
    oracle_name = str(projection.get("oracle_name", ""))
    family = "access_control" if oracle_name == LOGIC_ACCESS_ORACLES["access_control"] else "logic"
    return {
        "dataset_id": V2_DATASET_ID,
        "sampling_seed": seed,
        "target_instance_id": str(reset.get("target_instance_id", "")),
        "rule_key": f"{family}::typed_boundary",
        "accepted": bool(accepted),
        "oracle_revalidated": bool(accepted),
        "false_positive": bool(false_positive),
        "evaluation_status": status,
        "evidence_hash": str(evidence.get("evidence_hash", "")),
        "source_hash": str(reset.get("fixture_source_sha256", "")),
        "local_only": True,
    }


def _pair_result(pair_rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]], source_hash: str) -> dict[str, Any]:
    positive_projection = next((row.get("oracle_projection") or {} for row in pair_rows if bool((row.get("oracle_projection") or {}).get("positive"))), {})
    expected_family = "access_control" if str(positive_projection.get("oracle_name")) == LOGIC_ACCESS_ORACLES["access_control"] else "logic"
    expected_signal = str(positive_projection.get("oracle_signal", ""))
    oracle_name = str(positive_projection.get("oracle_name", ""))
    candidate_rows = [dict(row, candidate_family=predictions[row["sample_id"]]["candidate_family"]) for row in pair_rows]
    if bool(positive_projection.get("positive")):
        result = revalidate_logic_access_pair(candidate_rows, authorized_source_hash=source_hash, expected_family=expected_family, oracle_name=oracle_name, expected_signal=expected_signal)
    else:
        result = {"schema_version": "sift-logic-access-v2-counterfactual-pair-v1", "accepted": False, "reasons": ["counterfactual_oracle_not_positive"], "record_count": len(candidate_rows), "pair_id": str((pair_rows[0].get("pair") or {}).get("pair_id", "")), "expected_family": "control"}
    result["pair_id"] = str((pair_rows[0].get("pair") or {}).get("pair_id", ""))
    result["candidate_predictions"] = [{"sample_id": row["sample_id"], "candidate_family": predictions[row["sample_id"]]["candidate_family"], "confidence": predictions[row["sample_id"]]["confidence"], "abstained": predictions[row["sample_id"]]["abstained"]} for row in pair_rows]
    result["oracle_positive"] = bool(positive_projection.get("positive"))
    result["expected_family"] = expected_family
    return result


def _normalise_v1_ledger(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["dataset_id"] = "logic_access_fixture_v1"
        # PG-PK-10 predates the explicit source_hash ledger field.  Bind its
        # historical rows to the current immutable fixture source so a later
        # three-source promotion cannot count an unattributed dataset.
        copied.setdefault("source_hash", logic_access_fixture_source_sha256())
        result.append(copied)
    return result


def _run_episode(*, port: int, variant: str, dataset_id: str, seed: int, logic_model: LogicAccessDecoder, logic_checkpoint: dict[str, Any], sql_model: SqlChannelDecoder, sql_mean: torch.Tensor, sql_std: torch.Tensor, sql_threshold: float, shared_router: SharedRouterBridge, source_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    server = make_logic_access_v2_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg16-{port}-{seed}", daemon=True)
    thread.start()
    records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    cross_family: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        specs = default_logic_access_v2_specs(dataset_id=f"{dataset_id}-seed-{seed}", target=target, marker=f"pg16-{variant}-{seed}")
        ordered = random.Random(seed).sample(specs, len(specs))
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in ordered:
            pair_id = str((spec.get("pair") or {}).get("pair_id", ""))
            by_pair[pair_id].append(spec)
        for group in by_pair.values():
            group.sort(key=lambda item: 0 if (item.get("pair") or {}).get("variant") == "plain" else 1)
        screening = [group[0] for _, group in sorted(by_pair.items())]
        refinements: list[dict[str, Any]] = []
        belief = MultiStepBelief()
        collector = LogicAccessV2Collector(base_url=target, target_instance_id=f"{dataset_id}-seed-{seed}", source_hash=source_hash)

        def observe(spec: dict[str, Any], stage: str) -> None:
            record = _collect(collector, spec)
            logic_output = _logic_decode_one(logic_model, logic_checkpoint, record)
            sql_output = _sql_decode(sql_model, sql_mean, sql_std, sql_threshold, record)
            shared_route = shared_router.inspect(record)
            step = belief.observe(spec["path"], _fuse_shared_route(_logic_likelihood(logic_output), shared_route), evidence_hash=record["evidence"]["evidence_hash"])
            record["logic_decoder"] = logic_output
            record["sql_decoder"] = sql_output
            record["shared_router"] = shared_route
            records.append(record)
            cross_family.append(_cross_family_row(record, sql_output=sql_output, shared_route=shared_route, seed=seed))
            pair = spec.get("pair") or {}
            trace.append({"stage": stage, "pair_id": pair.get("pair_id", ""), "variant": pair.get("variant", ""), "logic_candidate": logic_output.get("candidate_family"), "logic_abstained": logic_output.get("abstained", True), "sql_candidate": sql_output.get("candidate_family"), "sql_abstained": sql_output.get("abstained", True), "shared_route": shared_route, "posterior": step["posterior"], "evidence_hash": record["evidence"]["evidence_hash"]})
            for candidate in by_pair.get(str(pair.get("pair_id", "")), [])[1:]:
                candidate_copy = dict(candidate)
                candidate_copy["rule_ir_decoder"] = {"probabilities": step["posterior"], "confidence": max(step["posterior"].values())}
                candidate_copy["surface_discriminator"] = {"probabilities": step["posterior"]}
                refinements.append(candidate_copy)

        for spec in screening:
            observe(spec, "screen")
        while refinements and len(records) < MAX_REQUESTS:
            selected = choose_active_probe([belief.choose_next_probe(refinements)])
            observe(selected, "refine")
            pair = selected.get("pair") or {}
            refinements = [row for row in refinements if not (str((row.get("pair") or {}).get("pair_id", "")) == str(pair.get("pair_id", "")) and str((row.get("pair") or {}).get("variant", "")) == str(pair.get("variant", "")))]

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str((record.get("pair") or {}).get("pair_id", ""))].append(record)
        predictions = {record["sample_id"]: record["logic_decoder"] for record in records}
        for pair_id, pair_rows in sorted(grouped.items()):
            if not pair_id:
                continue
            result = _pair_result(pair_rows, predictions, source_hash)
            pair_results.append(result)
            oracle_positive = bool(result.get("oracle_positive"))
            if result.get("accepted"):
                ledger.extend(_logic_ledger_row(row, seed=seed, accepted=True) for row in pair_rows)
            elif not oracle_positive:
                complete = len(pair_rows) == 2
                for row in pair_rows:
                    prediction = predictions[row["sample_id"]]
                    model_noncontrol = prediction.get("candidate_family") != "control" and not prediction.get("abstained", True)
                    ledger.append(_logic_ledger_row(row, seed=seed, accepted=False, false_positive=bool(complete and model_noncontrol), status="negative_control" if complete else "incomplete_pair"))
        stats = {
            "target": f"{variant}:{port}",
            "seed": seed,
            "request_count": len(records),
            "static_request_count": len(specs),
            "complete_pair_count": sum(len(value) == 2 for value in grouped.values()),
            "logic_model_only_accept_count": sum(int(row["logic_decoder"].get("family") is not None and not row["logic_decoder"].get("abstained", True)) for row in records),
            "logic_model_counterfactual_candidate_count": sum(int(row["logic_decoder"].get("candidate_family") != "control" and not row["logic_decoder"].get("abstained", True)) for row in records if not bool((row.get("oracle_projection") or {}).get("positive"))),
            "sql_cross_family_candidate_count": sum(int(row["sql_cross_family_false_positive"]) for row in cross_family),
            "shared_injection_route_count": sum(int(row["shared_candidate_injection"]) for row in cross_family),
            "logic_decoder_abstain_count": sum(int(row["logic_decoder"].get("abstained", True)) for row in records),
            "shared_router_abstain_count": sum(int(bool(row["shared_router"].get("abstained", True))) for row in records),
            "shared_router_ood_count": sum(int(bool(row["shared_router"].get("ood", False))) for row in records),
            "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
            "belief": belief.snapshot(),
            "trace": trace,
        }
        return ledger, cross_family, {"stats": stats, "pair_results": pair_results}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main() -> None:
    for required in (LOGIC_CHECKPOINT, SQL_CHECKPOINT, SHARED_CHECKPOINT, V1_REPORT_PATH):
        if not required.exists():
            raise FileNotFoundError(f"PG-PK-16 missing artifact: {required}")
    if REPORT_PATH.exists() and not PRE_FIX_FAILURE_PATH.exists():
        PRE_FIX_FAILURE_PATH.write_text(REPORT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    logic_model, logic_checkpoint = _load_logic()
    sql_model, sql_mean, sql_std, sql_threshold = _load_sql()
    shared_router = SharedRouterBridge(SHARED_CHECKPOINT, strict_ood=True)
    source_hash = logic_access_v2_source_sha256()
    ledger: list[dict[str, Any]] = []
    cross_family: list[dict[str, Any]] = []
    target_runs: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for port, variant, dataset_id in TARGETS:
        for seed in SEEDS:
            rows, cross_rows, outcome = _run_episode(port=port, variant=variant, dataset_id=dataset_id, seed=seed, logic_model=logic_model, logic_checkpoint=logic_checkpoint, sql_model=sql_model, sql_mean=sql_mean, sql_std=sql_std, sql_threshold=sql_threshold, shared_router=shared_router, source_hash=source_hash)
            ledger.extend(rows)
            cross_family.extend(cross_rows)
            target_runs.append(outcome["stats"])
            pair_results.extend(outcome["pair_results"])
    v1_report = json.loads(V1_REPORT_PATH.read_text(encoding="utf-8"))
    v1_ledger = _normalise_v1_ledger([dict(row) for row in (v1_report.get("promotion_ledger") or [])])
    combined = v1_ledger + ledger
    local_memory = {family: assess_memory_promotion(f"{family}::typed_boundary", [row for row in ledger if row["rule_key"] == f"{family}::typed_boundary"]) for family in ("access_control", "logic")}
    cross_memory = {family: assess_memory_promotion(f"{family}::typed_boundary", [row for row in combined if row["rule_key"] == f"{family}::typed_boundary"]) for family in ("access_control", "logic")}
    sql_candidates = sum(int(row["sql_cross_family_false_positive"]) for row in cross_family)
    shared_candidates = sum(int(row["shared_candidate_injection"]) for row in cross_family)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-16-logic-v2-cross-family-guard-report-v1",
        # The cross-family guard can pass while durable memory remains
        # quarantined: v2 is one independent source, not three datasets.
        "status": "pass" if sql_candidates == 0 and sum(int(result.get("accepted", False)) for result in pair_results) == 36 and sum(int(run["logic_model_counterfactual_candidate_count"]) for run in target_runs) == 0 else "diagnostic_only",
        "target": {"target_count": len(TARGETS), "variants": [variant for _, variant, _ in TARGETS], "seed_count": len(SEEDS), "seeds": list(SEEDS), "fixture_source_sha256": source_hash, "loopback_only": True, "external_network": False},
        "training_boundary": {"logic_decoder_checkpoint": str(LOGIC_CHECKPOINT.relative_to(ROOT)), "sql_decoder_checkpoint": str(SQL_CHECKPOINT.relative_to(ROOT)), "shared_router_checkpoint": str(SHARED_CHECKPOINT.relative_to(ROOT)), "v2_fixture_seen_during_training": False, "sql_positive_authority": False},
        "static_request_count": sum(int(run["static_request_count"]) for run in target_runs),
        "request_count": sum(int(run["request_count"]) for run in target_runs),
        "complete_pair_count": sum(int(run["complete_pair_count"]) for run in target_runs),
        "logic_model_only_accept_count": sum(int(run["logic_model_only_accept_count"]) for run in target_runs),
        "logic_model_counterfactual_candidate_count": sum(int(run["logic_model_counterfactual_candidate_count"]) for run in target_runs),
        "logic_decoder_abstain_count": sum(int(run["logic_decoder_abstain_count"]) for run in target_runs),
        "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pair_results),
        "sql_cross_family_candidate_count": sql_candidates,
        "sql_cross_family_false_positive_rate": sql_candidates / max(len(cross_family), 1),
        "shared_injection_route_count": shared_candidates,
        "shared_injection_route_rate": shared_candidates / max(len(cross_family), 1),
        "cross_family_guard": {"sql_decoder_positive_authority": False, "shared_router_positive_authority": False, "sql_candidate_on_logic_surface_is_false_positive": True, "status": "pass" if sql_candidates == 0 else "quarantine"},
        "target_runs": target_runs,
        "pair_results": pair_results,
        "cross_family_ledger": cross_family,
        "promotion_ledger": combined,
        "local_memory_promotion": local_memory,
        "cross_source_memory_promotion": cross_memory,
        "provenance": {"dataset_ids": sorted({str(row.get("dataset_id", "")) for row in combined}), "source_hashes": sorted({str(row.get("source_hash", "")) for row in combined if row.get("source_hash")}), "source_count": len({str(row.get("source_hash", "")) for row in combined if row.get("source_hash")}), "target_instance_count": len({str(row.get("target_instance_id", "")) for row in combined}), "evidence_hash_count": len({str(row.get("evidence_hash", "")) for row in combined})},
        "safety": {"local_only": True, "read_only_get": True, "external_network": False, "script_execution": False, "database_touched": False, "real_sleep_performed": False, "state_mutated": False, "raw_body_stored": False, "credentials_stored": False},
        "preserved_pre_fix_failure": str(PRE_FIX_FAILURE_PATH.relative_to(ROOT)) if PRE_FIX_FAILURE_PATH.exists() else None,
        "report_path": str(REPORT_PATH.relative_to(ROOT)),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-PK-16 logic/access v2 族外与跨族 guard\n\n"
        f"请求：{report['request_count']}/{report['static_request_count']}；complete pair：{report['complete_pair_count']}；logic oracle pair：{report['oracle_revalidated_pair_count']}。\n\n"
        f"logic decoder model-only：{report['logic_model_only_accept_count']}；logic control candidate：{report['logic_model_counterfactual_candidate_count']}；SQL decoder 在 logic surface 上的 injection candidate：{report['sql_cross_family_candidate_count']}；shared router injection route：{report['shared_injection_route_count']}。\n\n"
        f"SQL 跨族 guard：`{report['cross_family_guard']['status']}`；v2 本地 logic/access memory：" + ", ".join(f"{key}={value['status']}" for key, value in local_memory.items()) + "；v1+v2 跨 source：" + ", ".join(f"{key}={value['status']}" for key, value in cross_memory.items()) + "。\n\n"
        "v2 更换 route、query 词汇、JSON 字段和响应长度；SQL/shared 输出只作诊断 prior，不具备 logic/access 正向权威。\n",
        encoding="utf-8",
    )
    PROTOCOL_PATH.write_text(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "surfaces": ["/authorize", "/reward", "/commit", "/status", "renamed_query_vocabulary", "variant_response_shapes"],
        "targets": [f"{variant}:{port}" for port, variant, _ in TARGETS],
        "seeds": list(SEEDS),
        "budget": {"max_requests_per_target_seed": MAX_REQUESTS, "static_specs_per_target_seed": 20},
        "typed_positive_oracles": sorted(set(LOGIC_ACCESS_ORACLES.values())),
        "cross_family_guard": {"sql_candidate_on_any_logic_row_is_false_positive": True, "shared_router_positive_authority": False, "sql_positive_authority": False},
        "promotion": {"local_v2": report["local_memory_promotion"], "cross_source_v1_v2": report["cross_source_memory_promotion"]},
        "safety": report["safety"],
        "preserved_pre_fix_failure": report["preserved_pre_fix_failure"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "request_count": report["request_count"], "complete_pair_count": report["complete_pair_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "sql_cross_family_candidate_count": report["sql_cross_family_candidate_count"], "shared_injection_route_count": report["shared_injection_route_count"], "local_memory": {key: value["status"] for key, value in local_memory.items()}, "cross_source_memory": {key: value["status"] for key, value in cross_memory.items()}, "report": report["report_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
