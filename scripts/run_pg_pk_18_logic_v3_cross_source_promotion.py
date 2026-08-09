"""PG-PK-18: third logic/access source and durable memory gate."""

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
from app.logic_access_fixture import LOGIC_ACCESS_ORACLES  # noqa: E402
from app.logic_access_fixture_v3 import (  # noqa: E402
    LogicAccessV3Collector,
    default_logic_access_v3_specs,
    logic_access_v3_source_sha256,
    make_logic_access_v3_fixture_server,
)
from app.logic_access_oracle import revalidate_logic_access_pair  # noqa: E402
from app.memory_promotion_gate import assess_memory_promotion  # noqa: E402
from app.pikachu_active_controller import _fuse_shared_route  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402


PROTOCOL_ID = "pg-pk-18-logic-v3-cross-source-promotion-v1"
LOGIC_CHECKPOINT = ROOT / "artifacts" / "logic-access-decoder-pg-pk-10" / "logic_access_decoder.pt"
SHARED_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
PG16_REPORT_PATH = ROOT / "research" / "pg_pk_16_logic_v2_cross_family_guard_v1.json"
REPORT_PATH = ROOT / "research" / "pg_pk_18_logic_v3_cross_source_promotion_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_18_logic_v3_cross_source_promotion_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_18_logic_v3_cross_source_promotion_protocol_v1.json"
PRE_FIX_FAILURE_PATH = ROOT / "research" / "pg_pk_18_logic_v3_pre_fix_quarantine_v1.json"
TARGETS = ((8815, "red", "logic_v3_red"), (8816, "blue", "logic_v3_blue"), (8817, "green", "logic_v3_green"))
SEEDS = (20710101, 20710107, 20710113)
MAX_REQUESTS = 20
V3_DATASET_ID = "logic_access_fixture_v3"


def _wait_ready(port: int) -> None:
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                break
        except OSError:
            threading.Event().wait(0.02)
    threading.Event().wait(0.10)


def _load_model() -> tuple[LogicAccessDecoder, dict[str, Any]]:
    checkpoint = torch.load(LOGIC_CHECKPOINT, map_location="cpu", weights_only=False)
    model = LogicAccessDecoder().eval()
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _decode(model: LogicAccessDecoder, checkpoint: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    raw = torch.tensor([logic_access_model_feature_vector(record)], dtype=torch.float32)
    mean = torch.tensor(checkpoint["normalisation_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["normalisation_std"], dtype=torch.float32).clamp_min(1e-4)
    return model.decode((raw - mean) / std, abstain_threshold=float(checkpoint.get("abstain_threshold", 0.80)), margin_threshold=float(checkpoint.get("margin_threshold", 0.10)), temperature=float(checkpoint.get("temperature", 1.0)))[0]


def _likelihood(output: dict[str, Any]) -> dict[str, float]:
    values = {family: 1.0 for family in DECODER_FAMILIES}
    candidate = str(output.get("candidate_family", ""))
    if candidate in values:
        values[candidate] = 5.0 if not output.get("abstained", True) else 0.45
    return values


def _ledger_row(record: dict[str, Any], *, seed: int, accepted: bool, false_positive: bool = False, status: str = "accepted") -> dict[str, Any]:
    evidence = record.get("evidence") or {}
    reset = evidence.get("reset") or {}
    projection = record.get("oracle_projection") or {}
    oracle = str(projection.get("oracle_name", ""))
    family = "access_control" if oracle == LOGIC_ACCESS_ORACLES["access_control"] else "logic"
    return {"dataset_id": V3_DATASET_ID, "sampling_seed": seed, "target_instance_id": str(reset.get("target_instance_id", "")), "rule_key": f"{family}::typed_boundary", "accepted": bool(accepted), "oracle_revalidated": bool(accepted), "false_positive": bool(false_positive), "evaluation_status": status, "evidence_hash": str(evidence.get("evidence_hash", "")), "source_hash": str(reset.get("fixture_source_sha256", "")), "local_only": True}


def _run_episode(*, port: int, variant: str, dataset_id: str, seed: int, model: LogicAccessDecoder, checkpoint: dict[str, Any], shared_router: SharedRouterBridge, source_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    server = make_logic_access_v3_fixture_server(port=port, variant=variant)
    thread = threading.Thread(target=server.serve_forever, name=f"pg18-{port}-{seed}", daemon=True)
    thread.start()
    records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    try:
        _wait_ready(port)
        target = f"http://127.0.0.1:{port}"
        specs = default_logic_access_v3_specs(dataset_id=f"{dataset_id}-seed-{seed}", target=target, marker=f"pg18-{variant}-{seed}")
        ordered = random.Random(seed).sample(specs, len(specs))
        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in ordered:
            by_pair[str((spec.get("pair") or {}).get("pair_id", ""))].append(spec)
        for group in by_pair.values():
            group.sort(key=lambda item: 0 if (item.get("pair") or {}).get("variant") == "plain" else 1)
        screening = [group[0] for _, group in sorted(by_pair.items())]
        refinements: list[dict[str, Any]] = []
        belief = MultiStepBelief()
        collector = LogicAccessV3Collector(base_url=target, target_instance_id=f"{dataset_id}-seed-{seed}", source_hash=source_hash)

        def observe(spec: dict[str, Any], stage: str) -> None:
            record = asyncio.run(collector.collect(spec))
            output = _decode(model, checkpoint, record)
            shared_route = shared_router.inspect(record)
            step = belief.observe(spec["path"], _fuse_shared_route(_likelihood(output), shared_route), evidence_hash=record["evidence"]["evidence_hash"])
            record["logic_decoder"] = output
            record["shared_router"] = shared_route
            records.append(record)
            pair = spec.get("pair") or {}
            trace.append({"stage": stage, "pair_id": pair.get("pair_id", ""), "variant": pair.get("variant", ""), "candidate_family": output.get("candidate_family"), "abstained": output.get("abstained", True), "shared_router": shared_route, "posterior": step["posterior"], "evidence_hash": record["evidence"]["evidence_hash"]})
            for candidate in by_pair.get(str(pair.get("pair_id", "")), [])[1:]:
                candidate_copy = dict(candidate)
                candidate_copy["rule_ir_decoder"] = {"probabilities": step["posterior"], "confidence": max(step["posterior"].values())}
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
            positive_projection = next((row.get("oracle_projection") or {} for row in pair_rows if bool((row.get("oracle_projection") or {}).get("positive"))), {})
            expected_family = "access_control" if str(positive_projection.get("oracle_name")) == LOGIC_ACCESS_ORACLES["access_control"] else "logic"
            result = revalidate_logic_access_pair(
                [dict(row, candidate_family=predictions[row["sample_id"]]["candidate_family"]) for row in pair_rows],
                authorized_source_hash=source_hash,
                expected_family=expected_family,
                oracle_name=str(positive_projection.get("oracle_name", "")),
                expected_signal=str(positive_projection.get("oracle_signal", "")),
            ) if bool(positive_projection.get("positive")) else {"accepted": False, "reasons": ["counterfactual_oracle_not_positive"], "record_count": len(pair_rows)}
            result.update({"pair_id": pair_id, "target": f"{variant}:{port}", "seed": seed, "oracle_positive": bool(positive_projection.get("positive")), "candidate_predictions": [{"sample_id": row["sample_id"], "candidate_family": predictions[row["sample_id"]]["candidate_family"], "confidence": predictions[row["sample_id"]]["confidence"], "abstained": predictions[row["sample_id"]]["abstained"]} for row in pair_rows]})
            pairs.append(result)
            if result.get("accepted"):
                ledger.extend(_ledger_row(row, seed=seed, accepted=True) for row in pair_rows)
            elif not bool(positive_projection.get("positive")):
                complete = len(pair_rows) == 2
                for row in pair_rows:
                    output = predictions[row["sample_id"]]
                    noncontrol = output.get("candidate_family") != "control" and not output.get("abstained", True)
                    ledger.append(_ledger_row(row, seed=seed, accepted=False, false_positive=bool(complete and noncontrol), status="negative_control" if complete else "incomplete_pair"))
        stats = {"target": f"{variant}:{port}", "seed": seed, "request_count": len(records), "static_request_count": len(specs), "complete_pair_count": sum(len(value) == 2 for value in grouped.values()), "oracle_revalidated_pair_count": sum(int(result.get("accepted", False)) for result in pairs), "model_only_accept_count": sum(int(row["logic_decoder"].get("family") is not None and not row["logic_decoder"].get("abstained", True)) for row in records), "counterfactual_model_candidate_count": sum(int(row["logic_decoder"].get("candidate_family") != "control" and not row["logic_decoder"].get("abstained", True)) for row in records if not bool((row.get("oracle_projection") or {}).get("positive"))), "decoder_abstain_count": sum(int(row["logic_decoder"].get("abstained", True)) for row in records), "shared_router_abstain_count": sum(int(bool(row["shared_router"].get("abstained", True))) for row in records), "shared_router_ood_count": sum(int(bool(row["shared_router"].get("ood", False))) for row in records), "belief": belief.snapshot(), "trace": trace}
        return ledger, stats, pairs
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def main() -> None:
    if not LOGIC_CHECKPOINT.exists() or not SHARED_CHECKPOINT.exists() or not PG16_REPORT_PATH.exists():
        raise FileNotFoundError("PG-PK-18 requires logic/shared checkpoints and PG-PK-16 report")
    if REPORT_PATH.exists() and not PRE_FIX_FAILURE_PATH.exists():
        PRE_FIX_FAILURE_PATH.write_text(REPORT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    model, checkpoint = _load_model()
    shared_router = SharedRouterBridge(SHARED_CHECKPOINT, strict_ood=True)
    source_hash = logic_access_v3_source_sha256()
    ledger: list[dict[str, Any]] = []
    target_runs: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for port, variant, dataset_id in TARGETS:
        for seed in SEEDS:
            rows, stats, pairs = _run_episode(port=port, variant=variant, dataset_id=dataset_id, seed=seed, model=model, checkpoint=checkpoint, shared_router=shared_router, source_hash=source_hash)
            ledger.extend(rows)
            target_runs.append(stats)
            pair_results.extend(pairs)
    pg16 = json.loads(PG16_REPORT_PATH.read_text(encoding="utf-8"))
    combined = [dict(row) for row in (pg16.get("promotion_ledger") or [])] + ledger
    local_memory = {family: assess_memory_promotion(f"{family}::typed_boundary", [row for row in ledger if row["rule_key"] == f"{family}::typed_boundary"]) for family in ("access_control", "logic")}
    cross_memory = {family: assess_memory_promotion(f"{family}::typed_boundary", [row for row in combined if row["rule_key"] == f"{family}::typed_boundary"]) for family in ("access_control", "logic")}
    v3_positive_pair_coverage = {
        "expected": 36,
        "accepted": sum(int(result.get("accepted", False)) for result in pair_results),
        "ratio": sum(int(result.get("accepted", False)) for result in pair_results) / 36.0,
        "passed": sum(int(result.get("accepted", False)) for result in pair_results) == 36,
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-pg-pk-18-logic-v3-cross-source-promotion-report-v1",
        "status": "promote" if all(value["status"] == "promote" for value in cross_memory.values()) and v3_positive_pair_coverage["passed"] else "diagnostic_only",
        "target": {"target_count": len(TARGETS), "variants": [variant for _, variant, _ in TARGETS], "seed_count": len(SEEDS), "seeds": list(SEEDS), "fixture_source_sha256": source_hash, "loopback_only": True, "external_network": False},
        "training_boundary": {"logic_decoder_checkpoint": str(LOGIC_CHECKPOINT.relative_to(ROOT)), "shared_router_checkpoint": str(SHARED_CHECKPOINT.relative_to(ROOT)), "v3_fixture_seen_during_training": False, "positive_authority": False},
        "request_count": sum(int(run["request_count"]) for run in target_runs),
        "static_request_count": sum(int(run["static_request_count"]) for run in target_runs),
        "complete_pair_count": sum(int(run["complete_pair_count"]) for run in target_runs),
        "oracle_revalidated_pair_count": sum(int(run["oracle_revalidated_pair_count"]) for run in target_runs),
        "model_only_accept_count": sum(int(run["model_only_accept_count"]) for run in target_runs),
        "counterfactual_model_candidate_count": sum(int(run["counterfactual_model_candidate_count"]) for run in target_runs),
        "decoder_abstain_count": sum(int(run["decoder_abstain_count"]) for run in target_runs),
        "shared_router_abstain_count": sum(int(run["shared_router_abstain_count"]) for run in target_runs),
        "shared_router_ood_count": sum(int(run["shared_router_ood_count"]) for run in target_runs),
        "false_positive_ledger_row_count": sum(int(row.get("false_positive", False)) for row in combined),
        "v3_positive_pair_coverage": v3_positive_pair_coverage,
        "target_runs": target_runs,
        "pair_results": pair_results,
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
    MARKDOWN_PATH.write_text("# PG-PK-18 logic/access v3 三 source 晋升\n\n" + f"请求：{report['request_count']}/{report['static_request_count']}；complete pair：{report['complete_pair_count']}；typed oracle pair：{report['oracle_revalidated_pair_count']}；decoder abstain：{report['decoder_abstain_count']}；反事实 candidate：{report['counterfactual_model_candidate_count']}；false-positive ledger：{report['false_positive_ledger_row_count']}。\n\n" + f"v3 本地门：" + ", ".join(f"{key}={value['status']}" for key, value in local_memory.items()) + f"；v1+v2+v3 跨 source 门：" + ", ".join(f"{key}={value['status']}" for key, value in cross_memory.items()) + f"；source hash：{report['provenance']['source_count']}。\n\n" + "v3 使用新的 permit/benefit/finalize/heartbeat surface 与 red/blue/green response schema；共享 router 只作 prior，typed oracle 才是正证据。\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "targets": [f"{variant}:{port}" for port, variant, _ in TARGETS], "seeds": list(SEEDS), "budget": {"max_requests_per_target_seed": MAX_REQUESTS, "static_specs_per_target_seed": 20}, "positive_oracle": "synthetic_authorization_boundary_v1 + synthetic_business_invariant_v1 + synthetic_history_binding_v1", "promotion": {"local_v3": local_memory, "cross_source_v1_v2_v3": cross_memory}, "safety": report["safety"], "preserved_pre_fix_failure": report["preserved_pre_fix_failure"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"], "request_count": report["request_count"], "complete_pair_count": report["complete_pair_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "counterfactual_model_candidate_count": report["counterfactual_model_candidate_count"], "false_positive_ledger_row_count": report["false_positive_ledger_row_count"], "source_count": report["provenance"]["source_count"], "local_memory": {key: value["status"] for key, value in local_memory.items()}, "cross_source_memory": {key: value["status"] for key, value in cross_memory.items()}, "report": report["report_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
