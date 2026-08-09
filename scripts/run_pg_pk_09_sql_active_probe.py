"""Run a budgeted SQL active-probe loop with the calibrated channel head."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_probe import choose_active_probe  # noqa: E402
from app.belief_state import DECODER_FAMILIES, MultiStepBelief  # noqa: E402
from app.pikachu_active_controller import _fuse_shared_route, _projection_likelihood  # noqa: E402
from app.shared_router_bridge import SharedRouterBridge  # noqa: E402
from app.sql_channel_decoder import SqlChannelDecoder, sql_channel_feature_vector  # noqa: E402
from app.sql_differential_fixture import (  # noqa: E402
    SQL_FIXTURE_BASE_URL,
    SQL_FIXTURE_ORACLE,
    SqlDifferentialCollector,
    default_sql_fixture_specs,
    make_sql_fixture_server,
    sql_fixture_source_sha256,
)
from app.sql_oracle_revalidation import revalidate_sql_pair  # noqa: E402


PROTOCOL_ID = "pg-pk-09-sql-active-probe-v1"
DECODER_CHECKPOINT = ROOT / "artifacts" / "sql-channel-decoder-pg-pk-09" / "sql_channel_decoder.pt"
REPORT_PATH = ROOT / "research" / "pg_pk_09_sql_active_probe_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg_pk_09_sql_active_probe_v1.md"
PROTOCOL_PATH = ROOT / "research" / "pg_pk_09_sql_active_probe_protocol_v1.json"
SHARED_ROUTER_CHECKPOINT = ROOT / "artifacts" / "shared-family-router-pg-pk-11" / "shared_family_router.pt"
MAX_REQUESTS = 13


def _wait_for_fixture() -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{SQL_FIXTURE_BASE_URL}/query?mode=plain", timeout=0.3).status_code == 200:
                return
        except Exception:
            time.sleep(0.02)
    raise RuntimeError("SQL active fixture did not start")


def _load_decoder() -> tuple[SqlChannelDecoder, torch.Tensor, torch.Tensor, float]:
    if not DECODER_CHECKPOINT.exists():
        raise RuntimeError("SQL channel decoder checkpoint is required before active probing")
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
    if output.get("candidate_family") == "injection" and not output.get("abstained"):
        values["injection"] = 5.0
    else:
        values["injection"] = 0.45
    return values


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-PK-09 SQL active probe",
        "",
        "screen 阶段每个 SQL surface 只发一个 plain probe；只有 decoder likelihood 与 bounded differential evidence 形成疑点时，才补发 url-percent pair。服务端仍不执行 SQL、不 sleep、不访问数据库。",
        "",
        f"静态请求数：{report['static_request_count']}；active 请求数：{report['request_count']}；节省：{report['static_request_count'] - report['request_count']}；pair 完整数：{report['complete_pair_count']}；oracle 复核 pair：{report['oracle_revalidated_pair_count']}。",
        f"最终 belief entropy：{report['belief']['entropy']:.4f}；SQL decoder abstain：{report['decoder_abstain_count']}。",
        f"共享路由 head abstain：{report.get('shared_router_abstain_count', 0)}；OOD：{report.get('shared_router_ood_count', 0)}；它只作为 active prior，不拥有正向 authority。",
        "",
        "| stage | pair | variant | decoder | posterior injection |",
        "|---|---|---|---|---:|",
    ]
    for row in report["selection_trace"]:
        lines.append(f"| `{row['stage']}` | `{row['pair_id']}` | `{row['variant']}` | `{row['decoder_candidate']}` | {row['posterior']['injection']:.3f} |")
    lines.extend([
        "",
        "active controller 只决定安全探针顺序，不直接宣布漏洞；Rule IR 仍需 pair、sink/AST oracle、fresh target 和 SHA-256 evidence 全部通过。",
        f"完整 JSON：`{report['report_path']}`",
        f"协议：`{report['protocol_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    server = make_sql_fixture_server()
    thread = threading.Thread(target=server.serve_forever, name="sift-pg-pk-09-active-fixture", daemon=True)
    thread.start()
    try:
        _wait_for_fixture()
        source_hash = sql_fixture_source_sha256()
        collector = SqlDifferentialCollector(target_instance_id=f"sql-active-{threading.get_ident()}", source_hash=source_hash)
        model, mean, std, threshold = _load_decoder()
        # Active ranking may use a lower *diagnostic* threshold than positive
        # emission; the bridge remains strict-OOD and has no positive authority.
        shared_router = SharedRouterBridge(SHARED_ROUTER_CHECKPOINT, strict_ood=True, abstain_threshold=0.60) if SHARED_ROUTER_CHECKPOINT.exists() else None
        specs = default_sql_fixture_specs()
        by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for spec in specs:
            pair_id = str((spec.get("pair") or {}).get("pair_id", f"control::{spec.get('lab_id', '')}"))
            by_surface[pair_id].append(spec)
        for group in by_surface.values():
            group.sort(key=lambda spec: 0 if (spec.get("pair") or {}).get("variant") == "plain" else 1)
        screening = [group[0] for _, group in sorted(by_surface.items())]
        records: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        belief = MultiStepBelief()
        refinements: list[dict[str, Any]] = []
        shared_route_by_pair: dict[str, dict[str, Any]] = {}
        for spec in screening:
            record = await_collect(collector, spec)
            output = _decode(model, mean, std, threshold, record)
            probabilities = _decoder_likelihood(output)
            shared_route = shared_router.inspect(record) if shared_router is not None else None
            if shared_route is not None:
                shared_route_by_pair[str((spec.get("pair") or {}).get("pair_id", ""))] = shared_route
            probabilities = _fuse_shared_route(probabilities, shared_route)
            probabilities.update({key: probabilities[key] * value for key, value in _projection_likelihood(record).items()})
            step = belief.observe(spec["path"], probabilities, evidence_hash=record["evidence"]["evidence_hash"])
            records.append(record)
            trace.append({"stage": "screen", "pair_id": (spec.get("pair") or {}).get("pair_id", "control"), "variant": (spec.get("pair") or {}).get("variant", "plain"), "decoder_candidate": output.get("candidate_family"), "posterior": step["posterior"], "evidence_hash": record["evidence"]["evidence_hash"], "shared_router": shared_route})
            for candidate in by_surface.get(str((spec.get("pair") or {}).get("pair_id", "")), [])[1:]:
                candidate_copy = dict(candidate)
                candidate_output = _decode(model, mean, std, threshold, record)
                likelihood = _decoder_likelihood(candidate_output)
                candidate_shared = shared_route_by_pair.get(str((spec.get("pair") or {}).get("pair_id", "")))
                likelihood = _fuse_shared_route(likelihood, candidate_shared)
                candidate_copy["rule_ir_decoder"] = {"probabilities": likelihood, "confidence": max(likelihood.values()) / sum(likelihood.values())}
                candidate_copy["surface_discriminator"] = {"probabilities": likelihood}
                candidate_copy["model_score"] = float(output.get("confidence", 0.0))
                refinements.append(candidate_copy)

        while refinements and len(records) < MAX_REQUESTS:
            chosen = belief.choose_next_probe(refinements)
            selected = choose_active_probe([chosen])
            record = await_collect(collector, selected)
            output = _decode(model, mean, std, threshold, record)
            probabilities = _decoder_likelihood(output)
            shared_route = shared_router.inspect(record) if shared_router is not None else None
            probabilities = _fuse_shared_route(probabilities, shared_route)
            probabilities.update({key: probabilities[key] * value for key, value in _projection_likelihood(record).items()})
            step = belief.observe(selected["path"], probabilities, evidence_hash=record["evidence"]["evidence_hash"])
            records.append(record)
            pair = selected.get("pair") or {}
            trace.append({"stage": "refine", "pair_id": pair.get("pair_id", ""), "variant": pair.get("variant", ""), "decoder_candidate": output.get("candidate_family"), "posterior": step["posterior"], "evidence_hash": record["evidence"]["evidence_hash"], "shared_router": shared_route})
            refinements = [row for row in refinements if not (str((row.get("pair") or {}).get("pair_id", "")) == str(pair.get("pair_id", "")) and str((row.get("pair") or {}).get("variant", "")) == str(pair.get("variant", "")))]

        predictions = [_decode(model, mean, std, threshold, record) for record in records]
        grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("pair"):
                grouped_records[str(record["pair"]["pair_id"])].append(record)
        pair_results = []
        for pair_id, pair_records in sorted(grouped_records.items()):
            candidate_records = [dict(record, candidate_family=prediction["candidate_family"]) for record, prediction in zip(pair_records, [_decode(model, mean, std, threshold, record) for record in pair_records])]
            result = revalidate_sql_pair(candidate_records, authorized_source_hash=source_hash, oracle_name=SQL_FIXTURE_ORACLE)
            result["pair_id"] = pair_id
            pair_results.append(result)
        report = {
            "protocol_id": PROTOCOL_ID,
            "schema_version": "sift-pg-pk-09-sql-active-probe-report-v1",
            "target": {"base_url": SQL_FIXTURE_BASE_URL, "target_instance_id": f"sql-active-{threading.get_ident()}", "fixture_source_sha256": source_hash, "external_network": False, "fresh_target": True},
            "static_request_count": len(specs),
            "request_count": len(records),
            "complete_pair_count": sum(len(rows) == 2 for rows in grouped_records.values()),
            "oracle_revalidated_pair_count": sum(result["accepted"] for result in pair_results),
            "decoder_abstain_count": sum(_decode(model, mean, std, threshold, record)["abstained"] for record in records),
            "shared_router_abstain_count": sum(int(bool((row.get("shared_router") or {}).get("abstained", True))) for row in trace if row.get("shared_router") is not None),
            "shared_router_ood_count": sum(int(bool((row.get("shared_router") or {}).get("ood", False))) for row in trace if row.get("shared_router") is not None),
            "shared_router_positive_authority": False,
            "selection_trace": trace,
            "pair_results": pair_results,
            "belief": belief.snapshot(),
            "safety": {"loopback_only": True, "get_only": True, "external_network": False, "script_execution": False, "database_write": False, "real_sleep_performed": False, "raw_body_stored": False, "shared_router_diagnostic_only": True},
            "report_path": str(REPORT_PATH.relative_to(ROOT)),
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps({"protocol_id": PROTOCOL_ID, "static_request_count": report["static_request_count"], "request_count": report["request_count"], "complete_pair_count": report["complete_pair_count"], "oracle_revalidated_pair_count": report["oracle_revalidated_pair_count"], "decoder_abstain_count": report["decoder_abstain_count"], "report": report["report_path"], "markdown": str(MARKDOWN_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def await_collect(collector: SqlDifferentialCollector, spec: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(collector.collect(spec))


if __name__ == "__main__":
    main()
