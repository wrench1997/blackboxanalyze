"""PG-315: worst-seed replay of all PG-313 checkpoints on PG-314's lab.

PG-314 used the best PG-313 checkpoint.  This runner removes that selection
shortcut: every saved seed checkpoint is loaded and replayed on the second
independent image.  It also asks each model about a failure-feedback context;
that lane must abstain/repair rather than send a fresh candidate blindly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_pg314() -> Any:
    path = ROOT / "scripts" / "run_pg314_independent_variant_replay.py"
    spec = importlib.util.spec_from_file_location("pg314_for_pg315", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-314 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG314 = _load_pg314()
RESEARCH = ROOT / "research"
PG313_REPORT = RESEARCH / "pg313_probe_variant_moe_training_report_v1_local_morning.json"
REPORT = RESEARCH / "pg315_worst_seed_replay_report_v1.json"
CATALOG = RESEARCH / "pg315_worst_seed_human_catalog_v1.json"
DATASET = RESEARCH / "pg315_worst_seed_training_dataset_v1.json"
TRACE = RESEARCH / "pg315_worst_seed_trace_v1.json"
PROTOCOL = RESEARCH / "pg315_worst_seed_protocol_v1.json"
MARKDOWN = RESEARCH / "pg315_worst_seed_replay_report_v1.md"
SEEDS = (31301, 31302, 31303)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("PG315_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-315 requires explicit PG315_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-315 local replay is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    if not PG313_REPORT.exists():
        raise RuntimeError("PG-315 requires the PG-313 report with per-seed checkpoints")


def _repair_probe(model: Any, vocabulary: dict[str, int], device: torch.device, route: dict[str, Any]) -> dict[str, Any]:
    tokens = PG314.context_tokens(
        str(route["method"]),
        typed_available="1",
        replay_ready="1",
        evidence_present="1",
        feedback_state="observable_progress",
        negative_control="1",
        fresh_reset="1",
        history_action="candidate_failed",
        failure_class="effect_not_confirmed",
    )
    proposal = PG314._extended_proposal(model, vocabulary, device, tokens)
    values = PG314.target_map(proposal["guarded_tokens"])
    expected_safe = False
    expected_next_action = "repair_abstract_plan"
    expected_repair_action = "retry_bounded_variant"
    actual_safe = bool(proposal.get("model_safe_to_send"))
    return {
        "route_id": str(route["id"]),
        "history_action": "candidate_failed",
        "failure_class": "effect_not_confirmed",
        "expected": {"safe_to_send": expected_safe, "variant": "none", "next_action": expected_next_action, "repair_action": expected_repair_action},
        "actual": {"safe_to_send": actual_safe, "variant": str(values.get("probe_variant", "none")), "next_action": str(values.get("next_action", "none")), "repair_action": str(values.get("repair_action", "none"))},
        "abstain_or_repair_correct": not actual_safe and str(values.get("probe_variant", "none")) == "none" and str(values.get("next_action", "none")) == expected_next_action and str(values.get("repair_action", "none")) == expected_repair_action,
        "raw_payload_in_context": False,
    }


def main() -> int:
    _require_gate()
    pg313 = _load(PG313_REPORT)
    per_seed_checkpoints = {int(row["seed"]): ROOT / str(row["checkpoint"]) for row in pg313.get("per_seed", [])}
    if set(per_seed_checkpoints) != set(SEEDS) or not all(path.exists() for path in per_seed_checkpoints.values()):
        raise RuntimeError("PG-315 requires all three PG-313 seed checkpoints")
    routes = PG314._route_set()
    device = torch.device("cpu")
    all_human: list[dict[str, Any]] = []
    all_abstract: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    seed_reports: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed in SEEDS:
        PG314.SEED = seed
        model, vocabulary, symbolic = PG314.load_causal_checkpoint(per_seed_checkpoints[seed], device)
        if not symbolic:
            raise RuntimeError(f"PG-315 seed {seed} is not symbolic")
        preflight = PG314._preflight(model, vocabulary, device, routes)
        repair_rows = [_repair_probe(model, vocabulary, device, route) for route in routes]
        seed_human: list[dict[str, Any]] = []
        seed_abstract: list[dict[str, Any]] = []
        seed_trace: list[dict[str, Any]] = []
        for index, route in enumerate(routes):
            human, abstract, trace = PG314._run_route(route, index, model, vocabulary, device, None)
            human["record_id"] = f"pg315:{seed}:{route['id']}"
            human["seed"] = seed
            human["model"]["checkpoint"] = str(per_seed_checkpoints[seed].relative_to(ROOT))
            trace["record_id"] = human["record_id"]
            trace["seed"] = seed
            for row in abstract:
                row["seed"] = seed
            seed_human.append(human)
            seed_abstract.extend(abstract)
            seed_trace.append(trace)
        role_entries = [entry for row in seed_human for entry in row["model"]["entries"]]
        variant_exact = sum(int(bool(entry["proposal"].get("variant_exact"))) for entry in role_entries)
        negative_violations = sum(int(row["oracle"].get("all_variant_exact") is False and any(entry["role"] == "negative_control" and entry.get("variant") != "negative_control" for entry in row["model"]["entries"])) for row in seed_human)
        typed_count = sum(int(bool(row["oracle"].get("typed_effect_confirmed"))) for row in seed_human)
        seed_report = {
            "seed": seed,
            "checkpoint": str(per_seed_checkpoints[seed].relative_to(ROOT)),
            "route_count": len(seed_human),
            "preflight": {key: preflight[key] for key in ("count", "question_recall", "unsafe_allow")},
            "variant_role_count": len(role_entries),
            "variant_exact_count": variant_exact,
            "variant_exact_rate": round(variant_exact / max(len(role_entries), 1), 6),
            "typed_effect_count": typed_count,
            "negative_lane_violation_count": negative_violations,
            "repair_rows": repair_rows,
            "repair_abstain_rate": round(sum(int(row["abstain_or_repair_correct"]) for row in repair_rows) / max(len(repair_rows), 1), 6),
            "human": seed_human,
            "abstract": seed_abstract,
            "trace": seed_trace,
        }
        seed_reports.append(seed_report)
        all_human.extend(seed_human)
        all_abstract.extend(seed_abstract)
        all_trace.extend(seed_trace)
    elapsed = round(time.monotonic() - started, 3)
    worst_question = min(float(row["preflight"]["question_recall"]) for row in seed_reports)
    worst_variant = min(float(row["variant_exact_rate"]) for row in seed_reports)
    worst_repair = min(float(row["repair_abstain_rate"]) for row in seed_reports)
    negative_total = sum(int(row["negative_lane_violation_count"]) for row in seed_reports)
    typed_total = sum(int(row["typed_effect_count"]) for row in seed_reports)
    report = {
        "protocol_id": "pg-pk-315-worst-seed-replay-v1",
        "schema_version": "pg315-worst-seed-replay-report-v1",
        "status": "completed_real_local_docker_all_seed_replay",
        "runtime": {"execution_window": "Asia/Shanghai 08:00-18:00", "explicit_flag": "PG315_LOCAL_DOCKER_EVAL=1", "device": "cpu_inference_only", "image": PG314.INDEPENDENT_IMAGE, "network": "none", "route_ids": list(PG314.ROUTE_IDS), "seed_count": len(SEEDS)},
        "model": {"architecture": "causal_transformer_moe_next_token", "checkpoint_family": "PG-313 per-seed checkpoints", "symbolic_checkpoint": True, "wire_generation": "source_grounded_binding_after_model_variant_guard", "oracle_target_in_context": False, "raw_payload_in_context": False},
        "counts": {"seed_count": len(SEEDS), "route_count": len(all_human), "get_count": sum(int(str(row["route"]["method"]).upper() == "GET") for row in all_human), "post_count": sum(int(str(row["route"]["method"]).upper() == "POST") for row in all_human), "model_variant_role_count": sum(int(row["variant_role_count"]) for row in seed_reports), "model_variant_exact_count": sum(int(row["variant_exact_count"]) for row in seed_reports), "model_typed_effect_count": typed_total, "negative_lane_violation_count": negative_total, "repair_row_count": sum(len(row["repair_rows"]) for row in seed_reports), "repair_abstain_correct_count": sum(sum(int(item["abstain_or_repair_correct"]) for item in row["repair_rows"]) for row in seed_reports), "fresh_reset_count": len(all_human), "typed_evidence_hash_count": sum(int(bool(row["oracle"].get("evidence_sha256"))) for row in all_human), "elapsed_seconds": elapsed},
        "worst_seed_metrics": {"question_recall_min": worst_question, "variant_exact_min": worst_variant, "repair_abstain_min": worst_repair, "negative_lane_violation_max": negative_total},
        "per_seed": [{key: value for key, value in row.items() if key not in {"human", "abstract", "trace"}} for row in seed_reports],
        "hypothesis_gate": {"status": "blocked", "checks": {"all_seed_checkpoints": True, "get_post_pair": True, "question_recall_worst_seed": worst_question >= 0.9, "variant_exact_worst_seed": worst_variant >= 0.9, "repair_abstain_worst_seed": worst_repair >= 1.0, "negative_lane_zero_violation": negative_total == 0, "typed_effect_all_routes": typed_total == len(all_human), "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-313 worst-seed offline base/variant/hard-negative gate remains blocked", "SQL row-shape lane only; DOM/XSS typed browser lane is separate", "source-grounded adapter output is not literal payload generation", "no training, memory or payload catalog promotion"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "artifacts": {"human_catalog": str(CATALOG.relative_to(ROOT)), "training_dataset": str(DATASET.relative_to(ROOT)), "trace": str(TRACE.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))},
    }
    report["report_sha256"] = _digest(report)
    catalog = {"schema_version": "pg315-worst-seed-human-catalog-v1", "status": "completed_real_local_all_seed_human_catalog", "entries": all_human, "raw_payloads_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False}
    catalog["catalog_sha256"] = _digest(catalog)
    dataset = {"schema_version": "pg315-worst-seed-training-dataset-v1", "source_catalog": str(CATALOG.relative_to(ROOT)), "records": all_abstract, "counts": {"records": len(all_abstract), "seed_count": len(SEEDS), "route_count": len(all_human), "repair_rows": sum(len(row["repair_rows"]) for row in seed_reports)}, "contract": {"multi_seed": True, "failure_feedback_repair": True, "abstract_variant_only": True, "real_get_post_replay": True, "fresh_reset_required": True, "typed_oracle_required": True, "payload_strings_excluded": True, "response_bodies_excluded": True, "training_promotion_allowed": False, "memory_promotion_allowed": False}, "dataset_sha256": ""}
    dataset["dataset_sha256"] = _digest(dataset)
    trace = {"schema_version": "pg315-worst-seed-trace-v1", "episodes": all_trace, "repair_rows": [item for row in seed_reports for item in row["repair_rows"]], "raw_payloads_human_catalog_only": True, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}
    trace["trace_sha256"] = _digest(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg315-worst-seed-protocol-v1", "scope": {"target": "authorized local Docker independent Pikachu implementation", "image": PG314.INDEPENDENT_IMAGE, "network": "none", "loopback_only": True, "external_network": False, "methods": ["GET", "POST"], "seeds": list(SEEDS)}, "model_contract": {"decoder_only_next_token": True, "abstract_slot_assembly": True, "model_selects_variant": True, "failure_feedback_repair_or_abstain": True, "oracle_target_off_input": True}, "required_gates": {"all_seed_checkpoints": True, "worst_seed_question": True, "worst_seed_variant": True, "worst_seed_repair_abstain": True, "negative_zero_violation": True, "fresh_reset": True, "typed_evidence_hash": True, "raw_payload_training_excluded": True}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False}}
    protocol["protocol_sha256"] = _digest(protocol)
    _write(REPORT, report)
    _write(CATALOG, catalog)
    _write(DATASET, dataset)
    _write(TRACE, trace)
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-315 PG-313 三 seed worst-case 复放", "", f"seeds={len(SEEDS)} routes={len(all_human)} GET={report['counts']['get_count']} POST={report['counts']['post_count']}", f"worst question={worst_question}; variant={worst_variant}; repair abstain={worst_repair}; negative violations={negative_total}", "promotion 关闭；模型只输出抽象槽位，wire 由 source-grounded adapter 绑定。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "worst_seed_metrics": report["worst_seed_metrics"], "gate": report["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
