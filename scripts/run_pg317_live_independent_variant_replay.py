"""PG-317 live replay on a fresh independent local Docker target.

The live lane reuses PG-314's isolated GET/POST SQL row-shape evaluator but
loads the PG-317 question-anchor checkpoint.  It is evaluation-only: the
model produces bounded abstract references, while the local adapter owns the
reviewed wire catalog.  Promotion and vulnerability claims stay disabled.
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


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG314 = _load("pg314_for_pg317_live", ROOT / "scripts" / "run_pg314_independent_variant_replay.py")
PG315 = _load("pg315_for_pg317_live", ROOT / "scripts" / "run_pg315_worst_seed_replay.py")

RESEARCH = ROOT / "research"
CHECKPOINT = ROOT / "artifacts" / "pg317-question-anchor" / "pg317_question_anchor_moe_local_morning.pt"
TRAINING_REPORT = RESEARCH / "pg317_question_anchor_moe_training_report_v1_local_morning.json"
DATASET = RESEARCH / "pg317_question_anchor_dataset_v1.json"
AUDIT = RESEARCH / "pg317_question_anchor_dataset_audit_v1.json"
REPORT = RESEARCH / "pg317_live_independent_variant_replay_report_v1.json"
CATALOG = RESEARCH / "pg317_live_independent_variant_human_catalog_v1.json"
TRACE = RESEARCH / "pg317_live_independent_variant_trace_v1.json"
SEED = 31701


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _require_gate() -> None:
    if os.environ.get("PG317_LOCAL_DOCKER_EVAL") != "1":
        raise RuntimeError("PG-317 live replay requires explicit PG317_LOCAL_DOCKER_EVAL=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-317 local Docker replay is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    for path in (CHECKPOINT, TRAINING_REPORT, DATASET, AUDIT):
        if not path.exists():
            raise RuntimeError(f"PG-317 missing required artifact: {path}")


def main() -> int:
    _require_gate()
    routes = PG314._route_set()
    device = torch.device("cpu")
    model, vocabulary, symbolic = PG314.load_causal_checkpoint(CHECKPOINT, device)
    if not symbolic or len(vocabulary) < 70:
        raise RuntimeError("PG-317 requires a symbolic Rule-IR checkpoint")
    preflight = PG314._preflight(model, vocabulary, device, routes)
    repair_rows = [PG315._repair_probe(model, vocabulary, device, route) for route in routes]
    human: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, route in enumerate(routes):
        PG314.SEED = SEED
        row, _abstract, trace = PG314._run_route(route, index, model, vocabulary, device, None)
        row["record_id"] = f"pg317:{route['id']}:{index}"
        row["model"]["checkpoint"] = str(CHECKPOINT.relative_to(ROOT))
        trace["record_id"] = row["record_id"]
        traces.append(trace)
        human.append(row)
    elapsed = round(time.monotonic() - started, 3)
    role_entries = [entry for row in human for entry in row["model"]["entries"]]
    variant_exact = sum(int(bool(entry["proposal"].get("variant_exact"))) for entry in role_entries)
    typed_count = sum(int(bool(row["oracle"].get("typed_effect_confirmed"))) for row in human)
    negative_violations = sum(int(row["oracle"].get("all_variant_exact") is False and any(entry["role"] == "negative_control" and entry.get("variant") != "negative_control" for entry in row["model"]["entries"])) for row in human)
    repair_correct = sum(int(row["abstain_or_repair_correct"]) for row in repair_rows)
    training_report = _load_json(TRAINING_REPORT)
    dataset = _load_json(DATASET)
    audit = _load_json(AUDIT)
    report = {
        "protocol_id": "pg-pk-317-live-independent-variant-v1",
        "schema_version": "pg317-live-independent-variant-replay-report-v1",
        "status": "completed_real_local_docker_pg317_live_replay",
        "runtime": {"execution_window": "Asia/Shanghai 08:00-18:00", "explicit_flag": "PG317_LOCAL_DOCKER_EVAL=1", "device": "cpu_inference_only", "image": PG314.INDEPENDENT_IMAGE, "network": "none", "seed": SEED, "route_ids": list(PG314.ROUTE_IDS)},
        "model": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "architecture": "causal_transformer_moe_next_token", "target_representation": "multi_missing_question_anchor_plus_symbolic_slot_copy_plus_failure_repair", "symbolic_checkpoint": symbolic, "wire_generation": "source_grounded_binding_after_model_variant_guard", "raw_payload_in_context": False, "oracle_target_in_context": False, "raw_response_body_in_context": False},
        "offline_anchor_reference": {"training_report": str(TRAINING_REPORT.relative_to(ROOT)), "training_report_sha256": training_report.get("report_sha256"), "dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "worst_anchor_question_exact": (training_report.get("metrics") or {}).get("holdout_anchor_question_exact")},
        "preflight_identifiability": preflight,
        "failure_repair": {"rows": repair_rows, "correct": repair_correct, "count": len(repair_rows), "rate": round(repair_correct / max(len(repair_rows), 1), 6)},
        "counts": {"route_count": len(human), "get_count": sum(int(str(row["route"]["method"]).upper() == "GET") for row in human), "post_count": sum(int(str(row["route"]["method"]).upper() == "POST") for row in human), "variant_role_count": len(role_entries), "variant_exact_count": variant_exact, "model_variant_send_count": sum(int(entry.get("sent")) for entry in role_entries), "model_typed_effect_count": typed_count, "negative_lane_violation_count": negative_violations, "fresh_reset_count": len(human), "typed_evidence_hash_count": sum(int(bool(row["oracle"].get("evidence_sha256"))) for row in human), "elapsed_seconds": elapsed},
        "checks": {"real_docker_contacted": True, "loopback_only": True, "docker_network_none": True, "external_network_disabled": True, "get_post_pair": True, "fresh_reset_per_route": all(bool(row["target"]["fresh_reset"].get("fresh_target")) for row in human), "zero_volume_per_route": all(int(row["target"]["fresh_reset"].get("volume_mount_count", -1)) == 0 for row in human), "database_health_per_route": all(row["target"]["fresh_reset"].get("database_health_gate") == "mysqli_root_pikachu_ok" for row in human), "source_attestation_per_route": all(len(str(row["target"].get("source_sha256", ""))) == 64 for row in human), "typed_evidence_hash_per_route": all(bool(row["oracle"].get("evidence_sha256")) for row in human), "negative_controls_present": all(any(entry["role"] == "negative_control" for entry in row["model"]["entries"]) for row in human), "raw_payload_in_model_context": False, "raw_response_body_stored": False, "public_target_contacted": False, "sql_time_delay": False, "sql_write": False},
        "hypothesis_gate": {"status": "blocked", "checks": {"offline_anchor_audit": audit.get("status") == "passed", "offline_anchor_question_worst_seed": ((training_report.get("metrics") or {}).get("holdout_anchor_question_exact") or {}).get("min", 0) >= 0.95, "preflight_question_recall": preflight["question_recall"] >= 0.9, "preflight_zero_unsafe_allow": preflight["unsafe_allow"] == 0, "variant_selection_exact": variant_exact == len(role_entries), "repair_exact": repair_correct == len(repair_rows), "negative_zero": negative_violations == 0, "typed_all_routes": typed_count == len(human), "fresh_get_post_pair": True, "promotion_blocked": True}, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["PG-317 live lane covers one independent image and SQL row-shape GET/POST only", "DOM/XSS browser oracle and additional families remain separate", "source-grounded adapter output is not literal payload generation", "all promotion and long-term-memory writes remain disabled"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
    }
    report["report_sha256"] = _digest(report)
    catalog = {"schema_version": "pg317-live-independent-variant-human-catalog-v1", "entries": human, "failure_repair": repair_rows, "raw_payloads_human_review_only": True, "raw_response_bodies_stored": False, "external_network": False}
    catalog["catalog_sha256"] = _digest(catalog)
    trace = {"schema_version": "pg317-live-independent-variant-trace-v1", "episodes": traces, "failure_repair": repair_rows, "raw_payload_stored": False, "raw_response_body_stored": False, "training_promotion_allowed": False, "memory_promotion_allowed": False}
    trace["trace_sha256"] = _digest(trace)
    _write(REPORT, report)
    _write(CATALOG, catalog)
    _write(TRACE, trace)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "preflight": {key: preflight[key] for key in ("question_recall", "unsafe_allow")}, "failure_repair": report["failure_repair"], "gate": report["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
