"""Evaluate PG-303 guarded composition on the frozen PG-302B model."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel, generate_target  # noqa: E402
from app.pg301_payload_assembly import evaluate_assembly_rows  # noqa: E402
from app.pg302_symbolic_assembly import bind_symbolic_plan  # noqa: E402
from app.pg303_guarded_composer import compose_guarded_plan  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg302_symbolic_assembly_dataset_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg302b-symbolic-curriculum" / "pg302b_symbolic_curriculum_moe_local_morning.pt"
REPORT = RESEARCH / "pg303_guarded_composer_eval_report_v1.json"
TRACE = RESEARCH / "pg303_guarded_composer_eval_trace_v1.json"
PROTOCOL = RESEARCH / "pg303_guarded_composer_eval_protocol_v1.json"


def verify() -> datetime:
    if os.environ.get("PG303_LOCAL_EVAL") != "1" or os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-303 requires PG303_LOCAL_EVAL=1 and BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-303 local evaluation is limited to 08:00-18:00 Asia/Shanghai; now={now.isoformat()}")
    return now


def bound_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        clone = copy.deepcopy(row)
        clone["target_tokens"] = bind_symbolic_plan(row.get("target_tokens", []), row.get("context_tokens", [])) or []
        result.append(clone)
    return result


def main() -> None:
    now = verify()
    dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = CausalMoEConfig(**checkpoint["config"])
    model = CausalMoELanguageModel(vocab_size=len(checkpoint["vocabulary"]), config=config)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    rows = list(dataset.get("records") or [])
    expected_bound = bound_rows(rows)
    started = time.perf_counter()
    lanes: dict[str, dict[str, Any]] = {}
    for split in ("train", "implementation_holdout", "hard_negative_eval"):
        lane = [row for row in rows if row.get("split") == split]
        raw_predictions = [generate_target(model, row.get("context_tokens", []), len(row.get("target_tokens", [])), checkpoint["vocabulary"], torch.device("cpu")) for row in lane]
        raw_bound = [bind_symbolic_plan(prediction, row.get("context_tokens", [])) or [] for row, prediction in zip(lane, raw_predictions)]
        guarded = [compose_guarded_plan(prediction, row.get("context_tokens", [])) for row, prediction in zip(lane, raw_predictions)]
        lanes[split] = {"count": len(lane), "raw_bound": evaluate_assembly_rows(expected_bound_for(expected_bound, split), raw_bound), "guarded": evaluate_assembly_rows(expected_bound_for(expected_bound, split), guarded)}
    hold = lanes["implementation_holdout"]["guarded"]
    hard = lanes["hard_negative_eval"]["guarded"]
    checks = {
        "frozen_checkpoint_present": CHECKPOINT.exists(),
        "loopback_only_offline_eval": True,
        "guard_question_recall": hold.get("missing_question_recall", 0.0) >= 0.99,
        "guard_slot_exact": hold.get("assembly_slot_exact", 0.0) >= 0.99,
        "guard_hard_negative_false_allow": hard.get("hard_negative_false_allow", 999999) == 0,
        "guard_hard_negative_unnecessary_question": hard.get("unnecessary_question_rate", 999999) <= 0.0,
        "neural_claim_blocked": True,
        "wire_emission": False,
    }
    report = {
        "protocol_id": "pg303-guarded-composer-eval-v1",
        "schema_version": "pg303-guarded-composer-eval-report-v1",
        "status": "completed_local_morning_pg303_guarded_eval",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset.get("dataset_sha256"), "checkpoint": str(CHECKPOINT.relative_to(ROOT).as_posix()), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "wire_emission": False, "literal_payload": False},
        "model_role": "neural_symbolic_proposal_plus_visible-slot-identifiability-guard",
        "lanes": lanes,
        "checks": checks,
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "reasons": ["guarded offline evaluation only", "no fresh typed evaluator", "no real application gold", "guard is not neural evidence"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "payload_catalog_promotion_allowed": False},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg303-guarded-composer-trace-v1", "report_sha256": report["report_sha256"], "neural_model_is_proposal_only": True, "identifiability_guard": True, "training_eligible": False, "memory_write": False, "wire_emission": False}
    trace["trace_sha256"] = sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg303-guarded-composer-protocol-v1", "execution_mode": "local_morning_offline_eval", "window": "08:00-18:00 Asia/Shanghai", "visible_slots_only": True, "neural_proposal_only": True, "typed_oracle_required": True, "fresh_reset_required": True, "negative_control_required": True, "wire_emission": False, "promotion_blocked": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-304: add evaluator-only loopback typed replay for guarded abstract plans."}
    protocol["protocol_sha256"] = sha256_json(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "lanes": lanes, "engineering_gate": report["engineering_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


def expected_bound_for(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


if __name__ == "__main__":
    main()
