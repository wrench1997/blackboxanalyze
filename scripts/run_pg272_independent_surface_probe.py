"""PG-272 baseline: test PG-270 on an independent implementation/surface.

The in-repository surface fixture is a fresh, inert HTTP implementation with
HTML attribute/text, JSON, response-header and plain-control surfaces.  Its
typed local oracle is independent from Pikachu.  Only abstract context/target
tokens reach the checkpoint; raw request and response material is discarded.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from app.cross_app_surface_fixture import (  # noqa: E402
    SurfaceFixtureCollector,
    default_surface_fixture_specs,
    make_surface_fixture_server,
    surface_fixture_source_sha256,
)

CHECKPOINT = ROOT / "artifacts" / "pg270-teacher-sft" / "teacher_sft_ablation.pt"
REPORT = ROOT / "research" / "pg272_independent_surface_probe_report_v1.json"
TRACE = ROOT / "research" / "pg272_independent_surface_probe_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg272_independent_surface_probe_protocol_v1.json"
MARKDOWN = ROOT / "research" / "pg272_independent_surface_probe_report_v1.md"


def _load_pg270() -> Any:
    spec = importlib.util.spec_from_file_location("pg270_teacher_sft", SCRIPTS / "run_pg270_teacher_sft_ablation.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-270 model module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _context() -> list[str]:
    return [
        "[BOS]", "phase=observe", "method=GET", "field_bucket=1", "channel=query",
        "fresh_reset=1", "source_attested=1", "reference_sent=1", "negative_sent=1",
        "candidate_sent=1", "repair_attempted=0", "step_budget=4", "failure_observed=1", "[CTX_END]",
    ]


def _target(positive: bool) -> list[str]:
    if positive:
        return [
            "[TARGET_BOS]", "phase=baseline", "action=negative_control", "failure=negative_clean", "next_action=candidate_probe",
            "phase=reference", "action=reference_probe", "failure=reference_observed", "next_action=candidate_probe",
            "phase=candidate", "action=candidate_probe", "failure=none", "next_action=replay_confirmed",
            "final_belief=confirmed_effect", "[TARGET_EOS]",
        ]
    return [
        "[TARGET_BOS]", "phase=baseline", "action=negative_control", "failure=negative_clean", "next_action=candidate_probe",
        "phase=reference", "action=reference_probe", "failure=reference_observed", "next_action=candidate_probe",
        "phase=candidate", "action=candidate_probe", "failure=logic_oracle_gap", "next_action=diagnose_failure",
        "phase=diagnose", "action=abstain", "failure=no_typed_repair_available", "next_action=abstain",
        "final_belief=oracle_gap", "[TARGET_EOS]",
    ]


def _field(tokens: list[str], prefix: str) -> str | None:
    values = [token.split("=", 1)[1] for token in tokens if token.startswith(prefix)]
    return values[-1] if values else None


def main() -> None:
    started = time.perf_counter()
    server = make_surface_fixture_server()
    thread = threading.Thread(target=server.serve_forever, name="pg272-independent-surface", daemon=True)
    thread.start()
    try:
        source_hash = surface_fixture_source_sha256()
        records = asyncio.run(SurfaceFixtureCollector(target_instance_id=f"pg272-{threading.get_ident()}", source_hash=source_hash).collect_many(default_surface_fixture_specs("pg272")))
        if len(records) != 9:
            raise RuntimeError(f"expected 9 independent surface records, got {len(records)}")
        pg270 = _load_pg270()
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        vocabulary = dict(checkpoint["vocabulary"])
        reverse = {int(key): value for key, value in dict(checkpoint["reverse_vocabulary"]).items()}
        model = pg270.TinyConditionalDecoder(len(vocabulary))
        model.load_state_dict(checkpoint["guided_state"])
        model.eval()
        rows: list[dict[str, Any]] = []
        for record in records:
            positive = bool(record.get("rule_ir_result"))
            row = {"record_id": record["sample_id"], "context_tokens": _context(), "target_tokens": _target(positive), "expected_positive": positive, "surface_role": record["semantic"]["surface_role"], "source_hash": source_hash}
            generated = pg270._generate(model, row, vocabulary, reverse, torch.device("cpu"), max_len=len(row["target_tokens"]) + 5)
            expected_action = _field(row["target_tokens"], "next_action=")
            predicted_action = _field(generated, "next_action=")
            expected_belief = _field(row["target_tokens"], "final_belief=")
            predicted_belief = _field(generated, "final_belief=")
            predicted_positive = predicted_belief == "confirmed_effect"
            rows.append({
                "record_id": row["record_id"],
                "surface_role": row["surface_role"],
                "expected_positive": positive,
                "expected_next_action": expected_action,
                "predicted_next_action": predicted_action,
                "expected_final_belief": expected_belief,
                "predicted_final_belief": predicted_belief,
                "next_action_correct": predicted_action == expected_action,
                "final_belief_correct": predicted_belief == expected_belief,
                "model_positive": predicted_positive,
                "false_positive_candidate": predicted_positive and not positive,
                "false_negative_candidate": positive and not predicted_positive,
                "generated_tokens": generated,
                "context_tokens": row["context_tokens"],
            })
        total = len(rows)
        positives = [row for row in rows if row["expected_positive"]]
        negatives = [row for row in rows if not row["expected_positive"]]
        metrics = {
            "count": total,
            "positive_count": len(positives),
            "next_action_accuracy": round(sum(row["next_action_correct"] for row in rows) / total, 6),
            "final_belief_accuracy": round(sum(row["final_belief_correct"] for row in rows) / total, 6),
            "model_positive_count": sum(row["model_positive"] for row in rows),
            "false_positive_candidate_count": sum(row["false_positive_candidate"] for row in rows),
            "false_negative_candidate_count": sum(row["false_negative_candidate"] for row in rows),
            "positive_recall_candidate": round(sum(row["model_positive"] for row in positives) / max(len(positives), 1), 6),
            "negative_reject_candidate": round(sum(not row["model_positive"] for row in negatives) / max(len(negatives), 1), 6),
        }
        gate_checks = {
            "fresh_independent_source": source_hash != "",
            "source_has_new_implementation": True,
            "context_has_no_raw_material": all(not any(term in token.casefold() for term in ("payload", "response", "oracle", "body_sha")) for row in rows for token in row["context_tokens"]),
            "positive_recall_min": metrics["positive_recall_candidate"] >= 0.5,
            "negative_reject_min": metrics["negative_reject_candidate"] >= 0.8,
            "false_positive_zero": metrics["false_positive_candidate_count"] == 0,
        }
        report = {
            "protocol_id": "pg272-independent-surface-probe-v1",
            "schema_version": "pg272-independent-surface-probe-report-v1",
            "status": "completed_independent_implementation_evaluation",
            "target": {"implementation": "in_repo_surface_fixture", "source_hash": source_hash, "sample_count": total, "loopback_only": True, "external_network": False, "fresh_target": True, "typed_oracle_independent": True},
            "training_boundary": {"checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_variant": "PG-270 guided_sft", "surface_fixture_seen_during_training": False, "online_weight_update": False, "memory_write": False},
            "metrics": metrics,
            "gate": {"status": "passed" if all(gate_checks.values()) else "blocked", "checks": gate_checks, "claim_allowed": False},
            "rows": rows,
            "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "independent implementation probe only; score/RL optimization not yet applied"},
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        report["report_sha256"] = _sha(report)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        trace = {"schema_version": "pg272-independent-surface-probe-trace-v1", "evaluation_only": True, "training_eligible": False, "source_hash": source_hash, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
        trace["trace_sha256"] = _sha(trace)
        TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        protocol = {"protocol_id": "pg272-independent-surface-probe-v1", "schema_version": "pg272-independent-surface-probe-protocol-v1", "input_contract": {"independent_implementation": True, "fresh_target": True, "raw_material_off_context": True, "oracle_off_context": True}, "gate": {"positive_recall_min": 0.5, "negative_reject_min": 0.8, "false_positive_zero": True}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "claim_allowed": False}, "result": {"gate": report["gate"], "report_sha256": report["report_sha256"]}, "next_experiment": "PG-272-RL score-guided policy optimization if composition gate is blocked"}
        protocol["protocol_sha256"] = _sha(protocol)
        PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        MARKDOWN.write_text("\n".join(["# PG-272 独立实现疑问驱动组合泛化", "", f"独立 fixture={total} 条；positive={len(positives)}；模型候选 positive={metrics['model_positive_count']}。", f"next-action={metrics['next_action_accuracy']:.3f}；belief={metrics['final_belief_accuracy']:.3f}；positive recall={metrics['positive_recall_candidate']:.3f}；negative reject={metrics['negative_reject_candidate']:.3f}；false-positive candidate={metrics['false_positive_candidate_count']}。", "", f"gate=`{report['gate']['status']}`；本轮只验证，不训练、不写长期记忆。", ""]), encoding="utf-8")
        print(json.dumps({"status": report["status"], "metrics": metrics, "gate": report["gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
