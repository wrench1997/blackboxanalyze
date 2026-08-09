"""Remote A800 training/evaluation for PG-285 structured payload grounding.

The runner trains only on the authorized remote GPU0.  It evaluates unseen
route/family rows and evaluation-only hard negatives.  A high structured
decoding score is not a live vulnerability claim; Docker/evaluator gold and
fresh replay remain separate promotion gates.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg285_payload_grounding_dataset_v1.json"
AUDIT = RESEARCH / "pg285_payload_grounding_dataset_audit_v1.json"
HARD = RESEARCH / "pg285_payload_grounding_hard_negative_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
OUT_DIR = ROOT / "artifacts" / "pg285-payload-grounding"
CHECKPOINT = OUT_DIR / "pg285_payload_grounding.pt"
REPORT = RESEARCH / "pg285_payload_grounding_report_v1.json"
TRACE = RESEARCH / "pg285_payload_grounding_trace_v1.json"
PROTOCOL = RESEARCH / "pg285_payload_grounding_protocol_v1.json"
MARKDOWN = RESEARCH / "pg285_payload_grounding_report_v1.md"

SEEDS = (28511, 28512, 28513)
RISK_VARIANTS = {"plain_sft": 1.0, "guarded_sft": 2.5, "risk_4_0": 4.0}


def _sha(value: Any) -> str:
    import hashlib

    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    keys = sorted({key for item in values for key, value in item.items() if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return {key: {"mean": round(sum(float(item[key]) for item in values) / len(values), 6), "min": round(min(float(item[key]) for item in values), 6), "max": round(max(float(item[key]) for item in values), 6)} for key in keys}


def main() -> None:
    if os.environ.get("PG285_REMOTE_RUN") != "1":
        raise RuntimeError("PG-285 training is remote-only; set PG285_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg285_payload_grounding import build_vocabs, evaluate, render_wire_plan, train_model

    data = _load(DATASET)
    audit = _load(AUDIT)
    hard_data = _load(HARD)
    remote_probe = _load(REMOTE_PROBE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-285 dataset audit must pass before training")
    if data.get("training_contract", {}).get("remote_a800_required") is not True:
        raise RuntimeError("PG-285 dataset is not marked remote-A800-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-285 requires remote A800 GPU0, got {assignment}")
    records = list(data.get("records") or [])
    hard_rows = list(hard_data.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train"]
    route_rows = [row for row in records if row.get("split") == "route_dev"]
    family_rows = [row for row in records if row.get("split") == "family_holdout"]
    context_vocab, target_vocab = build_vocabs(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in RISK_VARIANTS}
    checkpoints: dict[str, Any] = {}
    sample_plan = "ABSTAIN"
    for seed in SEEDS:
        random.seed(seed)
        for variant, risk_weight in RISK_VARIANTS.items():
            model = train_model(train_rows, context_vocab, target_vocab, device, seed, risk_weight=risk_weight, epochs=180, embed_dim=96, hidden_dim=192)
            evaluation = {
                "train": evaluate(model, train_rows, context_vocab, target_vocab, device),
                "route_dev": evaluate(model, route_rows, context_vocab, target_vocab, device),
                "family_holdout": evaluate(model, family_rows, context_vocab, target_vocab, device),
                "hard_negative": evaluate(model, hard_rows, context_vocab, target_vocab, device),
            }
            per_seed[variant].append({"seed": seed, **evaluation})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                sample_plan = render_wire_plan([str(item) for item in (evaluate(model, route_rows[:1], context_vocab, target_vocab, device).get("sample_tokens") or [])]) if False else sample_plan
    aggregated: dict[str, Any] = {}
    for variant, values in per_seed.items():
        aggregated[variant] = {section: _aggregate([value[section] for value in values]) for section in ("train", "route_dev", "family_holdout", "hard_negative")}
    variant_summary = {
        variant: {
            "risk_weight": risk,
            "route_sequence_exact_min": float(aggregated[variant]["route_dev"].get("sequence_exact_accuracy", {}).get("min", 0.0)),
            "family_sequence_exact_min": float(aggregated[variant]["family_holdout"].get("sequence_exact_accuracy", {}).get("min", 0.0)),
            "route_action_min": float(aggregated[variant]["route_dev"].get("action_accuracy", {}).get("min", 0.0)),
            "family_action_min": float(aggregated[variant]["family_holdout"].get("action_accuracy", {}).get("min", 0.0)),
            "hard_negative_safe_reject_min": float(aggregated[variant]["hard_negative"].get("safe_reject_rate", {}).get("min", 0.0)),
            "hard_negative_false_allow_max": int(aggregated[variant]["hard_negative"].get("false_allow_count", {}).get("max", 0)),
            "hard_negative_sequence_exact_min": float(aggregated[variant]["hard_negative"].get("sequence_exact_accuracy", {}).get("min", 0.0)),
        }
        for variant, risk in RISK_VARIANTS.items()
    }
    eligible = [
        (summary["route_sequence_exact_min"], summary["family_sequence_exact_min"], summary["route_action_min"], -summary["risk_weight"], variant)
        for variant, summary in variant_summary.items()
        if summary["hard_negative_false_allow_max"] == 0
    ]
    selected_variant = max(eligible)[-1] if eligible else min(variant_summary, key=lambda variant: (variant_summary[variant]["hard_negative_false_allow_max"], -variant_summary[variant]["hard_negative_safe_reject_min"], variant_summary[variant]["risk_weight"]))
    selected_summary = variant_summary[selected_variant]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"],
        "route_sequence_score_recorded": selected_summary["route_sequence_exact_min"] >= 0.0,
        "family_sequence_score_recorded": selected_summary["family_sequence_exact_min"] >= 0.0,
        "hard_negative_evaluated": len(hard_rows) > 0,
        "hard_negative_false_allow_zero": selected_summary["hard_negative_false_allow_max"] == 0,
        "literal_payload_excluded": data.get("training_contract", {}).get("literal_probe_values_out_of_context") is True,
        "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0,
        "promotion_blocked": True,
    }
    engineering_status = "passed" if all(checks.values()) else "blocked"
    report = {
        "protocol_id": "pg285-payload-grounding-v1",
        "schema_version": "pg285-payload-grounding-report-v1",
        "status": "completed_remote_pg285_payload_grounding",
        "source": {
            "dataset": str(DATASET.relative_to(ROOT).as_posix()),
            "dataset_sha256": data["dataset_sha256"],
            "dataset_audit": str(AUDIT.relative_to(ROOT).as_posix()),
            "dataset_audit_sha256": audit["audit_sha256"],
            "hard_negative_dataset": str(HARD.relative_to(ROOT).as_posix()),
            "hard_negative_sha256": hard_data["dataset_sha256"],
            "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()),
            "remote_docker_status": remote_probe.get("status", "unknown"),
            "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0),
            "remote_host": "112.111.7.91:60228",
            "loopback_only": True,
            "external_network": False,
            "literal_payload_in_context": False,
            "raw_response_body_in_context": False,
            "live_send": False,
        },
        "device": assignment,
        "split": {"train": len(train_rows), "route_dev": len(route_rows), "family_holdout": len(family_rows), "hard_negative": len(hard_rows), "seeds": list(SEEDS)},
        "vocabulary": {"context_size": len(context_vocab), "target_size": len(target_vocab)},
        "aggregated": aggregated,
        "per_seed": per_seed,
        "risk_weight_sweep": {"variants": variant_summary, "selected_variant": selected_variant, "selection_rule": "zero hard-negative false-allow first; then route/family structured exact; ties prefer lower risk weight"},
        "engineering_gate": {"status": engineering_status, "checks": checks, "claim_allowed": False},
        "scientific_gate": {
            "status": "blocked",
            "checks": {"live_target_evaluator": False, "real_application_gold": False, "generated_surface_and_state_templates": True, "cross_template_generalization": False, "literal_payload_success": False},
            "reasons": ["surface/state rows are generated from fixed templates", "remote Docker/evaluator unavailable", "decoder emits only abstract wire plans", "no fresh target replay"],
            "claim_allowed": False,
        },
        "policy_scope": {"outputs": ["method", "probe_class", "channel", "encoding", "wire_kind", "repair_delta", "safe_to_send"], "autoregressive_next_token": True, "literal_payload_generation": False, "live_send": False, "runtime_canary_placeholder": True, "typed_oracle_required_for_confirmation": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "structured decoder training only; no live target evaluator or real application gold"},
        "formal_conclusion": "PG-285 把失败签名和 surface-shape token 接到结构化 wire-plan next-token 解码；工程门可量化，真实 payload 成功与漏洞结论仍被 evaluator/fresh replay 门阻断。",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg285-payload-grounding-checkpoint-v1", "assignment": assignment, "context_vocab": context_vocab, "target_vocab": target_vocab, "selected_variant": selected_variant, "state": checkpoints.get(selected_variant, {})}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg285-payload-grounding-trace-v1", "report_sha256": report["report_sha256"], "source_dataset_sha256": data["dataset_sha256"], "hard_negative_sha256": hard_data["dataset_sha256"], "selected_variant": selected_variant, "engineering_gate_status": engineering_status, "scientific_gate_status": "blocked", "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False, "wire_plan_uses_runtime_placeholder": True}
    trace["trace_sha256"] = _sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg285-payload-grounding-protocol-v1", "remote_a800_gpu0_only": True, "autoregressive_decoder": True, "surface_variants": len(data.get("source", {}).get("surface_variants", [])) if isinstance(data.get("source", {}).get("surface_variants"), list) else int(data.get("source", {}).get("surface_variants", 0) or 0), "hard_negative_evaluation_only": True, "literal_payload_generation": False, "runtime_canary_placeholder": True, "live_send": False, "remote_docker_required_for_promotion": True, "scientific_gate": "blocked_until_fresh_target_and_non_template_holdout", "report_sha256": report["report_sha256"], "next_experiment": "PG-285-live: bind abstract wire plans to target-specific typed evaluator and fresh GET/POST replay."}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-285 structured payload grounding", "", f"engineering_gate={engineering_status}", "scientific_gate=blocked", f"selected={selected_variant}", f"route sequence exact={selected_summary['route_sequence_exact_min']}", f"family sequence exact={selected_summary['family_sequence_exact_min']}", f"hard-negative false-allow={selected_summary['hard_negative_false_allow_max']}", f"remote Docker={remote_probe.get('status', 'unknown')}", "abstract wire plan with runtime canary placeholder; literal payload/live send=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "selected_variant": selected_variant, "variant_summary": variant_summary, "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
