"""PG-288 remote A800 experiment: structured Rule-IR verification.

This runner reuses the audited PG-285 abstract wire-plan dataset, trains the
decoder only on the authorized remote GPU0, and reports sequence/slot/safety
metrics separately.  It never stores or emits literal payload strings and it
does not send a request to a target.
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
OUT_DIR = ROOT / "artifacts" / "pg288-rule-ir-verifier"
CHECKPOINT = OUT_DIR / "pg288_rule_ir_verifier.pt"
REPORT = RESEARCH / "pg288_rule_ir_verifier_report_v1.json"
TRACE = RESEARCH / "pg288_rule_ir_verifier_trace_v1.json"
PROTOCOL = RESEARCH / "pg288_rule_ir_verifier_protocol_v1.json"
MARKDOWN = RESEARCH / "pg288_rule_ir_verifier_report_v1.md"

SEEDS = (28801, 28802, 28803)
VARIANTS = {"plain_sft": 1.0, "guarded_sft": 2.5, "risk_4_0": 4.0}


def _sha(value: Any) -> str:
    import hashlib

    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    keys = sorted(
        {
            key
            for item in values
            for key, value in item.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None
        }
    )
    return {
        key: {
            "mean": round(sum(float(item[key]) for item in values) / len(values), 6),
            "min": round(min(float(item[key]) for item in values), 6),
            "max": round(max(float(item[key]) for item in values), 6),
        }
        for key in keys
    }


def _decode_rows(model: Any, rows: list[dict[str, Any]], context_vocab: dict[str, int], target_vocab: dict[str, int], device: torch.device, encode_rows: Any, greedy_decode: Any) -> list[list[str]]:
    context_values, context_lengths, _, _ = encode_rows(rows, context_vocab, target_vocab)
    with torch.inference_mode():
        return greedy_decode(
            model,
            context_values.to(device),
            context_lengths.to(device),
            target_vocab,
            max_tokens=max(len(list(row.get("target_tokens") or [])) for row in rows) + 2,
        )


def main() -> None:
    if os.environ.get("PG288_REMOTE_RUN") != "1":
        raise RuntimeError("PG-288 training is remote-only; set PG288_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg285_payload_grounding import build_vocabs, encode_rows, evaluate, greedy_decode, train_model
    from app.pg288_rule_ir_verifier import evaluate_decoded_plans

    data = _load(DATASET)
    audit = _load(AUDIT)
    hard_data = _load(HARD)
    remote_probe = _load(REMOTE_PROBE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-288 requires the audited PG-285 dataset")
    if data.get("training_contract", {}).get("literal_probe_values_out_of_context") is not True:
        raise RuntimeError("literal probe values must remain outside the training context")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-288 requires remote A800 GPU0, got {assignment}")

    records = list(data.get("records") or [])
    hard_rows = list(hard_data.get("records") or [])
    train_rows = [row for row in records if row.get("split") == "train"]
    route_rows = [row for row in records if row.get("split") == "route_dev"]
    family_rows = [row for row in records if row.get("split") == "family_holdout"]
    context_vocab, target_vocab = build_vocabs(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    checkpoints: dict[str, Any] = {}
    examples: dict[str, Any] = {}

    for seed in SEEDS:
        random.seed(seed)
        for variant, risk_weight in VARIANTS.items():
            model = train_model(
                train_rows,
                context_vocab,
                target_vocab,
                device,
                seed,
                risk_weight=risk_weight,
                epochs=180,
                embed_dim=96,
                hidden_dim=192,
            )
            sections: dict[str, Any] = {}
            for name, rows, is_hard in (
                ("train", train_rows, False),
                ("route_dev", route_rows, False),
                ("family_holdout", family_rows, False),
                ("hard_negative", hard_rows, True),
            ):
                decoded = _decode_rows(model, rows, context_vocab, target_vocab, device, encode_rows, greedy_decode)
                baseline = evaluate(model, rows, context_vocab, target_vocab, device)
                verified = evaluate_decoded_plans(rows, decoded, hard_negative=is_hard)
                sections[name] = {**baseline, "rule_ir_verifier": verified}
                if seed == SEEDS[-1] and name in {"route_dev", "family_holdout", "hard_negative"}:
                    examples[f"{variant}:{name}"] = [
                        {"predicted_tokens": list(decoded[index]), "verification": verify}
                        for index, verify in enumerate(
                            __import__("app.pg288_rule_ir_verifier", fromlist=["verify_plan_tokens"]).verify_plan_tokens(item)
                            for item in decoded[:2]
                        )
                    ]
            per_seed[variant].append({"seed": seed, **sections})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    aggregated: dict[str, Any] = {}
    for variant, values in per_seed.items():
        aggregated[variant] = {
            section: {
                "baseline": _aggregate([value[section] for value in values]),
                "verifier": _aggregate([value[section]["rule_ir_verifier"] for value in values]),
            }
            for section in ("train", "route_dev", "family_holdout", "hard_negative")
        }
    variant_summary = {}
    for variant, risk_weight in VARIANTS.items():
        hard = aggregated[variant]["hard_negative"]["verifier"]
        family = aggregated[variant]["family_holdout"]["verifier"]
        route = aggregated[variant]["route_dev"]["verifier"]
        variant_summary[variant] = {
            "risk_weight": risk_weight,
            "route_structural_valid_min": float(route.get("structural_valid_rate", {}).get("min", 0.0)),
            "route_renderable_min": float(route.get("renderable_rate", {}).get("min", 0.0)),
            "family_structural_valid_min": float(family.get("structural_valid_rate", {}).get("min", 0.0)),
            "family_sequence_exact_min": float(family.get("sequence_exact_accuracy", {}).get("min", 0.0)),
            "hard_negative_structural_valid_min": float(hard.get("structural_valid_rate", {}).get("min", 0.0)),
            "hard_negative_safe_consistent_min": float(hard.get("safe_consistent_rate", {}).get("min", 0.0)),
            "hard_negative_false_allow_max": int(hard.get("false_allow_count", {}).get("max", 0)),
            "hard_negative_sequence_exact_min": float(hard.get("sequence_exact_accuracy", {}).get("min", 0.0)),
        }
    eligible = [
        (
            summary["hard_negative_false_allow_max"] == 0,
            summary["hard_negative_structural_valid_min"],
            summary["family_structural_valid_min"],
            summary["route_renderable_min"],
            -summary["risk_weight"],
            variant,
        )
        for variant, summary in variant_summary.items()
    ]
    selected_variant = max(eligible)[-1]
    selected_summary = variant_summary[selected_variant]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"],
        "verifier_metrics_recorded": all("structural_valid_rate" in aggregated[selected_variant][section]["verifier"] for section in ("route_dev", "family_holdout", "hard_negative")),
        "hard_negative_evaluated": len(hard_rows) > 0,
        "hard_negative_false_allow_zero": selected_summary["hard_negative_false_allow_max"] == 0,
        "literal_payload_excluded": data.get("training_contract", {}).get("literal_probe_values_out_of_context") is True,
        "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0,
        "promotion_blocked": True,
    }
    engineering_status = "passed" if all(checks.values()) else "blocked"
    report = {
        "protocol_id": "pg288-rule-ir-verifier-v1",
        "schema_version": "pg288-rule-ir-verifier-report-v1",
        "status": "completed_remote_pg288_rule_ir_verifier",
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
        "examples": examples,
        "variant_summary": variant_summary,
        "selected_variant": selected_variant,
        "selection_rule": "zero hard-negative false-allow first; then hard-negative structural validity; then family/route renderability; ties prefer lower risk weight",
        "engineering_gate": {"status": engineering_status, "checks": checks, "claim_allowed": False},
        "scientific_gate": {
            "status": "blocked",
            "checks": {"live_target_evaluator": False, "real_application_gold": False, "fresh_target_replay": False, "literal_payload_success": False, "abstract_structure_verified": True},
            "reasons": ["PG-285 surfaces are generated templates", "remote Docker/evaluator unavailable", "verifier checks abstract plans only", "no fresh target replay"],
            "claim_allowed": False,
        },
        "policy_scope": {"outputs": ["method", "probe_class", "channel", "encoding", "wire_kind", "repair_delta", "safe_to_send"], "autoregressive_next_token": True, "literal_payload_generation": False, "live_send": False, "runtime_canary_placeholder": True, "typed_oracle_required_for_confirmation": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "structural verifier experiment only; no live evaluator or real application gold"},
        "formal_conclusion": "PG-288 将 Rule-IR 序列完整性、slot/wire 一致性、safe 位和 hard-negative false-allow 分开计分；动作高分不能再掩盖结构缺失，但这仍不是现实漏洞成功证明。",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg288-rule-ir-verifier-checkpoint-v1", "assignment": assignment, "context_vocab": context_vocab, "target_vocab": target_vocab, "selected_variant": selected_variant, "state": checkpoints.get(selected_variant, {})}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg288-rule-ir-verifier-trace-v1", "report_sha256": report["report_sha256"], "source_dataset_sha256": data["dataset_sha256"], "hard_negative_sha256": hard_data["dataset_sha256"], "selected_variant": selected_variant, "engineering_gate_status": engineering_status, "scientific_gate_status": "blocked", "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False, "structural_verifier": True}
    trace["trace_sha256"] = _sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg288-rule-ir-verifier-protocol-v1", "remote_a800_gpu0_only": True, "structural_verifier": True, "hard_negative_evaluation_only": True, "literal_payload_generation": False, "runtime_canary_placeholder": True, "live_send": False, "remote_docker_required_for_promotion": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-288-live: bind verified abstract plans to target-specific typed evaluator and fresh GET/POST replay."}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-288 Rule-IR verifier", "", f"engineering_gate={engineering_status}", "scientific_gate=blocked", f"selected={selected_variant}", f"route structural={selected_summary['route_structural_valid_min']}", f"family structural={selected_summary['family_structural_valid_min']}", f"hard-negative structural={selected_summary['hard_negative_structural_valid_min']}", f"hard-negative false-allow={selected_summary['hard_negative_false_allow_max']}", "abstract verifier only; literal payload/live send=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "selected_variant": selected_variant, "variant_summary": variant_summary, "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
