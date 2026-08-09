"""Remote A800 PG-289: train on evaluator-gap abstain counterfactuals."""

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
EXPERIMENT = os.environ.get("PG_ABSTAIN_EXPERIMENT", "pg289_safe_abstain")
AUGMENTED = RESEARCH / os.environ.get("PG_ABSTAIN_DATASET", f"{EXPERIMENT}_dataset_v1.json")
AUGMENTED_AUDIT = RESEARCH / os.environ.get("PG_ABSTAIN_AUDIT", f"{EXPERIMENT}_dataset_audit_v1.json")
BASE = RESEARCH / os.environ.get("PG_ABSTAIN_BASE_DATASET", "pg285_payload_grounding_dataset_v1.json")
BASE_AUDIT = RESEARCH / os.environ.get("PG_ABSTAIN_BASE_AUDIT", "pg285_payload_grounding_dataset_audit_v1.json")
HARD = RESEARCH / os.environ.get("PG_ABSTAIN_HARD", "pg285_payload_grounding_hard_negative_v1.json")
REMOTE_PROBE = RESEARCH / os.environ.get("PG_ABSTAIN_REMOTE_PROBE", "pg280_remote_docker_probe_v2.json")
OUT_DIR = ROOT / "artifacts" / EXPERIMENT.replace("_", "-")
CHECKPOINT = OUT_DIR / f"{EXPERIMENT}.pt"
REPORT = RESEARCH / f"{EXPERIMENT}_report_v1.json"
TRACE = RESEARCH / f"{EXPERIMENT}_trace_v1.json"
PROTOCOL = RESEARCH / f"{EXPERIMENT}_protocol_v1.json"
MARKDOWN = RESEARCH / f"{EXPERIMENT}_report_v1.md"
SEEDS = (28901, 28902, 28903)
VARIANTS = {"plain_sft": 1.0, "guarded_sft": 2.5, "risk_4_0": 4.0}


def _sha(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for value in values for key, item in value.items() if isinstance(item, (int, float)) and not isinstance(item, bool) and item is not None})
    return {key: {"mean": round(sum(float(value[key]) for value in values) / len(values), 6), "min": round(min(float(value[key]) for value in values), 6), "max": round(max(float(value[key]) for value in values), 6)} for key in keys}


def _decode(model: Any, rows: list[dict[str, Any]], context_vocab: dict[str, int], target_vocab: dict[str, int], device: torch.device, encode_rows: Any, decoder: Any) -> list[list[str]]:
    context_values, context_lengths, _, _ = encode_rows(rows, context_vocab, target_vocab)
    with torch.inference_mode():
        return decoder(model, context_values.to(device), context_lengths.to(device), target_vocab, max_tokens=max(len(list(row.get("target_tokens") or [])) for row in rows) + 2)


def main() -> None:
    if os.environ.get("PG289_REMOTE_RUN") != "1":
        raise RuntimeError("PG abstain training is remote-only; set PG289_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg285_payload_grounding import build_vocabs, encode_rows, evaluate, greedy_decode, train_model
    from app.pg288_rule_ir_verifier import apply_context_safety_gate, constrained_greedy_decode, evaluate_decoded_plans

    augmented = _load(AUGMENTED)
    augmented_audit = _load(AUGMENTED_AUDIT)
    base = _load(BASE)
    base_audit = _load(BASE_AUDIT)
    hard_data = _load(HARD)
    remote_probe = _load(REMOTE_PROBE)
    if augmented_audit.get("status") != "passed" or base_audit.get("status") != "passed":
        raise RuntimeError(f"{EXPERIMENT} source/augmentation audits must pass")
    if augmented.get("training_contract", {}).get("literal_probe_values_out_of_context") is not True:
        raise RuntimeError(f"{EXPERIMENT} literal probe values must remain out of context")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"{EXPERIMENT} requires remote A800 GPU0, got {assignment}")

    base_rows = list(base.get("records") or [])
    base_train_rows = [row for row in base_rows if row.get("split") == "train"]
    counterfactual_rows = list(augmented.get("records") or [])
    # The augmentation is a supplement, never a replacement for positive and
    # repair trajectories.  A previous run used only the decoys and therefore
    # collapsed to all-abstain; the explicit mix is now part of the contract.
    train_rows = base_train_rows + counterfactual_rows
    route_rows = [row for row in base_rows if row.get("split") == "route_dev"]
    family_rows = [row for row in base_rows if row.get("split") == "family_holdout"]
    hard_rows = list(hard_data.get("records") or [])
    context_vocab, target_vocab = build_vocabs(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        random.seed(seed)
        for variant, risk_weight in VARIANTS.items():
            model = train_model(train_rows, context_vocab, target_vocab, device, seed, risk_weight=risk_weight, epochs=180, embed_dim=96, hidden_dim=192)
            sections: dict[str, Any] = {}
            for name, rows, is_hard in (("train", train_rows, False), ("route_dev", route_rows, False), ("family_holdout", family_rows, False), ("hard_negative", hard_rows, True)):
                greedy = _decode(model, rows, context_vocab, target_vocab, device, encode_rows, greedy_decode)
                constrained = _decode(model, rows, context_vocab, target_vocab, device, encode_rows, constrained_greedy_decode)
                guarded, gate_changed = apply_context_safety_gate(rows, constrained)
                sections[name] = {"baseline": evaluate(model, rows, context_vocab, target_vocab, device), "greedy_verifier": evaluate_decoded_plans(rows, greedy, hard_negative=is_hard), "verifier": evaluate_decoded_plans(rows, constrained, hard_negative=is_hard), "guarded_verifier": evaluate_decoded_plans(rows, guarded, hard_negative=is_hard), "safety_gate_changed": gate_changed}
            per_seed[variant].append({"seed": seed, **sections})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    aggregated = {
        variant: {section: {head: _aggregate([value[section][head] for value in values]) for head in ("baseline", "greedy_verifier", "verifier", "guarded_verifier")} for section in ("train", "route_dev", "family_holdout", "hard_negative")}
        for variant, values in per_seed.items()
    }
    summary: dict[str, Any] = {}
    for variant, risk_weight in VARIANTS.items():
        hard = aggregated[variant]["hard_negative"]["verifier"]
        family = aggregated[variant]["family_holdout"]["verifier"]
        route = aggregated[variant]["route_dev"]["verifier"]
        guarded_hard = aggregated[variant]["hard_negative"]["guarded_verifier"]
        guarded_route = aggregated[variant]["route_dev"]["guarded_verifier"]
        summary[variant] = {"risk_weight": risk_weight, "route_structural_valid_min": float(route.get("structural_valid_rate", {}).get("min", 0.0)), "route_renderable_min": float(route.get("renderable_rate", {}).get("min", 0.0)), "family_structural_valid_min": float(family.get("structural_valid_rate", {}).get("min", 0.0)), "hard_negative_structural_valid_min": float(hard.get("structural_valid_rate", {}).get("min", 0.0)), "hard_negative_safe_consistent_min": float(hard.get("safe_consistent_rate", {}).get("min", 0.0)), "hard_negative_false_allow_max": int(hard.get("false_allow_count", {}).get("max", 0)), "hard_negative_sequence_exact_min": float(hard.get("sequence_exact_accuracy", {}).get("min", 0.0)), "guarded_route_renderable_min": float(guarded_route.get("renderable_rate", {}).get("min", 0.0)), "guarded_hard_negative_false_allow_max": int(guarded_hard.get("false_allow_count", {}).get("max", 0)), "guarded_hard_negative_structural_valid_min": float(guarded_hard.get("structural_valid_rate", {}).get("min", 0.0))}
    selected_variant = max((item["hard_negative_false_allow_max"] == 0, item["hard_negative_structural_valid_min"], item["family_structural_valid_min"], item["route_renderable_min"], -item["risk_weight"], variant) for variant, item in summary.items())[-1]
    selected = summary[selected_variant]
    checks = {"augmentation_audit_pass": augmented_audit.get("status") == "passed", "base_audit_pass": base_audit.get("status") == "passed", "base_train_mixed": len(base_train_rows) > 0 and len(counterfactual_rows) > 0 and len(train_rows) == len(base_train_rows) + len(counterfactual_rows), "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"], "hard_negative_evaluated": len(hard_rows) > 0, "literal_payload_excluded": augmented.get("training_contract", {}).get("literal_probe_values_out_of_context") is True, "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0, "promotion_blocked": True}
    report = {"protocol_id": f"{EXPERIMENT}-v1", "schema_version": f"{EXPERIMENT}-report-v1", "status": f"completed_remote_{EXPERIMENT}", "source": {"augmented_dataset": str(AUGMENTED.relative_to(ROOT).as_posix()), "augmented_dataset_sha256": augmented["dataset_sha256"], "augmented_audit": str(AUGMENTED_AUDIT.relative_to(ROOT).as_posix()), "augmented_audit_sha256": augmented_audit["audit_sha256"], "base_dataset": str(BASE.relative_to(ROOT).as_posix()), "base_dataset_sha256": base["dataset_sha256"], "hard_negative_dataset": str(HARD.relative_to(ROOT).as_posix()), "hard_negative_sha256": hard_data["dataset_sha256"], "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()), "remote_docker_status": remote_probe.get("status", "unknown"), "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0), "remote_host": "112.111.7.91:60228", "loopback_only": True, "external_network": False, "literal_payload_in_context": False, "live_send": False}, "device": assignment, "split": {"base_train": len(base_train_rows), "counterfactual_train": len(counterfactual_rows), "mixed_train": len(train_rows), "route_dev": len(route_rows), "family_holdout": len(family_rows), "hard_negative": len(hard_rows), "seeds": list(SEEDS)}, "vocabulary": {"context_size": len(context_vocab), "target_size": len(target_vocab)}, "aggregated": aggregated, "per_seed": per_seed, "variant_summary": summary, "selected_variant": selected_variant, "selection_rule": "report false-allow and structural validity; no promotion based on this synthetic experiment", "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "scientific_gate": {"status": "blocked", "checks": {"live_target_evaluator": False, "real_application_gold": False, "fresh_target_replay": False, "literal_payload_success": False, "abstain_augmentation_evaluated": True}, "reasons": ["training rows include counterfactual templates", "remote Docker/evaluator unavailable", "no fresh target replay"], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": f"{EXPERIMENT} only tests unresolved/evaluator-gap abstain generalization"}, "formal_conclusion": f"{EXPERIMENT} 用训练-only、族/来源无关的 evaluator-gap 反事实增强安全 abstain；其效果仍必须在真实 fresh evaluator 上复核，不能替代 payload 成功证据。", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = _sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg289-safe-abstain-checkpoint-v1", "assignment": assignment, "context_vocab": context_vocab, "target_vocab": target_vocab, "selected_variant": selected_variant, "state": checkpoints.get(selected_variant, {})}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": f"{EXPERIMENT}-trace-v1", "report_sha256": report["report_sha256"], "augmented_dataset_sha256": augmented["dataset_sha256"], "hard_negative_sha256": hard_data["dataset_sha256"], "selected_variant": selected_variant, "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False}
    trace["trace_sha256"] = _sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": report["protocol_id"], "schema_version": f"{EXPERIMENT}-protocol-v1", "remote_a800_gpu0_only": True, "augmentation_train_only": True, "family_labels_out_of_context": True, "hard_negative_evaluation_only": True, "literal_payload_generation": False, "live_send": False, "report_sha256": report["report_sha256"], "next_experiment": f"{EXPERIMENT}-live: fresh evaluator replay of abstain/candidate boundary on unseen surfaces."}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = summary[selected_variant]
    MARKDOWN.write_text("\n".join(["# PG-289 safe abstain", "", f"engineering_gate={report['engineering_gate']['status']}", "scientific_gate=blocked", f"selected={selected_variant}", f"hard-negative structural={selected['hard_negative_structural_valid_min']}", f"hard-negative false-allow={selected['hard_negative_false_allow_max']}", "counterfactual training only; literal payload/live send=false", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "selected_variant": selected_variant, "variant_summary": summary, "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
