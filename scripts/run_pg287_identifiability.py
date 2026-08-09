"""Remote A800 experiment for PG-287 evidence-conditioned decoding.

This is an engineering experiment over controlled counterfactuals.  It
measures whether the decoder asks when encoding is unobservable and decodes a
resolved encoding only when an explicit observation token is present.  It is
not a live vulnerability claim and cannot promote memory without PG-286-live
target evidence.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg287_identifiability_dataset_v1.json"
AUDIT = RESEARCH / "pg287_identifiability_dataset_audit_v1.json"
REMOTE_PROBE = RESEARCH / "pg280_remote_docker_probe_v2.json"
REPORT = RESEARCH / "pg287_identifiability_training_report_v1.json"
TRACE = RESEARCH / "pg287_identifiability_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg287_identifiability_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg287_identifiability_training_report_v1.md"
OUT_DIR = ROOT / "artifacts" / "pg287-identifiability"
CHECKPOINT = OUT_DIR / "pg287_identifiability.pt"
SEEDS = (28711, 28712, 28713)
VARIANTS = {"plain_sft": 1.0, "guarded_sft": 2.5}


def _sha(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract(tokens: Sequence[str], prefix: str, fallback: str = "unknown") -> str:
    for token in tokens:
        if str(token).startswith(prefix):
            return str(token).split("=", 1)[1]
    return fallback


def _decode_rows(model: Any, rows: Sequence[Mapping[str, Any]], context_vocab: Mapping[str, int], target_vocab: Mapping[str, int], device: torch.device) -> list[list[str]]:
    from app.pg285_payload_grounding import encode_rows, greedy_decode

    context_values, context_lengths, _, target_tokens = encode_rows(rows, context_vocab, target_vocab)
    with torch.inference_mode():
        return greedy_decode(model, context_values.to(device), context_lengths.to(device), target_vocab, max_tokens=max(len(row) for row in target_tokens) + 2)


def _identifiability_metrics(model: Any, rows: Sequence[Mapping[str, Any]], context_vocab: Mapping[str, int], target_vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    predicted = _decode_rows(model, rows, context_vocab, target_vocab, device)
    ambiguous = [row for row in rows if row.get("variant") == "ambiguous" and not row.get("hard_negative")]
    resolved = [row for row in rows if row.get("variant") == "resolved"]
    hard = [row for row in rows if row.get("hard_negative")]
    by_id = {row.get("record_id"): pred for row, pred in zip(rows, predicted)}

    def action(row: Mapping[str, Any]) -> str:
        return _extract(by_id.get(row.get("record_id"), []), "final_action=", _extract(by_id.get(row.get("record_id"), []), "plan=", "abstain"))

    def encoding(row: Mapping[str, Any]) -> str:
        return _extract(by_id.get(row.get("record_id"), []), "encoding=", "unknown")

    ambiguous_ask = sum(action(row) == "ask_typed" for row in ambiguous)
    resolved_encoding = sum(encoding(row) == str(row.get("target", {}).get("encoding", "unknown")) for row in resolved)
    hard_ask = sum(action(row) == "ask_typed" for row in hard)
    hard_false_allow = sum(_extract(by_id.get(row.get("record_id"), []), "safe_to_send=", "0") == "1" for row in hard)
    return {
        "ambiguous_count": len(ambiguous),
        "ambiguous_ask_recall": round(ambiguous_ask / len(ambiguous), 6) if ambiguous else None,
        "resolved_count": len(resolved),
        # A zero-denominator split is a coverage gap, not a 0% model score.
        # Keep it explicit so a missing family-heldout supervision signal
        # cannot be mistaken for a measured failure.
        "resolved_encoding_accuracy": round(resolved_encoding / len(resolved), 6) if resolved else None,
        "hard_negative_count": len(hard),
        "hard_negative_ask_recall": round(hard_ask / len(hard), 6) if hard else None,
        "hard_negative_false_allow": hard_false_allow,
        "predicted_plan_hash": _sha(predicted),
    }


def main() -> None:
    if os.environ.get("PG287_REMOTE_RUN") != "1":
        raise RuntimeError("PG-287 training is remote-only; set PG287_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from app.pg285_payload_grounding import build_vocabs, evaluate, train_model

    data = _load(DATASET)
    audit = _load(AUDIT)
    remote_probe = _load(REMOTE_PROBE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-287 dataset audit must pass before training")
    if data.get("training_contract", {}).get("remote_a800_required") is not True:
        raise RuntimeError("PG-287 dataset is not marked remote-A800-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or assignment["visible_device_count"] != 1 or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-287 requires remote A800 GPU0, got {assignment}")
    rows = list(data.get("records") or [])
    hard_rows = list(data.get("hard_negative_records") or [])
    train_rows = [row for row in rows if row.get("split") == "train"]
    route_rows = [row for row in rows if row.get("split") == "route_dev"]
    family_rows = [row for row in rows if row.get("split") == "family_holdout"]
    context_vocab, target_vocab = build_vocabs(train_rows)
    started = time.perf_counter()
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in VARIANTS}
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        random.seed(seed)
        for variant, risk_weight in VARIANTS.items():
            model = train_model(train_rows, context_vocab, target_vocab, device, seed, risk_weight=risk_weight, epochs=160, embed_dim=112, hidden_dim=256)
            metrics = {
                "train": evaluate(model, train_rows, context_vocab, target_vocab, device),
                "route_dev": evaluate(model, route_rows, context_vocab, target_vocab, device),
                "family_holdout": evaluate(model, family_rows, context_vocab, target_vocab, device),
                "hard_negative": evaluate(model, hard_rows, context_vocab, target_vocab, device),
                "identifiability": {
                    "train": _identifiability_metrics(model, train_rows, context_vocab, target_vocab, device),
                    "route_dev": _identifiability_metrics(model, route_rows, context_vocab, target_vocab, device),
                    "family_holdout": _identifiability_metrics(model, family_rows, context_vocab, target_vocab, device),
                    "all": _identifiability_metrics(model, [*rows, *hard_rows], context_vocab, target_vocab, device),
                },
            }
            per_seed[variant].append({"seed": seed, **metrics})
            if seed == SEEDS[-1]:
                checkpoints[variant] = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def aggregate(values: list[dict[str, Any]], key: str) -> dict[str, Any]:
        numbers = [float(item[key]) for item in values if item.get(key) is not None]
        if not numbers:
            return {"mean": None, "min": None, "max": None, "available_count": 0}
        return {"mean": round(sum(numbers) / len(numbers), 6), "min": round(min(numbers), 6), "max": round(max(numbers), 6), "available_count": len(numbers)}

    variants: dict[str, Any] = {}
    for variant, values in per_seed.items():
        route_ident = [value["identifiability"]["route_dev"] for value in values]
        family_ident = [value["identifiability"]["family_holdout"] for value in values]
        all_ident = [value["identifiability"]["all"] for value in values]
        variants[variant] = {
            "risk_weight": VARIANTS[variant],
            "route_ambiguous_ask_recall": aggregate(route_ident, "ambiguous_ask_recall"),
            "route_resolved_encoding_accuracy": aggregate(route_ident, "resolved_encoding_accuracy"),
            "family_ambiguous_ask_recall": aggregate(family_ident, "ambiguous_ask_recall"),
            "family_resolved_encoding_accuracy": aggregate(family_ident, "resolved_encoding_accuracy"),
            "hard_negative_ask_recall": aggregate(all_ident, "hard_negative_ask_recall"),
            "hard_negative_false_allow_max": max(int(item["hard_negative_false_allow"]) for item in all_ident),
            "route_sequence_exact": aggregate([value["route_dev"] for value in values], "sequence_exact_accuracy"),
            "family_sequence_exact": aggregate([value["family_holdout"] for value in values], "sequence_exact_accuracy"),
        }
    eligible = [
        (summary["route_ambiguous_ask_recall"]["min"], summary["route_resolved_encoding_accuracy"]["min"], -summary["risk_weight"], variant)
        for variant, summary in variants.items()
        if summary["hard_negative_false_allow_max"] == 0
        and summary["route_ambiguous_ask_recall"]["min"] is not None
        and summary["route_resolved_encoding_accuracy"]["min"] is not None
    ]
    selected = max(eligible)[-1] if eligible else min(variants, key=lambda variant: (variants[variant]["hard_negative_false_allow_max"], variants[variant]["risk_weight"]))
    selected_summary = variants[selected]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"],
        "ambiguous_ask_recall_recorded": selected_summary["route_ambiguous_ask_recall"]["min"] is not None,
        "resolved_encoding_accuracy_recorded": selected_summary["route_resolved_encoding_accuracy"]["min"] is not None,
        "family_resolved_coverage_recorded": selected_summary["family_resolved_encoding_accuracy"]["available_count"] > 0,
        "hard_negative_false_allow_zero": selected_summary["hard_negative_false_allow_max"] == 0,
        "family_hidden_in_context": data.get("training_contract", {}).get("family_hidden_in_context") is True,
        "literal_payload_excluded": data.get("training_contract", {}).get("literal_probe_values_out_of_context") is True,
        "remote_docker_honest": remote_probe.get("status") != "available" or remote_probe.get("real_application_gold_rows", 0) == 0,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg287-identifiability-training-v1",
        "schema_version": "pg287-identifiability-training-report-v1",
        "status": "completed_remote_pg287_identifiability_training",
        "source": {"dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": data["dataset_sha256"], "dataset_audit": str(AUDIT.relative_to(ROOT).as_posix()), "dataset_audit_sha256": audit["audit_sha256"], "remote_probe": str(REMOTE_PROBE.relative_to(ROOT).as_posix()), "remote_docker_status": remote_probe.get("status", "unknown"), "real_application_gold_rows": int(remote_probe.get("real_application_gold_rows", 0) or 0), "live_send": False, "literal_payload_in_context": False},
        "device": assignment,
        "split": {"train": len(train_rows), "route_dev": len(route_rows), "family_holdout": len(family_rows), "hard_negative": len(hard_rows), "seeds": list(SEEDS)},
        "vocabulary": {"context_size": len(context_vocab), "target_size": len(target_vocab)},
        "variants": variants,
        "per_seed": per_seed,
        "selected_variant": selected,
        "engineering_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "scientific_gate": {"status": "blocked", "checks": {"live_target_evaluator": False, "real_application_gold": False, "template_source_heldout": False, "cross_seed_observation": False, "literal_payload_success": False, "family_resolved_coverage": selected_summary["family_resolved_encoding_accuracy"]["available_count"] > 0}, "reasons": ["counterfactual evidence tokens are derived from PG-285 templates", "family-heldout split has no resolved encoding rows", "remote Docker/evaluator unavailable", "no fresh target replay"], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "PG-287 only tests ask-on-ambiguity versus resolved encoding decoding; it does not supply live application gold."},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    trace = {"schema_version": "pg287-identifiability-training-trace-v1", "report_sha256": report["report_sha256"], "selected_variant": selected, "training_eligible": False, "memory_write": False, "live_send": False, "literal_payload_in_context": False, "ambiguity_is_ask_target": True}
    trace["trace_sha256"] = _sha(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg287-identifiability-training-protocol-v1", "remote_a800_gpu0_only": True, "ambiguous_encoding_action": "ask_typed", "resolved_encoding_action": "bounded_wire_plan", "hard_negative_evaluation_only": True, "live_replay_required_for_promotion": True, "report_sha256": report["report_sha256"], "next_experiment": "PG-287-live：把 encoding_observed/field_role token 替换为真实 GET/POST evaluator projection，再做 source-heldout replay。"}
    protocol["protocol_sha256"] = _sha(protocol)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg287-identifiability-checkpoint-v1", "assignment": assignment, "context_vocab": context_vocab, "target_vocab": target_vocab, "selected_variant": selected, "state": checkpoints.get(selected, {})}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-287 identifiability-conditioned decoding", "", f"selected={selected}", f"route ambiguous ask min={selected_summary['route_ambiguous_ask_recall']['min']}", f"route resolved encoding min={selected_summary['route_resolved_encoding_accuracy']['min']}", f"family resolved encoding min={selected_summary['family_resolved_encoding_accuracy']['min']} (available={selected_summary['family_resolved_encoding_accuracy']['available_count']})", f"hard-negative false-allow={selected_summary['hard_negative_false_allow_max']}", "scientific_gate=blocked", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": assignment, "selected_variant": selected, "selected_summary": selected_summary, "engineering_gate": report["engineering_gate"], "scientific_gate": report["scientific_gate"], "report": str(REPORT.relative_to(ROOT).as_posix())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
