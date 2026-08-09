"""PG-320 targeted question-lattice fine-tuning and retention test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG313 = _load_module("pg313_for_pg320", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG314 = _load_module("pg314_for_pg320", ROOT / "scripts" / "run_pg314_independent_variant_replay.py")
PG316 = _load_module("pg316_for_pg320", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402
from app.pg295_causal_moe import CausalMoELanguageModel  # noqa: E402

RESEARCH = ROOT / "research"
LATTICE = RESEARCH / "pg320_observation_lattice_dataset_v1.json"
LATTICE_AUDIT = RESEARCH / "pg320_observation_lattice_dataset_audit_v1.json"
PG319_DATASET = RESEARCH / "pg319_cross_impl_rule_ir_dataset_v1.json"
PG319_REPORT = RESEARCH / "pg319_cross_impl_moe_training_report_v1_local_morning.json"
PG318_TRACE = RESEARCH / "pg318_family_holdout_trace_v1.json"
REPORT = RESEARCH / "pg320_observation_lattice_finetune_report_v1_local_morning.json"
BASE_CHECKPOINT_DIR = ROOT / "artifacts" / "pg319-cross-impl" / "seeds"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg320-observation-lattice" / "seeds"
CHECKPOINT = ROOT / "artifacts" / "pg320-observation-lattice" / "pg320_observation_lattice_moe_local_morning.pt"
SEEDS = (31901, 31902, 31903)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-320 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-320 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _preflight_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(trace.get("multi_missing_preflight") or []):
        values = {key: "1" for key in OBSERVATION_KEYS}
        values["feedback_state"] = "negative_control_clear"
        for key in item.get("missing_slots") or []:
            values[str(key)] = "unknown"
        method = str(item.get("method", "GET")).upper()
        field = "query_param" if method == "GET" else "form_field"
        encoding = "url_percent" if method == "GET" else "form_urlencoded"
        raw = ["[BOS]"] + [f"{key}={values[key]}" for key in OBSERVATION_KEYS] + [f"surface_method={method}", f"surface_field_role={field}", f"surface_encoding={encoding}", "history_action=none", "failure_class=none", "step_budget=present", "[EOS]"]
        context = canonical_assembly_context(raw)
        rows.append({"record_id": f"pg318-preflight:{index}", "context_tokens": context, "target_tokens": probe_target_for_context(context), "training_eligible": False, "raw_payload_stored": False, "raw_response_body_stored": False})
    return rows


def _lane(model: Any, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    rows_list = [dict(row) for row in rows]
    predictions = PG313._predictions(model, rows_list, vocab, device)
    return {"bound_probe": PG313._bound_metrics(rows_list, predictions), "repair": PG316._repair_metrics(rows_list, predictions)}


def _drop(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    def score(item: Mapping[str, Any]) -> float:
        bound = item.get("bound_probe") or {}
        return float(bound.get("missing_question_recall") or 0.0) + float(bound.get("variant_exact") or 0.0) + float((item.get("repair") or {}).get("repair_exact") or 0.0) - 2.0 * float(bound.get("hard_negative_false_allow") or 0.0)
    return round(max(0.0, score(before) - score(after)), 6)


def _aggregate(values: Sequence[Mapping[str, Any]], section: str, key: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key) or 0.0) for value in values]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def _expand_vocabulary_model(model: Any, vocabulary: Mapping[str, int], missing: Sequence[str]) -> tuple[Any, dict[str, int]]:
    """Append unseen abstract surface tokens without changing old token ids."""

    expanded = dict(vocabulary)
    for token in sorted(set(str(item) for item in missing)):
        expanded[token] = len(expanded)
    if len(expanded) == len(vocabulary):
        return model, expanded
    resized = CausalMoELanguageModel(vocab_size=len(expanded), config=model.config).to("cpu")
    old_state = model.state_dict()
    new_state = resized.state_dict()
    for key, value in old_state.items():
        if key in {"token_embedding.weight", "lm_head.weight"}:
            new_state[key][: value.shape[0]] = value.detach().cpu()
        elif key in new_state:
            new_state[key] = value.detach().cpu()
    resized.load_state_dict(new_state)
    return resized, expanded


def main() -> int:
    _require_gate()
    lattice = _load(LATTICE)
    lattice_audit = _load(LATTICE_AUDIT)
    pg319 = _load(PG319_DATASET)
    pg319_report = _load(PG319_REPORT)
    trace = _load(PG318_TRACE)
    if lattice_audit.get("status") != "passed":
        raise RuntimeError("PG-320 requires passed observation lattice audit")
    if pg319_report.get("promotion", {}).get("training_allowed") is not False or trace.get("training_eligible") is not False:
        raise RuntimeError("PG-320 refuses a promotable holdout")
    lattice_train = [dict(row) for row in lattice.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    lattice_holdout = [dict(row) for row in lattice.get("records", []) if row.get("split") == "lattice_holdout"]
    prior_train = [dict(row) for row in pg319.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    replay_rows = [row for index, row in enumerate(prior_train) if index % 2 == 0]
    old_holdout = [dict(row) for row in pg319.get("records", []) if row.get("split") in {"implementation_holdout", "real_live_holdout"} and str(row.get("source")) != "pg246_vulnerableapp_independent_dom_holdout"]
    hard = [dict(row) for row in pg319.get("records", []) if row.get("split") == "hard_negative_eval" or (bool(row.get("hard_negative")) and row.get("split") != "train")]
    family = [dict(row) for row in trace.get("episodes", [])] + _preflight_rows(trace)
    vocab_source = lattice_train + lattice_holdout + prior_train + old_holdout + hard + family
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_vocab: dict[str, int] | None = None
    best_score = float("-inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    weights = {"question=ask_typed_availability": 12.0, "question=ask_replay_readiness": 12.0, "question=ask_evidence_presence": 12.0, "question=ask_feedback_state": 12.0, "question=ask_negative_control": 12.0, "question=ask_fresh_reset": 12.0, "safe_to_send=0": 7.0, "safe_to_send=1": 2.0, "next_action=request_observation": 10.0, "next_action=repair_abstract_plan": 7.0, "repair_action=retry_bounded_variant": 7.0, "probe_variant_ref=none": 7.0}
    device = torch.device("cpu")
    for seed in SEEDS:
        base_path = BASE_CHECKPOINT_DIR / f"pg319_cross_impl_moe_seed_{seed}.pt"
        base_model, vocab, symbolic = PG314.load_causal_checkpoint(base_path, device)
        if not symbolic:
            raise RuntimeError(f"PG-320 base checkpoint {seed} is not symbolic")
        missing = sorted({str(token) for row in vocab_source for token in (row.get("context_tokens") or []) + (row.get("target_tokens") or []) if str(token) not in vocab})
        base_model, vocab = _expand_vocabulary_model(base_model, vocab, missing)
        before = {"family": _lane(base_model, family, vocab, device), "lattice": _lane(base_model, lattice_holdout, vocab, device), "old": _lane(base_model, old_holdout, vocab, device), "hard": _lane(base_model, hard, vocab, device)}
        new_only = PG313.train_causal_moe(lattice_train, vocab, device, seed=seed + 500, config=PG313.CausalMoEConfig(**base_model.config.__dict__) if hasattr(base_model, "config") else PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72), epochs=80, learning_rate=0.0005, token_weights=weights, initial_state=base_model.state_dict())
        mix = PG313.train_causal_moe(lattice_train + replay_rows, vocab, device, seed=seed + 600, config=PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72), epochs=80, learning_rate=0.0005, token_weights=weights, initial_state=base_model.state_dict())
        new_metrics = {"family": _lane(new_only, family, vocab, device), "lattice": _lane(new_only, lattice_holdout, vocab, device), "old": _lane(new_only, old_holdout, vocab, device), "hard": _lane(new_only, hard, vocab, device)}
        mix_metrics = {"family": _lane(mix, family, vocab, device), "lattice": _lane(mix, lattice_holdout, vocab, device), "old": _lane(mix, old_holdout, vocab, device), "hard": _lane(mix, hard, vocab, device)}
        score = 2.0 * float((mix_metrics["family"]["bound_probe"].get("missing_question_recall") or 0.0)) + float((mix_metrics["family"]["bound_probe"].get("variant_exact") or 0.0)) - 4.0 * float((mix_metrics["hard"]["bound_probe"].get("hard_negative_false_allow") or 0.0))
        checkpoint = CHECKPOINT_DIR / f"pg320_question_lattice_seed_{seed}.pt"
        torch.save({"schema_version": "pg320-observation-lattice-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "seed": seed, "source_checkpoint": str(base_path.relative_to(ROOT)), "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": {"d_model": 64, "n_heads": 4, "n_layers": 2, "experts": 2, "expert_hidden": 128, "top_k": 1, "dropout": 0.0, "max_length": 72}, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in mix.state_dict().items()}, "lattice_dataset_sha256": lattice.get("dataset_sha256"), "lattice_audit_sha256": lattice_audit.get("audit_sha256"), "base_report_sha256": pg319_report.get("report_sha256"), "promotion_blocked": True}, checkpoint)
        results.append({"seed": seed, "before": before, "new_only": new_metrics, "replay_mix": mix_metrics, "new_only_old_drop": _drop(before["old"], new_metrics["old"]), "replay_mix_old_drop": _drop(before["old"], mix_metrics["old"]), "selection_score": round(score, 6), "checkpoint": str(checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = mix
            best_vocab = dict(vocab)
    if best_model is None:
        raise RuntimeError("PG-320 did not produce a checkpoint")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda item: item["selection_score"])
    source_ck = BASE_CHECKPOINT_DIR / f"pg319_cross_impl_moe_seed_{best['seed']}.pt"
    source = torch.load(source_ck, map_location="cpu", weights_only=False)
    torch.save({"schema_version": "pg320-observation-lattice-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "selected_seed": best["seed"], "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": source["config"], "vocabulary": best_vocab or source["vocabulary"], "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "lattice_dataset_sha256": lattice.get("dataset_sha256"), "lattice_audit_sha256": lattice_audit.get("audit_sha256"), "base_report_sha256": pg319_report.get("report_sha256"), "promotion_blocked": True}, CHECKPOINT)
    family_values = [item["replay_mix"]["family"] for item in results]
    lattice_values = [item["replay_mix"]["lattice"] for item in results]
    hard_values = [item["replay_mix"]["hard"] for item in results]
    metrics = {"family_question_min": _aggregate(family_values, "bound_probe", "missing_question_recall"), "family_variant_min": _aggregate(family_values, "bound_probe", "variant_exact"), "family_false_allow_max": _aggregate(family_values, "bound_probe", "hard_negative_false_allow"), "lattice_question_min": _aggregate(lattice_values, "bound_probe", "missing_question_recall"), "lattice_variant_min": _aggregate(lattice_values, "bound_probe", "variant_exact"), "hard_false_allow_max": _aggregate(hard_values, "bound_probe", "hard_negative_false_allow"), "new_only_old_drop_max": round(max(item["new_only_old_drop"] for item in results), 6), "replay_mix_old_drop_max": round(max(item["replay_mix_old_drop"] for item in results), 6), "best_seed": best["seed"]}
    report = {"protocol_id": "pg-pk-320-observation-lattice-finetune-v1", "schema_version": "pg320-observation-lattice-finetune-report-v1", "status": "completed_local_morning_pg320_observation_lattice", "sources": {"lattice": str(LATTICE.relative_to(ROOT)), "lattice_sha256": lattice.get("dataset_sha256"), "lattice_audit": str(LATTICE_AUDIT.relative_to(ROOT)), "lattice_audit_sha256": lattice_audit.get("audit_sha256"), "base_training_report": str(PG319_REPORT.relative_to(ROOT)), "base_report_sha256": pg319_report.get("report_sha256"), "family_holdout_trace": str(PG318_TRACE.relative_to(ROOT)), "family_holdout_trace_sha256": trace.get("trace_sha256")}, "training": {"architecture": "causal_transformer_moe_next_token", "target": "observation_priority_rule_ir_question_then_assembly", "device": "cpu", "seeds": list(SEEDS), "lattice_train_count": len(lattice_train), "lattice_holdout_count": len(lattice_holdout), "replay_rows": len(replay_rows), "epochs": 80, "wire_emission": False, "raw_payload_in_context": False, "raw_response_body_in_context": False}, "metrics": metrics, "per_seed": results, "catastrophic_forgetting_canary": {"new_only_drop_max": metrics["new_only_old_drop_max"], "replay_mix_drop_max": metrics["replay_mix_old_drop_max"], "threshold": 0.05, "replay_mix_within_threshold": metrics["replay_mix_old_drop_max"] <= 0.05}, "hypothesis_gate": {"status": "blocked", "checks": {"lattice_question_min": metrics["lattice_question_min"]["min"] >= 0.95, "family_question_min": metrics["family_question_min"]["min"] >= 0.9, "family_variant_min": metrics["family_variant_min"]["min"] >= 0.9, "family_zero_false_allow": metrics["family_false_allow_max"]["max"] == 0, "hard_zero_false_allow": metrics["hard_false_allow_max"]["max"] == 0, "replay_forgetting_within_threshold": metrics["replay_mix_old_drop_max"] <= 0.05, "promotion_blocked": True}, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"}, "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "elapsed_seconds": round(time.monotonic() - started, 3), "report_sha256": ""}
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gate": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
