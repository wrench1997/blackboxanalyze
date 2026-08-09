"""PG-321 fine-tune variant-role selection without sacrificing ASK safety."""

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


PG313 = _load_module("pg313_for_pg321", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG314 = _load_module("pg314_for_pg321", ROOT / "scripts" / "run_pg314_independent_variant_replay.py")
PG316 = _load_module("pg316_for_pg321", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
PG320 = _load_module("pg320_for_pg321", ROOT / "scripts" / "run_pg320_observation_lattice_finetune.py")
from app.pg295_causal_moe import CausalMoELanguageModel  # noqa: E402
from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
ROLE_DATASET = RESEARCH / "pg321_variant_role_lattice_dataset_v1.json"
ROLE_AUDIT = RESEARCH / "pg321_variant_role_lattice_dataset_audit_v1.json"
PG319_DATASET = RESEARCH / "pg319_cross_impl_rule_ir_dataset_v1.json"
PG320_DATASET = RESEARCH / "pg320_observation_lattice_dataset_v1.json"
PG320_REPORT = RESEARCH / "pg320_observation_lattice_finetune_report_v1_local_morning.json"
PG318_TRACE = RESEARCH / "pg318_family_holdout_trace_v1.json"
REPORT = RESEARCH / "pg321_variant_role_finetune_report_v1_local_morning.json"
BASE_DIR = ROOT / "artifacts" / "pg320-observation-lattice" / "seeds"
OUT_DIR = ROOT / "artifacts" / "pg321-variant-role" / "seeds"
CHECKPOINT = ROOT / "artifacts" / "pg321-variant-role" / "pg321_variant_role_moe_local_morning.pt"
SEEDS = (31901, 31902, 31903)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_gate() -> None:
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-321 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-321 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _preflight_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    return PG320._preflight_rows(trace)


def _expand_vocabulary_model(model: Any, vocabulary: Mapping[str, int], missing: Sequence[str]) -> tuple[Any, dict[str, int]]:
    """Append role tokens while preserving every PG-320 token id and weight."""

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


def _lane(model: Any, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    data = [dict(row) for row in rows]
    preds = PG313._predictions(model, data, vocab, device)
    return {"bound_probe": PG313._bound_metrics(data, preds), "repair": PG316._repair_metrics(data, preds)}


def _score(item: Mapping[str, Any]) -> float:
    bound = item.get("bound_probe") or {}
    return float(bound.get("missing_question_recall") or 0.0) + float(bound.get("variant_exact") or 0.0) + float((item.get("repair") or {}).get("repair_exact") or 0.0) - 4.0 * float(bound.get("hard_negative_false_allow") or 0.0)


def _drop(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    return round(max(0.0, _score(before) - _score(after)), 6)


def _agg(values: Sequence[Mapping[str, Any]], section: str, key: str) -> dict[str, float]:
    nums = [float((value.get(section) or {}).get(key) or 0.0) for value in values]
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6)} if nums else {"mean": 0.0, "min": 0.0, "max": 0.0}


def main() -> int:
    _require_gate()
    role = _load(ROLE_DATASET)
    role_audit = _load(ROLE_AUDIT)
    pg319 = _load(PG319_DATASET)
    pg320 = _load(PG320_DATASET)
    pg320_report = _load(PG320_REPORT)
    trace = _load(PG318_TRACE)
    if role_audit.get("status") != "passed" or pg320_report.get("promotion", {}).get("training_allowed") is not False:
        raise RuntimeError("PG-321 requires audited role data and blocked PG-320 promotion")
    role_train = [dict(row) for row in role.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    role_holdout = [dict(row) for row in role.get("records", []) if row.get("split") == "variant_holdout"]
    replay = [dict(row) for index, row in enumerate(pg319.get("records", [])) if row.get("split") == "train" and index % 2 == 0]
    lattice_replay = [dict(row) for row in pg320.get("records", []) if row.get("split") == "train" and row.get("training_eligible")][::2]
    family = [dict(row) for row in trace.get("episodes", [])] + _preflight_rows(trace)
    hard = [dict(row) for row in pg319.get("records", []) if row.get("split") == "hard_negative_eval" or (bool(row.get("hard_negative")) and row.get("split") != "train")]
    old = [dict(row) for row in pg319.get("records", []) if row.get("split") in {"implementation_holdout", "real_live_holdout"}]
    mix_rows = role_train + replay + lattice_replay
    weights = {"question=ask_typed_availability": 7.0, "question=ask_replay_readiness": 7.0, "question=ask_evidence_presence": 7.0, "question=ask_feedback_state": 7.0, "question=ask_negative_control": 7.0, "question=ask_fresh_reset": 7.0, "safe_to_send=0": 8.0, "safe_to_send=1": 2.0, "next_action=request_observation": 7.0, "next_action=repair_abstract_plan": 8.0, "probe_variant_ref=source_attested_candidate": 12.0, "probe_variant_ref=reference_canary": 12.0, "probe_variant_ref=negative_control": 12.0, "probe_variant_ref=none": 8.0, "encoding_chain_ref=surface_encoding": 8.0}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_vocab: dict[str, int] | None = None
    best_score = float("-inf")
    started = time.monotonic()
    device = torch.device("cpu")
    for seed in SEEDS:
        base, vocab, symbolic = PG314.load_causal_checkpoint(BASE_DIR / f"pg320_question_lattice_seed_{seed}.pt", device)
        if not symbolic:
            raise RuntimeError(f"PG-321 base checkpoint {seed} is not symbolic")
        missing = sorted({str(token) for row in mix_rows + role_holdout + family + hard + old for token in (row.get("context_tokens") or []) + (row.get("target_tokens") or []) if str(token) not in vocab})
        base, vocab = _expand_vocabulary_model(base, vocab, missing)
        before = {"role": _lane(base, role_holdout, vocab, device), "family": _lane(base, family, vocab, device), "old": _lane(base, old, vocab, device), "hard": _lane(base, hard, vocab, device)}
        model = PG313.train_causal_moe(mix_rows, vocab, device, seed=seed + 700, config=PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72), epochs=90, learning_rate=0.00035, token_weights=weights, initial_state=base.state_dict())
        after = {"role": _lane(model, role_holdout, vocab, device), "family": _lane(model, family, vocab, device), "old": _lane(model, old, vocab, device), "hard": _lane(model, hard, vocab, device)}
        score = _score(after["family"]) + _score(after["role"])
        checkpoint = OUT_DIR / f"pg321_variant_role_seed_{seed}.pt"
        torch.save({"schema_version": "pg321-variant-role-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "seed": seed, "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": {"d_model": 64, "n_heads": 4, "n_layers": 2, "experts": 2, "expert_hidden": 128, "top_k": 1, "dropout": 0.0, "max_length": 72}, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "role_dataset_sha256": role.get("dataset_sha256"), "role_audit_sha256": role_audit.get("audit_sha256"), "promotion_blocked": True}, checkpoint)
        results.append({"seed": seed, "before": before, "after": after, "old_drop": _drop(before["old"], after["old"]), "selection_score": round(score, 6), "checkpoint": str(checkpoint.relative_to(ROOT))})
        if score > best_score:
            best_score = score
            best_model = model
            best_vocab = dict(vocab)
    if best_model is None:
        raise RuntimeError("PG-321 did not produce a checkpoint")
    best = max(results, key=lambda item: item["selection_score"])
    src = torch.load(OUT_DIR / f"pg321_variant_role_seed_{best['seed']}.pt", map_location="cpu", weights_only=False)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg321-variant-role-moe-checkpoint-v1", "assignment": {"execution_mode": "local_morning_cpu", "selected_seed": best["seed"], "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": src["config"], "vocabulary": best_vocab or src["vocabulary"], "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "role_dataset_sha256": role.get("dataset_sha256"), "role_audit_sha256": role_audit.get("audit_sha256"), "promotion_blocked": True}, CHECKPOINT)
    after_family = [item["after"]["family"] for item in results]
    after_role = [item["after"]["role"] for item in results]
    after_hard = [item["after"]["hard"] for item in results]
    metrics = {"family_question_min": _agg(after_family, "bound_probe", "missing_question_recall"), "family_variant_min": _agg(after_family, "bound_probe", "variant_exact"), "family_false_allow_max": _agg(after_family, "bound_probe", "hard_negative_false_allow"), "role_holdout_variant_min": _agg(after_role, "bound_probe", "variant_exact"), "role_holdout_question_min": _agg(after_role, "bound_probe", "missing_question_recall"), "hard_false_allow_max": _agg(after_hard, "bound_probe", "hard_negative_false_allow"), "old_drop_max": round(max(item["old_drop"] for item in results), 6), "best_seed": best["seed"]}
    report = {"protocol_id": "pg-pk-321-variant-role-finetune-v1", "schema_version": "pg321-variant-role-finetune-report-v1", "status": "completed_local_morning_pg321_variant_role", "sources": {"role_dataset": str(ROLE_DATASET.relative_to(ROOT)), "role_dataset_sha256": role.get("dataset_sha256"), "role_audit": str(ROLE_AUDIT.relative_to(ROOT)), "role_audit_sha256": role_audit.get("audit_sha256"), "base_pg320_report": str(PG320_REPORT.relative_to(ROOT)), "base_pg320_report_sha256": pg320_report.get("report_sha256"), "family_trace": str(PG318_TRACE.relative_to(ROOT)), "family_trace_sha256": trace.get("trace_sha256")}, "training": {"architecture": "causal_transformer_moe_next_token", "target": "history_conditioned_probe_variant_and_encoding_chain_assembly", "device": "cpu", "seeds": list(SEEDS), "role_train_count": len(role_train), "role_holdout_count": len(role_holdout), "replay_count": len(replay), "lattice_replay_count": len(lattice_replay), "epochs": 90, "wire_emission": False, "raw_payload_in_context": False, "raw_response_body_in_context": False}, "metrics": metrics, "per_seed": results, "hypothesis_gate": {"status": "blocked", "checks": {"role_variant_min": metrics["role_holdout_variant_min"]["min"] >= 0.95, "family_question_min": metrics["family_question_min"]["min"] >= 0.9, "family_variant_min": metrics["family_variant_min"]["min"] >= 0.9, "family_zero_false_allow": metrics["family_false_allow_max"]["max"] == 0, "hard_zero_false_allow": metrics["hard_false_allow_max"]["max"] == 0, "old_retention_drop": metrics["old_drop_max"] <= 0.05, "promotion_blocked": True}, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"}, "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "elapsed_seconds": round(time.monotonic() - started, 3), "report_sha256": ""}
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gate": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
