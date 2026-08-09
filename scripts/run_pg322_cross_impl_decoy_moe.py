"""PG-322 decoder-only Rule-IR fine-tune on a blind cross-implementation mix."""

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


PG313 = _load_module("pg313_for_pg322", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG314 = _load_module("pg314_for_pg322", ROOT / "scripts" / "run_pg314_independent_variant_replay.py")
PG316 = _load_module("pg316_for_pg322", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
PG320 = _load_module("pg320_for_pg322", ROOT / "scripts" / "run_pg320_observation_lattice_finetune.py")
from app.pg295_causal_moe import CausalMoELanguageModel  # noqa: E402

RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg322_cross_impl_decoy_dataset_v1.json"
AUDIT = RESEARCH / "pg322_cross_impl_decoy_dataset_audit_v1.json"
ROLE_DATASET = RESEARCH / "pg321_variant_role_lattice_dataset_v1.json"
PG320_DATASET = RESEARCH / "pg320_observation_lattice_dataset_v1.json"
PG321_TRACE = RESEARCH / "pg321_family_holdout_trace_v1.json"
REPORT = RESEARCH / "pg322_cross_impl_decoy_moe_training_report_v1_local_morning.json"
BASE_DIR = ROOT / "artifacts" / "pg321-variant-role" / "seeds"
OUT_DIR = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "seeds"
CHECKPOINT = ROOT / "artifacts" / "pg322-cross-impl-decoy" / "pg322_cross_impl_decoy_moe_local_morning.pt"
BASE_PREFIX = "pg321_variant_role_seed_"
SEEDS = (31901, 31902, 31903)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _remote_a800_enabled() -> bool:
    """Return whether this invocation is the explicitly authorized A800 lane.

    The default remains the historical local-morning CPU path.  The remote
    lane is intentionally opt-in and is constrained to the first visible
    device so a training run cannot silently fan out across the host.
    """

    return os.environ.get("BLACKBOX_REMOTE_A800_TRAIN") == "1"


def _execution_mode() -> str:
    return "remote_a800_gpu0" if _remote_a800_enabled() else "local_morning_cpu"


def _training_device() -> torch.device:
    return torch.device("cuda:0" if _remote_a800_enabled() else "cpu")


def _gate() -> None:
    if _remote_a800_enabled():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("remote A800 training requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("remote A800 training requires exactly one visible CUDA device")
        torch.cuda.set_device(0)
        name = torch.cuda.get_device_name(0)
        if "A800" not in str(name):
            raise RuntimeError(f"remote A800 training requires an A800, got {name!r}")
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        return
    if os.environ.get("BLACKBOX_LOCAL_MORNING_TRAIN") != "1":
        raise RuntimeError("PG-322 requires BLACKBOX_LOCAL_MORNING_TRAIN=1")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if not (8 <= now.hour < 18):
        raise RuntimeError(f"PG-322 local training is limited to 08:00-18:00 Asia/Shanghai (now {now.isoformat()})")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def _expand(model: Any, vocab: Mapping[str, int], rows: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, int]]:
    missing = sorted({str(token) for row in rows for token in (row.get("context_tokens") or []) + (row.get("target_tokens") or []) if str(token) not in vocab})
    if not missing:
        return model, dict(vocab)
    expanded = dict(vocab)
    for token in missing:
        expanded[token] = len(expanded)
    model_device = next(model.parameters()).device
    resized = CausalMoELanguageModel(vocab_size=len(expanded), config=model.config).to(model_device)
    old_state = model.state_dict()
    new_state = resized.state_dict()
    for key, value in old_state.items():
        if key in {"token_embedding.weight", "lm_head.weight"}:
            new_state[key][: value.shape[0]] = value.detach().to(model_device)
        elif key in new_state:
            new_state[key] = value.detach().to(model_device)
    resized.load_state_dict(new_state)
    return resized, expanded


def _lane(model: Any, rows: Sequence[Mapping[str, Any]], vocab: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    material = [dict(row) for row in rows]
    preds = PG313._predictions(model, material, vocab, device)
    return {"bound_probe": PG313._bound_metrics(material, preds), "repair": PG316._repair_metrics(material, preds)}


def _score(lane: Mapping[str, Any]) -> float:
    bound = lane.get("bound_probe") or {}
    repair = lane.get("repair") or {}
    return float(bound.get("missing_question_recall") or 0.0) + float(bound.get("variant_exact") or 0.0) + float(repair.get("repair_exact") or 0.0) - 4.0 * float(bound.get("hard_negative_false_allow") or 0.0)


def _drop(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    return round(max(0.0, _score(before) - _score(after)), 6)


def _agg(values: Sequence[Mapping[str, Any]], section: str, key: str) -> dict[str, Any]:
    """Aggregate only measured values; an empty denominator is not zero.

    In particular, a holdout with no missing-question rows must be reported as
    ``not_applicable`` rather than as a false 0% question-recall result.
    """

    nums = [float((value.get(section) or {}).get(key)) for value in values if (value.get(section) or {}).get(key) is not None]
    if not nums:
        return {"mean": None, "min": None, "max": None, "count": 0, "status": "not_applicable"}
    return {"mean": round(sum(nums) / len(nums), 6), "min": round(min(nums), 6), "max": round(max(nums), 6), "count": len(nums), "status": "measured"}


def main() -> int:
    _gate()
    dataset = _load(DATASET)
    audit = _load(AUDIT)
    role = _load(ROLE_DATASET)
    lattice = _load(PG320_DATASET)
    trace = _load(PG321_TRACE)
    if audit.get("status") != "passed":
        raise RuntimeError("PG-322 requires passed cross-implementation/decoy dataset audit")
    pg322_train = [dict(row) for row in dataset.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    implementation_holdout = [dict(row) for row in dataset.get("records", []) if row.get("split") == "implementation_holdout"]
    third_holdout = [dict(row) for row in dataset.get("records", []) if row.get("split") == "third_surface_holdout"]
    ask_holdout = [dict(row) for row in dataset.get("records", []) if row.get("split") == "ask_holdout"]
    hard = [dict(row) for row in dataset.get("records", []) if row.get("split") == "hard_negative_eval"]
    role_train = [dict(row) for row in role.get("records", []) if row.get("split") == "train" and row.get("training_eligible")]
    lattice_replay = [dict(row) for row in lattice.get("records", []) if row.get("split") == "train" and row.get("training_eligible")][::3]
    family = [dict(row) for row in trace.get("episodes", [])] + PG320._preflight_rows(trace)
    # Replay keeps old Rule-IR and ASK behavior alive; all fresh holdouts stay
    # evaluation-only and are never mixed into the gradient.
    mix_rows = pg322_train + role_train + lattice_replay
    weights = {
        "question=ask_typed_availability": 8.0,
        "question=ask_replay_readiness": 8.0,
        "question=ask_evidence_presence": 8.0,
        "question=ask_feedback_state": 8.0,
        "question=ask_negative_control": 8.0,
        "question=ask_fresh_reset": 8.0,
        "safe_to_send=0": 9.0,
        "safe_to_send=1": 2.0,
        "next_action=request_observation": 8.0,
        "next_action=repair_abstract_plan": 8.0,
        "probe_variant_ref=source_attested_candidate": 11.0,
        "probe_variant_ref=reference_canary": 11.0,
        "probe_variant_ref=negative_control": 11.0,
        "probe_variant_ref=none": 9.0,
        "encoding_chain_ref=surface_encoding": 8.0,
    }
    device = _training_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    best_model: Any = None
    best_vocab: dict[str, int] | None = None
    best_score = float("-inf")
    started = time.monotonic()
    for seed in SEEDS:
        base, vocab, symbolic = PG314.load_causal_checkpoint(BASE_DIR / f"{BASE_PREFIX}{seed}.pt", device)
        if not symbolic:
            raise RuntimeError(f"PG-322 base checkpoint {seed} is not symbolic")
        all_rows = mix_rows + implementation_holdout + third_holdout + ask_holdout + hard + family
        base, vocab = _expand(base, vocab, all_rows)
        before = {"implementation": _lane(base, implementation_holdout, vocab, device), "third": _lane(base, third_holdout, vocab, device), "ask": _lane(base, ask_holdout, vocab, device), "hard": _lane(base, hard, vocab, device), "family": _lane(base, family, vocab, device)}
        model = PG313.train_causal_moe(mix_rows, vocab, device, seed=seed + 900, config=PG313.CausalMoEConfig(d_model=64, n_heads=4, n_layers=2, experts=2, expert_hidden=128, top_k=1, dropout=0.0, max_length=72), epochs=70, learning_rate=0.00035, token_weights=weights, initial_state=base.state_dict())
        after = {"implementation": _lane(model, implementation_holdout, vocab, device), "third": _lane(model, third_holdout, vocab, device), "ask": _lane(model, ask_holdout, vocab, device), "hard": _lane(model, hard, vocab, device), "family": _lane(model, family, vocab, device)}
        score = _score(after["implementation"]) + _score(after["third"]) + _score(after["ask"])
        checkpoint = OUT_DIR / f"pg322_cross_impl_decoy_seed_{seed}.pt"
        torch.save({"schema_version": "pg322-cross-impl-decoy-moe-checkpoint-v1", "assignment": {"execution_mode": _execution_mode(), "seed": seed, "device": str(device), "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": {"d_model": 64, "n_heads": 4, "n_layers": 2, "experts": 2, "expert_hidden": 128, "top_k": 1, "dropout": 0.0, "max_length": 72}, "vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "promotion_blocked": True}, checkpoint)
        item = {"seed": seed, "before": before, "after": after, "old_family_drop": _drop(before["family"], after["family"]), "selection_score": round(score, 6), "checkpoint": str(checkpoint.relative_to(ROOT))}
        results.append(item)
        if score > best_score:
            best_score = score
            best_model = model
            best_vocab = dict(vocab)
    best = max(results, key=lambda item: item["selection_score"])
    source = torch.load(OUT_DIR / f"pg322_cross_impl_decoy_seed_{best['seed']}.pt", map_location="cpu", weights_only=False)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg322-cross-impl-decoy-moe-checkpoint-v1", "assignment": {"execution_mode": _execution_mode(), "selected_seed": best["seed"], "device": str(device), "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, "config": source["config"], "vocabulary": best_vocab or source["vocabulary"], "state": {key: value.detach().cpu() for key, value in best_model.state_dict().items()}, "dataset_sha256": dataset.get("dataset_sha256"), "audit_sha256": audit.get("audit_sha256"), "promotion_blocked": True}, CHECKPOINT)
    metrics = {
        "implementation_question_min": _agg([item["after"]["implementation"] for item in results], "bound_probe", "missing_question_recall"),
        "implementation_variant_min": _agg([item["after"]["implementation"] for item in results], "bound_probe", "variant_exact"),
        "third_surface_question_min": _agg([item["after"]["third"] for item in results], "bound_probe", "missing_question_recall"),
        "third_surface_variant_min": _agg([item["after"]["third"] for item in results], "bound_probe", "variant_exact"),
        "ask_question_min": _agg([item["after"]["ask"] for item in results], "bound_probe", "missing_question_recall"),
        "ask_unsafe_allow_max": _agg([item["after"]["ask"] for item in results], "bound_probe", "hard_negative_false_allow"),
        "hard_false_allow_max": _agg([item["after"]["hard"] for item in results], "bound_probe", "hard_negative_false_allow"),
        "family_question_min": _agg([item["after"]["family"] for item in results], "bound_probe", "missing_question_recall"),
        "family_variant_min": _agg([item["after"]["family"] for item in results], "bound_probe", "variant_exact"),
        "old_family_drop_max": round(max(item["old_family_drop"] for item in results), 6),
        "best_seed": best["seed"],
    }
    report = {
        "protocol_id": "pg-pk-322-cross-impl-decoy-moe-v1",
        "schema_version": "pg322-cross-impl-decoy-moe-training-report-v1",
        "status": "completed_local_morning_pg322_cross_impl_decoy",
        "sources": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": dataset.get("dataset_sha256"), "audit": str(AUDIT.relative_to(ROOT)), "audit_sha256": audit.get("audit_sha256"), "role_dataset": str(ROLE_DATASET.relative_to(ROOT)), "lattice_dataset": str(PG320_DATASET.relative_to(ROOT)), "family_trace": str(PG321_TRACE.relative_to(ROOT))},
        "training": {"architecture": "causal_transformer_moe_next_token", "target": "abstract Rule-IR payload assembly with role-conditioned probe variant", "device": str(device), "execution_mode": _execution_mode(), "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "seeds": list(SEEDS), "train_count": len(mix_rows), "implementation_holdout_count": len(implementation_holdout), "third_surface_holdout_count": len(third_holdout), "ask_holdout_count": len(ask_holdout), "hard_negative_count": len(hard), "wire_emission": False, "raw_payload_in_context": False, "raw_response_body_in_context": False},
        "metrics": metrics,
        "per_seed": results,
        "hypothesis_gate": {"status": "blocked", "checks": {"implementation_variant": metrics["implementation_variant_min"]["min"] >= 0.9, "third_surface_ask": metrics["third_surface_question_min"]["min"] >= 0.9, "ask_question": metrics["ask_question_min"]["min"] >= 0.95, "ask_zero_unsafe_allow": metrics["ask_unsafe_allow_max"]["max"] == 0, "hard_zero_false_allow": metrics["hard_false_allow_max"]["max"] == 0, "family_question": metrics["family_question_min"]["min"] >= 0.9, "old_retention": metrics["old_family_drop_max"] <= 0.05, "promotion_blocked": True}, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False, "checkpoint_role": "research_candidate_only"},
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "metrics": metrics, "gate": report["hypothesis_gate"], "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
