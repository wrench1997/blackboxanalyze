"""Evaluate PG-319 seed checkpoints on the complete frozen PG-318 family lane.

The original PG-319 training run intentionally kept the live episodes and
multi-missing preflight separate.  This evaluator makes that distinction
explicit: the 270 preflight rows are reconstructed from their missing-slot
metadata and scored alongside the 72 typed-process rows.  It never trains or
contacts a target.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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


PG313 = _load_module("pg313_for_pg319_eval", ROOT / "scripts" / "run_pg313_probe_variant_moe.py")
PG314 = _load_module("pg314_for_pg319_eval", ROOT / "scripts" / "run_pg314_independent_variant_replay.py")
PG316 = _load_module("pg316_for_pg319_eval", ROOT / "scripts" / "run_pg316_failure_repair_moe.py")
from app.pg301_payload_assembly import OBSERVATION_KEYS, canonical_assembly_context  # noqa: E402
from app.pg313_probe_variant import probe_target_for_context  # noqa: E402

RESEARCH = ROOT / "research"
TRACE = RESEARCH / "pg318_family_holdout_trace_v1.json"
REPORT = RESEARCH / "pg319_family_holdout_checkpoint_evaluation_v1.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "pg319-cross-impl" / "seeds"
SEEDS = (31901, 31902, 31903)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _preflight_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
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
        rows.append({"record_id": f"pg318-preflight:{index}", "context_tokens": context, "target_tokens": probe_target_for_context(context), "split": "family_holdout_eval", "raw_payload_stored": False, "raw_response_body_stored": False})
    return rows


def main() -> int:
    trace = json.loads(TRACE.read_text(encoding="utf-8-sig"))
    if trace.get("training_eligible") is not False or trace.get("memory_promotion_allowed") is not False:
        raise RuntimeError("frozen PG-318 trace must remain evaluation-only")
    rows = [dict(row) for row in trace.get("episodes") or []] + _preflight_rows(trace)
    device = torch.device("cpu")
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        checkpoint = CHECKPOINT_DIR / f"pg319_cross_impl_moe_seed_{seed}.pt"
        model, vocabulary, symbolic = PG314.load_causal_checkpoint(checkpoint, device)
        if not symbolic:
            raise RuntimeError(f"seed {seed} checkpoint is not symbolic")
        predictions = PG313._predictions(model, rows, vocabulary, device)
        bound = PG313._bound_metrics(rows, predictions)
        repair = PG316._repair_metrics(rows, predictions)
        per_seed.append({"seed": seed, "checkpoint": str(checkpoint.relative_to(ROOT)), "bound_probe": bound, "repair": repair, "predictions_training_ineligible": all(not row.get("training_eligible", False) for row in rows)})
    def agg(key: str) -> dict[str, float]:
        vals = [float((item["bound_probe"].get(key) or 0.0)) for item in per_seed]
        return {"mean": round(sum(vals) / len(vals), 6), "min": round(min(vals), 6), "max": round(max(vals), 6)}
    metrics = {"family_question_recall": agg("missing_question_recall"), "family_variant_exact": agg("variant_exact"), "family_false_allow": agg("hard_negative_false_allow"), "sequence_exact": agg("sequence_exact_accuracy"), "repair_exact": {"mean": round(sum(float(item["repair"].get("repair_exact") or 0.0) for item in per_seed) / len(per_seed), 6), "min": round(min(float(item["repair"].get("repair_exact") or 0.0) for item in per_seed), 6), "max": round(max(float(item["repair"].get("repair_exact") or 0.0) for item in per_seed), 6)}}
    result = {"protocol_id": "pg-pk-319-family-holdout-checkpoint-eval-v1", "schema_version": "pg319-family-holdout-checkpoint-evaluation-v1", "status": "completed_frozen_family_holdout_evaluation", "source": {"trace": str(TRACE.relative_to(ROOT)), "trace_sha256": trace.get("trace_sha256"), "checkpoint_dir": str(CHECKPOINT_DIR.relative_to(ROOT))}, "counts": {"seed_count": len(SEEDS), "episode_rows": len(trace.get("episodes") or []), "multi_missing_rows": len(rows) - len(trace.get("episodes") or []), "total_rows": len(rows)}, "metrics": metrics, "per_seed": per_seed, "hypothesis_gate": {"status": "blocked", "checks": {"question_min": metrics["family_question_recall"]["min"] >= 0.9, "variant_min": metrics["family_variant_exact"]["min"] >= 0.9, "zero_false_allow": metrics["family_false_allow"]["max"] == 0, "repair_min": metrics["repair_exact"]["min"] >= 0.9, "all_rows_training_ineligible": all(item["predictions_training_ineligible"] for item in per_seed), "promotion_blocked": True}, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "report_sha256": ""}
    result["report_sha256"] = _digest(result)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "metrics": result["metrics"], "gate": result["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
