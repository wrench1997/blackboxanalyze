"""PG-85: composite JSON/HTML/DOM training with independent holdouts.

PG-84 showed that a JSON-only geometry vocabulary cannot transfer to a
legacy HTML/DOM surface.  PG-85 adds only two PG-74 seeds to the training
partition and keeps the third PG-74 seed as a cross-dataset holdout.  PG-82
source/implementation holdout and PG-76 unknown-family abstention remain
separate gates.  No evaluator labels, family/source names or raw bodies are
used as model tokens.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG82_SCRIPT = ROOT / "scripts" / "train_pg82_effect_geometry_source_holdout.py"
PG84_SCRIPT = ROOT / "scripts" / "run_pg84_cross_dataset_frozen_replay.py"
PG82_TRACE = ROOT / "research" / "pg82_canonical_triplet_collector_trace_v1.json"
PG74_TRACE = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg85_multisurface_composite_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg85_multisurface_composite_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg85_multisurface_composite_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg85_multisurface_composite_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg85_multisurface_composite_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg85-multisurface-composite-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
PROTOCOL_ID = "pg-pk-85-multisurface-composite-v1"
ROW_TOKENIZER_OVERRIDE: Any = None
PROJECTION_ADAPTER_OVERRIDE: Any = None
PG74_TRAIN_SEEDS = {74101, 74102}
PG74_HOLDOUT_SEEDS = {74103}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pg74_rows(adapter: Any, tokenizer: Any, pg77: Any, steps: list[dict[str, Any]], *, split: str, sha256_json: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = PG74_TRAIN_SEEDS if split == "train" else PG74_HOLDOUT_SEEDS if split == "dev" else set()
    for step in steps:
        if int(step.get("sampling_seed", -1)) not in allowed:
            continue
        adapt = PROJECTION_ADAPTER_OVERRIDE or adapter._adapt
        tokenize = ROW_TOKENIZER_OVERRIDE or tokenizer._row_tokens_v2
        neutral = adapt(dict(step["neutral_projection"]), sha256_json)
        negative = adapt(dict(step["negative_probe_projection"]), sha256_json)
        positive = adapt(dict(step["response_projection"]), sha256_json)
        fake_step = {"action_manifest": step["action_manifest"], "neutral_projection": neutral, "negative_probe_projection": negative}
        for role, projection, oracle in (("positive", positive, step["oracle_projection"]), ("negative", negative, step["negative_oracle_projection"])):
            tokens, oracle_index = tokenize(pg77, fake_step, projection, dict(oracle))
            rows.append({"trace_id": f"{step['step_id']}-{role}", "split": split, "source_id": "pg74", "implementation": "pg74", "variant": "legacy_dom", "family": "held_out_family_metadata", "surface": "legacy_surface", "method": str(step["action_manifest"]["method"]).upper(), "sampling_seed": int(step["sampling_seed"]), "role": role, "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm" if bool(oracle.get("positive")) else "reject", "raw_probe_stored": False, "raw_response_stored": False})
    return rows


def _rewrite(report: dict[str, Any]) -> dict[str, Any]:
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg85-multisurface-composite-transformer-report-v1"
    report["source"]["train_trace"] = str(PG82_TRACE.relative_to(ROOT))
    report["source"]["composite_dataset"] = str(PG74_TRACE.relative_to(ROOT))
    report["source"]["projection_schema"] = "canonical_effect_projection_v2_with_pg84_generic_adapter"
    dev_details = report["details"]["dev_holdout"]
    cross_details = [item for item in dev_details if str(item.get("trace_id", "")).startswith("pg74-step-")]
    positives = [item for item in cross_details if item.get("expected") == "confirm"]
    cross_metrics = {"count": len(cross_details), "typed_positive_count": len(positives), "false_accept_count": sum(int(item.get("expected") == "reject" and item.get("decision") == "confirm") for item in cross_details), "confirm_recall": round(sum(int(item.get("decision") == "confirm") for item in positives) / max(len(positives), 1), 6), "abstain_count": sum(int(item.get("decision") == "abstain") for item in cross_details)}
    report["metrics"]["cross_dataset_holdout"] = cross_metrics
    checks = report["capability_gate"]["checks"]
    checks["cross_dataset_holdout_confirm_recall"] = cross_metrics["confirm_recall"] >= 0.80
    checks["cross_dataset_holdout_false_accept_zero"] = cross_metrics["false_accept_count"] == 0
    report["capability_gate"]["blocking_reasons"] = [key for key, value in checks.items() if not value]
    report["capability_gate"]["status"] = "passed" if not report["capability_gate"]["blocking_reasons"] else "blocked"
    report["promotion"]["reason"] = "PG85 requires composite cross-dataset holdout, source holdout and unknown-family abstention before promotion"
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["schema_version"] = "pg85-multisurface-composite-trace-dataset-v1"
    dataset["dataset_id"] = "pg85-json-html-dom-composite"
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg85-multisurface-composite-transformer-trace-v1"
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["schema_version"] = "pg85-multisurface-composite-transformer-protocol-v1"
    protocol["composite_split"] = {"pg82_train_sources": ["pg34/base", "pg35/alpha"], "pg74_train_seeds": sorted(PG74_TRAIN_SEEDS), "pg74_holdout_seeds": sorted(PG74_HOLDOUT_SEEDS), "pg82_source_holdout": ["pg36/north", "pg36/south"], "unknown_family": "pg76", "family_source_in_tokens": False}
    protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-85 Multisurface composite training\n\n" + f"train/dev/source-holdout/unknown={report['dataset']['train']}/{report['dataset']['dev']}/{report['dataset']['source_holdout']}/{report['dataset']['unknown_family_holdout']}；cross-dataset holdout={cross_metrics['confirm_recall']}；device={report['source']['device']}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "pg85-multisurface-composite-transformer-checkpoint-v1"
    checkpoint["projection_schema"] = "canonical_effect_projection_v2_with_pg84_generic_adapter"
    torch.save(checkpoint, CHECKPOINT_PATH)
    return report


def run() -> dict[str, Any]:
    geometry = _load(PG84_SCRIPT, "pg85_pg84_geometry_runtime")
    pg82 = _load(PG82_SCRIPT, "pg85_pg82_runtime")
    pg74 = json.loads(PG74_TRACE.read_text(encoding="utf-8"))
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    original_hook = pg82.STEP_ROWS_HOOK

    def hook(original: Any) -> Any:
        def composite(pg77: Any, trace: dict[str, Any], pg53: Any, *, split: str, allowed_sources: set[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
            rows = original(pg77, trace, pg53, split=split, allowed_sources=allowed_sources)
            if split in {"train", "dev"}:
                rows.extend(_pg74_rows(geometry, pg82, pg77, pg74["steps"], split=split, sha256_json=sha256_json))
            return rows

        return composite

    pg82.PG82_TRACE = PG82_TRACE
    pg82.PG76_TRACE = PG76_TRACE
    pg82.DATASET_PATH = DATASET_PATH
    pg82.REPORT_PATH = REPORT_PATH
    pg82.PROTOCOL_PATH = PROTOCOL_PATH
    pg82.TRACE_PATH = TRACE_PATH
    pg82.MARKDOWN_PATH = MARKDOWN_PATH
    pg82.OUTPUT_DIR = OUTPUT_DIR
    pg82.CHECKPOINT_PATH = CHECKPOINT_PATH
    pg82.PROTOCOL_ID = PROTOCOL_ID
    pg82.STEP_ROWS_HOOK = hook
    original_tokenizer = pg82._row_tokens_v2
    if ROW_TOKENIZER_OVERRIDE is not None:
        pg82._row_tokens_v2 = ROW_TOKENIZER_OVERRIDE
    try:
        report = pg82.run()
    finally:
        pg82.STEP_ROWS_HOOK = original_hook
        pg82._row_tokens_v2 = original_tokenizer
    return _rewrite(report)


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "train_count": result["dataset"]["train"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "cross_dataset_holdout_recall": result["metrics"]["cross_dataset_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
