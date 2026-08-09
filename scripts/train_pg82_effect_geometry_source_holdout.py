"""PG-82: source-isolated Trace Transformer with canonical effect geometry.

This is a versioned PG-81 runner.  It reuses the split, optimizer, OOD rule
and Rule-IR head, but replaces the old response tokenizer with a fixed,
label-free geometry vocabulary.  The evaluator's family/source/variant and
typed oracle remain metadata; the oracle is still visible only after the
``ORACLE_TARGET`` marker.  PG82 and PG76 remain evaluation-only.
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

PG81_SCRIPT = ROOT / "scripts" / "train_pg81_source_holdout_transformer.py"
PG82_TRACE = ROOT / "research" / "pg82_canonical_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg82_effect_geometry_source_holdout_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg82_effect_geometry_source_holdout_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg82_effect_geometry_source_holdout_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg82_effect_geometry_source_holdout_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg82_effect_geometry_source_holdout_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg82-effect-geometry-source-holdout-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
PROTOCOL_ID = "pg-pk-82-effect-geometry-source-holdout-v1"
STEP_ROWS_HOOK: Any = None

SURFACE_FIELDS = ("boolean_field_count", "true_boolean_count", "numeric_field_count", "nonzero_numeric_count", "array_field_count")
GEOMETRY_FIELDS = ("object_count", "array_count", "boolean_count", "numeric_count", "string_count", "leaf_count", "max_depth")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bucket(module: Any, value: Any) -> str:
    return module._bucket(value)


def _delta(module: Any, before: Any, after: Any) -> str:
    return module._signed_delta(before, after)


def _compact_shape_tokens(module: Any, prefix: str, projection: dict[str, Any]) -> list[str]:
    shape = projection.get("shape") or projection.get("json_shape") or {}
    dom = projection.get("dom_shape") or {}
    return [
        f"{prefix}_STATUS_{module._status_class(projection)}",
        f"{prefix}_CONTENT_{module._content_class(projection)}",
        f"{prefix}_LENGTH_{str(projection.get('body_length_bucket', 'unknown')).upper().replace('-', '_').replace('+', 'P')}",
        f"{prefix}_JSON_KIND_{str(shape.get('kind', 'none')).upper()}",
        f"{prefix}_JSON_KEYS_{_bucket(module, shape.get('key_count'))}",
        f"{prefix}_JSON_SCALARS_{_bucket(module, shape.get('scalar_count'))}",
        f"{prefix}_JSON_ARRAYS_{_bucket(module, shape.get('array_count'))}",
        f"{prefix}_DOM_NODES_{_bucket(module, dom.get('node_count'))}",
        f"{prefix}_DOM_EVENTS_{_bucket(module, dom.get('event_handler_attribute_count'))}",
    ]


def _effect_tokens(module: Any, prefix: str, projection: dict[str, Any]) -> list[str]:
    surface = projection.get("effect_surface")
    geometry = projection.get("effect_geometry")
    if not isinstance(surface, dict) or not isinstance(geometry, dict):
        return [f"{prefix}_EFFECT_MISSING"]
    tokens = [f"{prefix}_EFFECT_SURFACE_V2"]
    tokens.extend(f"{prefix}_SURFACE_{name.upper()}_{_bucket(module, surface.get(name))}" for name in SURFACE_FIELDS)
    tokens.append(f"{prefix}_EFFECT_GEOMETRY_V2")
    tokens.extend(f"{prefix}_GEOMETRY_{name.upper()}_{_bucket(module, geometry.get(name))}" for name in GEOMETRY_FIELDS)
    return tokens


def _effect_delta_tokens(module: Any, prefix: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_surface = before.get("effect_surface") if isinstance(before.get("effect_surface"), dict) else {}
    after_surface = after.get("effect_surface") if isinstance(after.get("effect_surface"), dict) else {}
    before_geometry = before.get("effect_geometry") if isinstance(before.get("effect_geometry"), dict) else {}
    after_geometry = after.get("effect_geometry") if isinstance(after.get("effect_geometry"), dict) else {}
    if not before_surface or not after_surface or not before_geometry or not after_geometry:
        return [f"{prefix}_EFFECT_MISSING"]
    tokens = [f"{prefix}_EFFECT_DELTA_V2"]
    tokens.extend(f"{prefix}_SURFACE_{name.upper()}_{_delta(module, before_surface.get(name), after_surface.get(name))}" for name in SURFACE_FIELDS)
    tokens.extend(f"{prefix}_GEOMETRY_{name.upper()}_{_delta(module, before_geometry.get(name), after_geometry.get(name))}" for name in GEOMETRY_FIELDS)
    return tokens


def _row_tokens_v2(module: Any, step: dict[str, Any], candidate: dict[str, Any], oracle: dict[str, Any]) -> tuple[list[str], int]:
    method = str((step.get("action_manifest") or {}).get("method", "GET")).upper()
    neutral = dict(step.get("neutral_projection") or step.get("baseline_projection") or {})
    negative = dict(step.get("negative_probe_projection") or neutral)
    tokens = [
        "BOS",
        f"CHANNEL_{method}",
        "CONTROL_TARGET",
        *_compact_shape_tokens(module, "CONTROL", neutral),
        *_effect_tokens(module, "CONTROL", neutral),
        "SCREEN_TARGET",
        *_compact_shape_tokens(module, "SCREEN", negative),
        *_effect_tokens(module, "SCREEN", negative),
        *_effect_delta_tokens(module, "SCREEN_DIFF", neutral, negative),
        "CANDIDATE_TARGET",
        *_compact_shape_tokens(module, "CANDIDATE", candidate),
        *_effect_tokens(module, "CANDIDATE", candidate),
        *_effect_delta_tokens(module, "CANDIDATE_DIFF", neutral, candidate),
        f"BELIEF_ACTION_{method}",
        "BELIEF_IG_0",
        "BELIEF_DUP_0",
        "NEXT_ACTION_TARGET",
        f"ACTION_{method}_CANDIDATE_CANDIDATE",
        "ORACLE_TARGET",
    ]
    oracle_index = len(tokens) - 1
    modality = module._oracle_modality(oracle)
    outcome = bool(oracle.get("positive"))
    tokens.extend([f"ORACLE_MODALITY_{modality}", f"ORACLE_OUTCOME_{'POSITIVE' if outcome else 'NEGATIVE'}", "RULE_IR_TARGET", f"RULE_EFFECT_{'CONFIRMED' if outcome else 'REJECTED'}", f"RULE_TRANSPORT_{method}", f"RULE_ORACLE_{modality}", "EOS"])
    if len(tokens) > 128:
        raise RuntimeError(f"PG-82 sequence exceeds transformer max_len: {len(tokens)}")
    return tokens, oracle_index


def _rewrite_artifacts(module: Any, report: dict[str, Any]) -> dict[str, Any]:
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg82-effect-geometry-source-holdout-transformer-report-v1"
    report["source"]["train_trace"] = str(PG82_TRACE.relative_to(ROOT))
    report["source"]["projection_schema"] = "canonical_effect_projection_v2"
    report["promotion"]["reason"] = "PG82 source/family holdout must pass with geometry projection before promotion"
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset["schema_version"] = "pg82-effect-geometry-source-holdout-trace-dataset-v1"
    dataset["dataset_id"] = "pg82-effect-geometry-source-holdout-triplets"
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg82-effect-geometry-source-holdout-transformer-trace-v1"
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["schema_version"] = "pg82-effect-geometry-source-holdout-transformer-protocol-v1"
    protocol["input_contract"]["projection_schema"] = "canonical_effect_projection_v2"
    protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-82 Effect geometry source holdout\n\n" + f"train/dev/source-holdout/unknown={report['dataset']['train']}/{report['dataset']['dev']}/{report['dataset']['source_holdout']}/{report['dataset']['unknown_family_holdout']}；device={report['source']['device']}；projection=`canonical_effect_projection_v2`。\n\ndev recall={report['metrics']['dev_holdout']['confirm_recall']}；source holdout recall={report['metrics']['source_holdout']['confirm_recall']}；unknown strict abstain={report['metrics']['unknown_family_holdout']['strict_abstain']}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "pg82-effect-geometry-source-holdout-transformer-checkpoint-v1"
    checkpoint["projection_schema"] = "canonical_effect_projection_v2"
    torch.save(checkpoint, CHECKPOINT_PATH)
    return report


def run() -> dict[str, Any]:
    module = _load(PG81_SCRIPT, "pg82_pg81_runtime")
    module.PG79_TRACE = PG82_TRACE
    module.PG76_TRACE = PG76_TRACE
    module.DATASET_PATH = DATASET_PATH
    module.REPORT_PATH = REPORT_PATH
    module.PROTOCOL_PATH = PROTOCOL_PATH
    module.TRACE_PATH = TRACE_PATH
    module.MARKDOWN_PATH = MARKDOWN_PATH
    module.OUTPUT_DIR = OUTPUT_DIR
    module.CHECKPOINT_PATH = CHECKPOINT_PATH
    original_load = module._load

    def loader(path: Path, name: str) -> Any:
        loaded = original_load(path, name)
        if "pg77" in name:
            loaded._row_tokens = lambda step, candidate, oracle: _row_tokens_v2(loaded, step, candidate, oracle)
        return loaded

    module._load = loader
    original_step_rows = module._step_rows
    if STEP_ROWS_HOOK is not None:
        module._step_rows = STEP_ROWS_HOOK(original_step_rows)
    try:
        report = module.run()
    finally:
        module._load = original_load
        module._step_rows = original_step_rows
    return _rewrite_artifacts(module, report)


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "projection_schema": "canonical_effect_projection_v2", "training_allowed": False}, ensure_ascii=False, indent=2))
