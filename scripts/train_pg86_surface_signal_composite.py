"""PG-86: richer observable surface-shape tokens for the PG-85 composite.

PG-85 added HTML/DOM samples but left several response distinctions only in a
coarse recursive count.  PG-86 adds six bounded, non-oracle surface-shape
signals (HTML/DOM size, event count, row count, location presence and a
reflection flag) and their matched deltas.  They are never family labels and
remain behind the same typed-oracle boundary.
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

from app.trace_aligned_dataset import sha256_json  # noqa: E402

PG85_SCRIPT = ROOT / "scripts" / "train_pg85_multisurface_composite.py"
PG84_SCRIPT = ROOT / "scripts" / "run_pg84_cross_dataset_frozen_replay.py"
PG82_SCRIPT = ROOT / "scripts" / "train_pg82_effect_geometry_source_holdout.py"
PG82_TRACE = ROOT / "research" / "pg82_canonical_triplet_collector_trace_v1.json"
PG74_TRACE = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
PG76_TRACE = ROOT / "research" / "pg76_independent_unknown_triplet_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg86_surface_signal_composite_trace_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg86_surface_signal_composite_transformer_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg86_surface_signal_composite_transformer_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg86_surface_signal_composite_transformer_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg86_surface_signal_composite_transformer_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg86-surface-signal-composite-transformer"
CHECKPOINT_PATH = OUTPUT_DIR / "model.pt"
PROTOCOL_ID = "pg-pk-86-surface-signal-composite-v1"

RICH_FIELDS = ("html_tag_count", "dom_node_count", "dom_event_count", "result_row_count", "location_present", "reflection_present")
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
    if before is None or after is None:
        return "u"
    return module._signed_delta(before, after)


def _compact_shape(module: Any, prefix: str, projection: dict[str, Any]) -> list[str]:
    shape = projection.get("shape") or projection.get("json_shape") or {}
    return [
        f"{prefix}_STATUS_{module._status_class(projection)}",
        f"{prefix}_CONTENT_{module._content_class(projection)}",
        f"{prefix}_LENGTH_{str(projection.get('body_length_bucket', 'unknown')).upper().replace('-', '_').replace('+', 'P')}",
        f"{prefix}_JSON_KIND_{str(shape.get('kind', 'none')).upper()}",
    ]


def _effect(module: Any, prefix: str, projection: dict[str, Any]) -> list[str]:
    surface = projection.get("effect_surface")
    geometry = projection.get("effect_geometry")
    if not isinstance(surface, dict) or not isinstance(geometry, dict):
        return [f"{prefix}_EFFECT_MISSING"]
    tokens: list[str] = []
    tokens.extend(f"{prefix}_SURFACE_{name.upper()}_{_bucket(module, surface.get(name))}" for name in SURFACE_FIELDS)
    tokens.extend(f"{prefix}_GEOMETRY_{name.upper()}_{_bucket(module, geometry.get(name))}" for name in GEOMETRY_FIELDS)
    return tokens


def _effect_delta(module: Any, prefix: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    bs, a_s = before.get("effect_surface"), after.get("effect_surface")
    bg, a_g = before.get("effect_geometry"), after.get("effect_geometry")
    if not isinstance(bs, dict) or not isinstance(a_s, dict) or not isinstance(bg, dict) or not isinstance(a_g, dict):
        return [f"{prefix}_EFFECT_MISSING"]
    tokens: list[str] = []
    tokens.extend(f"{prefix}_SURFACE_{name.upper()}_{_delta(module, bs.get(name), a_s.get(name))}" for name in SURFACE_FIELDS)
    tokens.extend(f"{prefix}_GEOMETRY_{name.upper()}_{_delta(module, bg.get(name), a_g.get(name))}" for name in GEOMETRY_FIELDS)
    return tokens


def _rich(module: Any, prefix: str, projection: dict[str, Any]) -> list[str]:
    values = projection.get("effect_surface_rich")
    if not isinstance(values, dict):
        values = {name: None for name in RICH_FIELDS}
    return [f"{prefix}_RICH_{name.upper()}_{_bucket(module, values.get(name))}" for name in RICH_FIELDS]


def _rich_delta(module: Any, prefix: str, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    bs, a_s = before.get("effect_surface_rich"), after.get("effect_surface_rich")
    if not isinstance(bs, dict):
        bs = {name: None for name in RICH_FIELDS}
    if not isinstance(a_s, dict):
        a_s = {name: None for name in RICH_FIELDS}
    return [f"{prefix}_RICH_{name.upper()}_{_delta(module, bs.get(name), a_s.get(name))}" for name in RICH_FIELDS]


def _row_tokens(module: Any, step: dict[str, Any], candidate: dict[str, Any], oracle: dict[str, Any]) -> tuple[list[str], int]:
    method = str((step.get("action_manifest") or {}).get("method", "GET")).upper()
    neutral = dict(step.get("neutral_projection") or step.get("baseline_projection") or {})
    negative = dict(step.get("negative_probe_projection") or neutral)
    tokens = ["BOS", f"CHANNEL_{method}", "CONTROL_TARGET", *_compact_shape(module, "CONTROL", neutral), *_effect(module, "CONTROL", neutral), *_rich(module, "CONTROL", neutral), "SCREEN_TARGET", *_compact_shape(module, "SCREEN", negative), *_effect(module, "SCREEN", negative), *_rich(module, "SCREEN", negative), *_effect_delta(module, "SCREEN_DIFF", neutral, negative), *_rich_delta(module, "SCREEN_DIFF", neutral, negative), "CANDIDATE_TARGET", *_compact_shape(module, "CANDIDATE", candidate), *_effect(module, "CANDIDATE", candidate), *_rich(module, "CANDIDATE", candidate), *_effect_delta(module, "CANDIDATE_DIFF", neutral, candidate), *_rich_delta(module, "CANDIDATE_DIFF", neutral, candidate), f"BELIEF_ACTION_{method}", "BELIEF_IG_0", "BELIEF_DUP_0", "NEXT_ACTION_TARGET", f"ACTION_{method}_CANDIDATE_CANDIDATE", "ORACLE_TARGET"]
    oracle_index = len(tokens) - 1
    modality = module._oracle_modality(oracle)
    outcome = bool(oracle.get("positive"))
    tokens.extend([f"ORACLE_MODALITY_{modality}", f"ORACLE_OUTCOME_{'POSITIVE' if outcome else 'NEGATIVE'}", "RULE_IR_TARGET", f"RULE_EFFECT_{'CONFIRMED' if outcome else 'REJECTED'}", f"RULE_TRANSPORT_{method}", f"RULE_ORACLE_{modality}", "EOS"])
    if len(tokens) > 128:
        raise RuntimeError(f"PG-86 sequence exceeds max_len: {len(tokens)}")
    return tokens, oracle_index


def _adapter(projection: dict[str, Any], sha256: Any) -> dict[str, Any]:
    adapter = _load(PG84_SCRIPT, "pg86_pg84_adapter_runtime")
    result = adapter._adapt(dict(projection), sha256)
    dom = projection.get("dom_shape") or {}
    def first(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None
    rich = {"html_tag_count": first(projection.get("html_tag_count")), "dom_node_count": first(dom.get("node_count")), "dom_event_count": first(dom.get("event_handler_attribute_count")), "result_row_count": first(projection.get("result_row_count")), "location_present": first(projection.get("has_location"), None if "location_origin" not in projection else projection.get("location_origin") not in {"", "none", None}), "reflection_present": first(projection.get("marker_reflected"))}
    result["effect_surface_rich"] = rich
    result["projection_schema"] = "canonical_effect_projection_v3_surface_signal"
    result["projection_sha256"] = sha256({key: value for key, value in result.items() if key != "projection_sha256"})
    return result


def _rewrite(report: dict[str, Any]) -> dict[str, Any]:
    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg86-surface-signal-composite-transformer-report-v1"
    report["source"]["projection_schema"] = "canonical_effect_projection_v3_surface_signal"
    dev_details = report["details"]["dev_holdout"]
    cross = [item for item in dev_details if str(item.get("trace_id", "")).startswith("pg74-step-")]
    positives = [item for item in cross if item.get("expected") == "confirm"]
    cross_metrics = {"count": len(cross), "typed_positive_count": len(positives), "false_accept_count": sum(int(item.get("expected") == "reject" and item.get("decision") == "confirm") for item in cross), "confirm_recall": round(sum(int(item.get("decision") == "confirm") for item in positives) / max(len(positives), 1), 6), "abstain_count": sum(int(item.get("decision") == "abstain") for item in cross)}
    report["metrics"]["cross_dataset_holdout"] = cross_metrics
    checks = report["capability_gate"]["checks"]
    checks["cross_dataset_holdout_confirm_recall"] = cross_metrics["confirm_recall"] >= 0.80
    checks["cross_dataset_holdout_false_accept_zero"] = cross_metrics["false_accept_count"] == 0
    report["capability_gate"]["blocking_reasons"] = [key for key, value in checks.items() if not value]
    report["capability_gate"]["status"] = "passed" if not report["capability_gate"]["blocking_reasons"] else "blocked"
    report["promotion"]["reason"] = "PG86 requires surface-signal composite cross-dataset holdout before promotion"
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8")); dataset["schema_version"] = "pg86-surface-signal-composite-trace-dataset-v1"; DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8")); trace["schema_version"] = "pg86-surface-signal-composite-transformer-trace-v1"; TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")); protocol["protocol_id"] = PROTOCOL_ID; protocol["schema_version"] = "pg86-surface-signal-composite-transformer-protocol-v1"; protocol["projection_schema"] = "canonical_effect_projection_v3_surface_signal"; protocol["run_result"] = {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}; PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-86 Surface signal composite training\n\n" + f"cross-dataset holdout recall={cross_metrics['confirm_recall']}；source holdout recall={report['metrics']['source_holdout']['confirm_recall']}；unknown strict abstain={report['metrics']['unknown_family_holdout']['strict_abstain']}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False); checkpoint["schema_version"] = "pg86-surface-signal-composite-transformer-checkpoint-v1"; checkpoint["projection_schema"] = "canonical_effect_projection_v3_surface_signal"; torch.save(checkpoint, CHECKPOINT_PATH)
    return report


def run() -> dict[str, Any]:
    pg85 = _load(PG85_SCRIPT, "pg86_pg85_runtime")
    pg85.PG82_TRACE = PG82_TRACE; pg85.PG74_TRACE = PG74_TRACE; pg85.PG76_TRACE = PG76_TRACE; pg85.DATASET_PATH = DATASET_PATH; pg85.REPORT_PATH = REPORT_PATH; pg85.PROTOCOL_PATH = PROTOCOL_PATH; pg85.TRACE_PATH = TRACE_PATH; pg85.MARKDOWN_PATH = MARKDOWN_PATH; pg85.OUTPUT_DIR = OUTPUT_DIR; pg85.CHECKPOINT_PATH = CHECKPOINT_PATH; pg85.PROTOCOL_ID = PROTOCOL_ID; pg85.ROW_TOKENIZER_OVERRIDE = _row_tokens; pg85.PROJECTION_ADAPTER_OVERRIDE = _adapter
    return _rewrite(pg85.run())


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "dev_confirm_recall": result["metrics"]["dev_holdout"]["confirm_recall"], "cross_dataset_holdout_recall": result["metrics"]["cross_dataset_holdout"]["confirm_recall"], "source_holdout_confirm_recall": result["metrics"]["source_holdout"]["confirm_recall"], "unknown_strict_abstain": result["metrics"]["unknown_family_holdout"]["strict_abstain"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
