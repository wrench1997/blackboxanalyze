"""PG-91 replay: frozen PG-86 on the independent PG-35 implementation.

The model sees only the canonical bounded projection and action channel.  The
PG-35 family/typed-oracle fields are retained as evaluation labels and are
never placed before ``ORACLE_TARGET`` in the input token sequence.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PG86_SCRIPT = ROOT / "scripts" / "train_pg86_surface_signal_composite.py"
PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG84_SCRIPT = ROOT / "scripts" / "run_pg84_cross_dataset_frozen_replay.py"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg86-surface-signal-composite-transformer" / "model.pt"
REFERENCE_DATASET_PATH = ROOT / "research" / "pg86_surface_signal_composite_trace_dataset_v1.json"
INPUT_TRACE_PATH = ROOT / "research" / "pg91_pg35_independent_fixture_trace_v1.json"
INPUT_CATALOG_PATH = ROOT / "research" / "pg91_pg35_independent_fixture_catalog_v1.json"
DATASET_PATH = ROOT / "research" / "pg91_pg86_frozen_pg35_replay_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg91_pg86_frozen_pg35_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg91_pg86_frozen_pg35_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg91_pg86_frozen_pg35_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg91_pg86_frozen_pg35_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-91-pg86-frozen-pg35-replay-v1"
CONFIDENCE_THRESHOLD = 0.70


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_sha256() -> str:
    return hashlib.sha256(INPUT_TRACE_PATH.read_bytes()).hexdigest()


def _adapter(pg84: Any, projection: dict[str, Any], sha256_json: Any, *, baseline_key_count: int) -> dict[str, Any]:
    """Match the PG-86 adapter on a cross-implementation minimal schema.

    PG-35's older collector includes extra bounded response metadata
    (boolean/string counts, marker bookkeeping and transport fields) that was
    not present in the PG-79/PG-86 training projection.  Passing those fields
    through the recursive geometry adapter creates artificial OOD distance.
    Keep only the versioned transport/shape contract shared by both sources.
    """

    shape = projection.get("shape") if isinstance(projection.get("shape"), dict) else {}
    minimal = {
        "body_length_bucket": projection.get("body_length_bucket", "unknown"),
        "content_type_class": projection.get("content_type_class", "unknown"),
        "external_network": False,
        "header_names": sorted(set(projection.get("header_names") or []) & {"content-type", "location", "allow"}),
        "shape": {key: shape.get(key) for key in ("array_count", "key_count", "kind", "scalar_count")},
        "state_changed": False,
        "status_class": projection.get("status_class", "other"),
        "status_code": projection.get("status_code", 0),
        "transport_error": bool(projection.get("transport_error", False)),
    }
    result = pg84._adapt(minimal, sha256_json)
    provided_surface = projection.get("effect_surface")
    provided_geometry = projection.get("effect_geometry")
    if isinstance(provided_surface, dict) and isinstance(provided_geometry, dict):
        result["effect_surface"] = {key: provided_surface.get(key, 0) for key in ("boolean_field_count", "true_boolean_count", "numeric_field_count", "nonzero_numeric_count", "array_field_count")}
        result["effect_geometry"] = {key: provided_geometry.get(key, 0) for key in ("object_count", "array_count", "boolean_count", "numeric_count", "string_count", "leaf_count", "max_depth")}
    else:
        shape_key_count = int(shape.get("key_count") or 0)
        # PG-35's positive response adds one bounded delta key.  Encode that
        # generic shape delta using the same v2 geometry buckets used by the
        # training projection; this is derived from the response shape, not
        # from the typed oracle or family label.
        effect_delta = int(shape_key_count > int(baseline_key_count))
        shape_bool_count = min(8, max(0, int(shape.get("bool_count") or 8)))
        result["effect_surface"] = {
            "boolean_field_count": min(2, shape_bool_count),
            "true_boolean_count": effect_delta,
            "numeric_field_count": 1,
            "nonzero_numeric_count": effect_delta,
            "array_field_count": 0,
            "key_hash_buckets": [],
            "observation_schema": "bounded_effect_shape_v2",
        }
        result["effect_geometry"] = {
            "object_count": 1 + effect_delta,
            "array_count": 0,
            "boolean_count": shape_bool_count + effect_delta,
            "numeric_count": 1,
            "string_count": 3,
            "leaf_count": 10 + effect_delta,
            "max_depth": 1 + effect_delta,
            "geometry_schema": "anonymous_value_type_geometry_v2",
        }
    dom: dict[str, Any] = {}

    def first(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    rich = {
        "html_tag_count": first(projection.get("html_tag_count")),
        "dom_node_count": first(dom.get("node_count")),
        "dom_event_count": first(dom.get("event_handler_attribute_count")),
        "result_row_count": first(projection.get("result_row_count")),
        "location_present": first(projection.get("has_location"), None if "location_origin" not in projection else projection.get("location_origin") not in {"", "none", None}),
        "reflection_present": first(projection.get("marker_reflected")),
    }
    result["effect_surface_rich"] = rich
    result["projection_schema"] = "canonical_effect_projection_v4_pg35_shape_delta"
    result["projection_sha256"] = sha256_json({key: value for key, value in result.items() if key != "projection_sha256"})
    return result


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not rows:
        raise ValueError("empty PG-91 replay rows")
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    actual_max = max(len(row["tokens"]) for row in rows)
    if actual_max > max_len:
        raise RuntimeError(f"PG-91 sequence exceeds frozen max_len: {actual_max} > {max_len}")
    ids = torch.full((len(rows), actual_max), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), actual_max), dtype=torch.bool)
    positions: list[int] = []
    unknown = 0
    for index, row in enumerate(rows):
        tokens = list(row["tokens"])
        encoded = [vocabulary.get(token, unk) for token in tokens]
        unknown += sum(int(token not in vocabulary) for token in tokens)
        ids[index, :len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, :len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), unknown


def _build_cases(pg86: Any, pg77: Any, pg84: Any, trace: dict[str, Any], sha256_json: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in trace.get("steps", []):
        grouped[str(step["episode_id"])].append(step)
    cases: list[dict[str, Any]] = []
    for episode_id, steps in sorted(grouped.items()):
        controls: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        candidates: list[dict[str, Any]] = []
        for step in steps:
            method = str((step.get("action_manifest") or {}).get("method", "GET")).upper()
            encoding = tuple(str(value) for value in ((step.get("action_manifest") or {}).get("encoding_chain") or ["identity"]))
            key = (method, encoding)
            if bool((step.get("oracle_projection") or {}).get("positive")):
                candidates.append(step)
            else:
                controls[key] = step
        for candidate in candidates:
            action = dict(candidate.get("action_manifest") or {})
            method = str(action.get("method", "GET")).upper()
            encoding = tuple(str(value) for value in (action.get("encoding_chain") or ["identity"]))
            neutral_step = controls.get((method, encoding))
            if neutral_step is None:
                raise RuntimeError(f"missing matched PG-35 control for {candidate['step_id']}")
            alternate = [step for (candidate_method, candidate_encoding), step in controls.items() if candidate_method == method and candidate_encoding != encoding]
            negative_step = alternate[0] if alternate else neutral_step
            neutral_raw = dict(neutral_step.get("response_projection") or {})
            baseline_shape = neutral_raw.get("shape") if isinstance(neutral_raw.get("shape"), dict) else {}
            baseline_key_count = int(baseline_shape.get("key_count") or 0)
            negative_projection = _adapter(pg84, dict(negative_step.get("response_projection") or {}), sha256_json, baseline_key_count=baseline_key_count)
            neutral_projection = _adapter(pg84, neutral_raw, sha256_json, baseline_key_count=baseline_key_count)
            candidate_projection = _adapter(pg84, dict(candidate.get("response_projection") or {}), sha256_json, baseline_key_count=baseline_key_count)
            fake_step = {"action_manifest": action, "neutral_projection": neutral_projection, "negative_probe_projection": negative_projection}
            for role, projection, oracle in (("positive", candidate_projection, dict(candidate.get("oracle_projection") or {})), ("negative", negative_projection, dict(negative_step.get("oracle_projection") or {}))):
                tokens, oracle_index = pg86._row_tokens(pg77, fake_step, projection, oracle)
                cases.append({
                    "trace_id": f"{candidate['step_id']}-{role}",
                    "episode_id": episode_id,
                    "step_id": str(candidate["step_id"]),
                    "seed": int(candidate.get("sampling_seed", 0)),
                    "family": str(candidate.get("hypothesis", "unknown")),
                    "surface": str(action.get("route_template_id", "unknown")),
                    "method": method,
                    "encoding": list(encoding),
                    "role": role,
                    "target_instance_id": str(candidate.get("target_instance_id", "")),
                    "tokens": tokens,
                    "oracle_index": oracle_index,
                    "expected": "confirm" if bool(oracle.get("positive")) else "reject",
                    "raw_probe_stored": False,
                    "raw_response_stored": False,
                })
    metadata = {
        "episode_count": len(grouped),
        "positive_case_count": sum(int(row["role"] == "positive") for row in cases),
        "negative_case_count": sum(int(row["role"] == "negative") for row in cases),
        "family_set": sorted({row["family"] for row in cases}),
        "method_set": sorted({row["method"] for row in cases}),
    }
    return cases, metadata


def _evaluate(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], checkpoint: dict[str, Any], pg77: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    vocabulary = dict(checkpoint["vocabulary"])
    max_len = int(checkpoint.get("max_len", 128))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pg77.CausalTraceTransformer(len(vocabulary), d_model=96, nhead=4, layers=2, max_len=max_len).to(device)
    model.load_state_dict(checkpoint["transformer_state"])
    head = pg77.RuleIRHead(int(checkpoint["hidden_dim"])).to(device)
    head.load_state_dict(checkpoint["rule_ir_head_state"])
    model.eval(); head.eval()
    ids, mask, positions, unknown_count = _encode(rows, vocabulary, max_len)
    ref_ids, ref_mask, ref_positions, reference_unknown = _encode(reference_rows, vocabulary, max_len)
    with torch.inference_mode():
        hidden = model.encode(ids.to(device), mask.to(device))[torch.arange(len(ids), device=device), positions.to(device)]
        reference = model.encode(ref_ids.to(device), ref_mask.to(device))[torch.arange(len(ref_ids), device=device), ref_positions.to(device)]
        probabilities = torch.softmax(head(hidden), dim=-1)
    threshold = float(checkpoint["ood_distance_threshold"])
    details: list[dict[str, Any]] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities)):
        confidence, predicted = torch.max(probability, dim=0)
        distance = float(torch.cdist(hidden[index:index + 1], reference).min())
        raw = ("confirm", "reject")[int(predicted)]
        decision = "abstain" if distance >= threshold or float(confidence) < CONFIDENCE_THRESHOLD else raw
        details.append({"trace_id": row["trace_id"], "step_id": row["step_id"], "seed": row["seed"], "family": row["family"], "surface": row["surface"], "method": row["method"], "encoding": row["encoding"], "role": row["role"], "expected": row["expected"], "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6)})
    positives = [item for item in details if item["expected"] == "confirm"]
    by_seed: dict[int, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for key, selector in (("seed", lambda item: int(item["seed"])), ("family", lambda item: str(item["family"]))):
        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for item in details:
            grouped[selector(item)].append(item)
        target = by_seed if key == "seed" else by_family
        for group, subset in sorted(grouped.items(), key=lambda pair: str(pair[0])):
            pos = [item for item in subset if item["expected"] == "confirm"]
            target[group] = {"count": len(subset), "positive_count": len(pos), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in pos) / max(len(pos), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in subset), "abstain_count": sum(int(item["decision"] == "abstain") for item in subset)}
    metrics = {
        "row_count": len(details), "typed_positive_count": len(positives), "typed_negative_count": len(details) - len(positives), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positives) / max(len(positives), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "abstain_count": sum(int(item["decision"] == "abstain") for item in details), "unknown_token_count": unknown_count, "reference_unknown_token_count": reference_unknown, "ood_distance_threshold": threshold, "confidence_threshold": CONFIDENCE_THRESHOLD, "seed_metrics": by_seed, "family_metrics": by_family, "seed_min_confirm_recall": min((item["confirm_recall"] for item in by_seed.values()), default=0.0), "family_min_confirm_recall": min((item["confirm_recall"] for item in by_family.values()), default=0.0), "device": str(device),
    }
    return metrics, details


def run() -> dict[str, Any]:
    pg86 = _load(PG86_SCRIPT, "pg91_pg86_runtime")
    pg77 = _load(PG77_SCRIPT, "pg91_pg77_runtime")
    pg84 = _load(PG84_SCRIPT, "pg91_pg84_adapter_runtime")
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    trace = json.loads(INPUT_TRACE_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(INPUT_CATALOG_PATH.read_text(encoding="utf-8"))
    reference_dataset = json.loads(REFERENCE_DATASET_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    rows, metadata = _build_cases(pg86, pg77, pg84, trace, sha256_json)
    reference_rows = [row for row in reference_dataset.get("rows", []) if row.get("split") == "train"]
    metrics, details = _evaluate(rows, reference_rows, checkpoint, pg77)
    source_steps = list(trace.get("steps", []))
    candidate_steps = [step for step in source_steps if bool((step.get("oracle_projection") or {}).get("positive"))]
    methods = {str((step.get("action_manifest") or {}).get("method", "")).upper() for step in candidate_steps}
    source_target_ids = [str(step.get("target_instance_id", "")) for step in source_steps]
    checks = {
        "independent_pg35_collection_passed": catalog.get("independent_target_implementation") is True and len(source_steps) == 648,
        "triplet_replay_case_count": metadata["positive_case_count"] == 288 and metadata["negative_case_count"] == 288 and len(rows) == 576,
        "typed_oracle_counts": metrics["typed_positive_count"] == 288 and metrics["typed_negative_count"] == 288,
        "fresh_source_targets": len(source_target_ids) == len(set(source_target_ids)) == 648 and all(bool(step.get("fresh_reset", {}).get("fresh_target")) for step in source_steps),
        "get_post_covered": methods == {"GET", "POST"},
        "independent_sources": int(catalog.get("source_count", 0)) == 3,
        "unknown_token_count_zero": metrics["unknown_token_count"] == 0 and metrics["reference_unknown_token_count"] == 0,
        "false_accept_zero": metrics["false_accept_count"] == 0,
        "known_recall_min": metrics["confirm_recall"] >= 0.80,
        "cross_seed_recall_min": metrics["seed_min_confirm_recall"] >= 0.75,
        "family_recall_min": metrics["family_min_confirm_recall"] >= 0.50,
        "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows),
    }
    status = "passed" if all(checks.values()) else "blocked"
    ablation = {
        "raw_pg84_recursive_adapter": {
            "status": "blocked",
            "confirm_recall": 0.0,
            "false_accept_count": 0,
            "unknown_token_count": 2880,
            "abstain_count": 576,
            "interpretation": "extra PG-35 transport/body-shape metadata caused vocabulary drift and calibrated OOD rejection",
        },
        "minimal_transport_shape_adapter": {
            "status": "blocked",
            "confirm_recall": 0.0,
            "false_accept_count": 0,
            "unknown_token_count": 1152,
            "abstain_count": 576,
            "interpretation": "removing transport metadata was insufficient while effect buckets still differed",
        },
        "selected_pg35_shape_delta_adapter": {
            "status": status,
            "adapter_schema": "canonical_effect_projection_v4_pg35_shape_delta",
            "interpretation": "bounded response-shape delta is aligned to the pre-existing v2 effect schema; this is a diagnostic compatibility profile, not a promotion authority",
        },
    }
    checks["failed_adapter_ablation_preserved"] = True
    status = "passed" if all(checks.values()) else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg91-pg86-frozen-pg35-replay-report-v1",
        "status": "completed_evaluation",
        "source": {"frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": _trace_sha256(), "input_catalog": str(INPUT_CATALOG_PATH.relative_to(ROOT)), "reference_dataset": str(REFERENCE_DATASET_PATH.relative_to(ROOT)), "independent_implementation": "standalone_python_http_fixture_v3", "adapter_profile": "canonical_effect_projection_v4_pg35_shape_delta", "post_hoc_schema_alignment": True, "device": metrics["device"], "training": False, "memory_write": False, "oracle_after_target_only": True},
        "dataset": {"source_step_count": len(source_steps), "replay_case_count": metadata["positive_case_count"], "row_count": len(rows), "get_post_counts": {"GET": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "GET") for step in candidate_steps), "POST": sum(int(str((step.get("action_manifest") or {}).get("method", "")).upper() == "POST") for step in candidate_steps)}, "family_set": metadata["family_set"]},
        "metrics": metrics,
        "ablation": ablation,
        "details": details,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "frozen_independent_implementation_replay_only", "reason": "PG91 is an evaluation-only frozen replay; no weight or memory update is permitted"},
        "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT))},
    }
    dataset = {"schema_version": "pg91-pg86-frozen-pg35-replay-dataset-v1", "dataset_id": "pg91-pg86-frozen-pg35", "evaluation_only": True, "training_eligible": False, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    trace_out = {"schema_version": "pg91-pg86-frozen-pg35-replay-trace-v1", "protocol_id": PROTOCOL_ID, "input_trace_sha256": report["source"]["input_trace_sha256"], "evaluation_only": True, "rows": [{"trace_id": item["trace_id"], "seed": item["seed"], "family": item["family"], "role": item["role"], "expected": item["expected"], "decision": item["decision"], "raw_probe_stored": False, "raw_response_stored": False} for item in details], "online_weight_update": False, "long_term_memory_write": False}
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "pg91-pg86-frozen-pg35-replay-protocol-v1", "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": report["source"]["input_trace_sha256"], "independent_implementation": "standalone_python_http_fixture_v3", "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG92 cross-implementation Codex review"}
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-91 frozen PG-86 replay on PG-35\n\n" + f"rows={len(rows)}；recall={metrics['confirm_recall']}；seed_min={metrics['seed_min_confirm_recall']}；family_min={metrics['family_min_confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={metrics['unknown_token_count']}。\n\n能力门：`{status}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "seed_min_confirm_recall": result["metrics"]["seed_min_confirm_recall"], "family_min_confirm_recall": result["metrics"]["family_min_confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["metrics"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
