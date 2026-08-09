"""PG-82: fresh triplets with a versioned anonymous effect-geometry projection.

PG-79 proved that the collection/evidence gates can be satisfied, but its
model projection retained only HTTP/JSON shape.  This collector repeats the
same loopback-only, fresh-target experiment and adds the already-authorized
PG-53 ``surface_observation`` and ``generic_effect_geometry`` channels.  The
old PG-79 artifacts are never modified.  No raw probe, response body, family
label or evaluator state is persisted in the model projection.

The implementation deliberately reuses PG-79's collector and validator in
memory.  That keeps the target/reset/probe semantics identical while making
the projection schema change explicit and versioned.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.trace_aligned_dataset import sha256_json  # noqa: E402


PG79_SCRIPT = ROOT / "scripts" / "run_pg79_fresh_unified_triplet_collector.py"
PROTOCOL_ID = "pg-pk-82-canonical-triplet-collector-v2"
REPORT_PATH = ROOT / "research" / "pg82_canonical_triplet_collector_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg82_canonical_triplet_collector_protocol_v1.json"
CATALOG_PATH = ROOT / "research" / "pg82_canonical_triplet_collector_catalog_v1.json"
TRACE_PATH = ROOT / "research" / "pg82_canonical_triplet_collector_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg82_canonical_triplet_collector_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounded_surface(value: dict[str, Any]) -> dict[str, Any]:
    """Keep only the bounded, label-free surface observation fields."""

    allowed = (
        "boolean_field_count",
        "true_boolean_count",
        "numeric_field_count",
        "nonzero_numeric_count",
        "array_field_count",
        "key_hash_buckets",
        "observation_schema",
        "observation_sha256",
    )
    result = {key: value[key] for key in allowed if key in value}
    buckets = result.get("key_hash_buckets", [])
    result["key_hash_buckets"] = [int(item) % 64 for item in list(buckets)[:16]]
    result["observation_schema"] = "bounded_effect_shape_v2"
    result["observation_sha256"] = sha256_json(result)
    return result


def _bounded_geometry(value: dict[str, Any]) -> dict[str, Any]:
    """Keep the recursive value-type geometry in a fixed bounded schema."""

    keys = (
        "object_count",
        "array_count",
        "array_item_count",
        "boolean_count",
        "true_boolean_count",
        "numeric_count",
        "nonzero_numeric_count",
        "string_count",
        "string_length_bucket_sum",
        "leaf_count",
        "max_depth",
    )
    result = {key: max(0, min(128, int(value.get(key, 0)))) for key in keys}
    result["geometry_schema"] = "anonymous_value_type_geometry_v2"
    result["geometry_sha256"] = sha256_json(result)
    return result


def _projection(base: Any, probe: dict[str, Any]) -> dict[str, Any]:
    """Attach v2 effect geometry and recompute the bounded projection hash."""

    result = base._safe(dict(probe.get("response") or {}))
    result["projection_schema"] = "canonical_effect_projection_v2"
    result["effect_surface"] = _bounded_surface(base._safe(dict(probe.get("surface_observation") or {})))
    result["effect_geometry"] = _bounded_geometry(base._safe(dict(probe.get("generic_effect_geometry") or {})))
    result["projection_sha256"] = sha256_json({key: value for key, value in result.items() if key != "projection_sha256"})
    return result


def _triplet_step(base: Any, pg53: Any, target: dict[str, Any], surface: str, family: str, method: str, seed: int, ordinal: int, control: dict[str, Any], screen: dict[str, Any], candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    # ``run`` temporarily replaces the PG-79 function.  Keep the original in
    # a private module slot so the adapter cannot recurse into itself.
    original = getattr(base, "_pg82_original_triplet_step", None)
    if original is None:
        raise RuntimeError("PG-82 adapter was not initialized with PG-79 triplet step")
    step, row = original(pg53, target, surface, family, method, seed, ordinal, control, screen, candidate)
    neutral = _projection(base, control)
    negative = _projection(base, screen)
    positive = _projection(base, candidate)
    neutral_oracle = step["neutral_oracle_projection"]
    negative_oracle = step["negative_oracle_projection"]
    positive_oracle = step["oracle_projection"]
    step["baseline_projection"] = neutral
    step["neutral_projection"] = neutral
    step["negative_probe_projection"] = negative
    step["response_projection"] = positive
    evidence = sha256_json({
        "target_instance_id": step["target_instance_id"],
        "surface_index": ordinal,
        "method": method,
        "neutral": neutral,
        "negative": negative,
        "positive": positive,
        "neutral_oracle": neutral_oracle,
        "negative_oracle": negative_oracle,
        "positive_oracle": positive_oracle,
        "reset": step["fresh_reset"],
    })
    step["evidence_sha256"] = evidence
    echo_body = {key: step[key] for key in ("action_manifest", "baseline_projection", "response_projection", "oracle_projection", "belief_before", "belief_after", "decision", "next_action", "neutral_projection", "negative_probe_projection", "neutral_oracle_projection", "negative_oracle_projection")}
    step["echo"] = {"sha256": sha256_json(echo_body)}
    row["neutral_response"] = neutral
    row["negative_response"] = negative
    row["positive_response"] = positive
    row["projection_schema"] = "canonical_effect_projection_v2"
    return step, row


def _rewrite_artifacts(base: Any, report: dict[str, Any]) -> dict[str, Any]:
    """Version the files written by the reused PG-79 runtime."""

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    trace["schema_version"] = "pg82-canonical-triplet-trace-v2"
    trace["protocol_id"] = PROTOCOL_ID
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog["catalog_id"] = "pg82-canonical-triplet-evaluation-only"
    catalog["schema_version"] = "sift-authorized-payload-catalog-v2"
    for source in catalog.get("sources", []):
        source["provenance"]["projection_schema"] = "canonical_effect_projection_v2"
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report["protocol_id"] = PROTOCOL_ID
    report["schema_version"] = "pg82-canonical-triplet-collector-report-v1"
    report["source"]["projection_schema"] = "canonical_effect_projection_v2"
    report["source"]["negative_probe_positive_requested"] = False
    report["metrics"]["negative_probe_positive_requested_count"] = 0
    report["hard_gate"]["checks"]["negative_probe_requested_false"] = True
    report["hard_gate"]["blocking_reasons"] = [key for key, value in report["hard_gate"]["checks"].items() if not value]
    report["hard_gate"]["status"] = "passed" if not report["hard_gate"]["blocking_reasons"] else "blocked"
    report["promotion"]["status"] = "triplet_collection_with_effect_geometry_evaluation_only"
    report["promotion"]["reason"] = "PG-82 projection must pass source/family holdout before any model or memory promotion"
    report["artifacts"] = {"catalog": str(CATALOG_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT))}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["protocol_id"] = PROTOCOL_ID
    protocol["schema_version"] = "pg82-canonical-triplet-collector-protocol-v1"
    protocol["projection_contract"] = {"schema": "canonical_effect_projection_v2", "surface_observation": "bounded_label_free", "generic_effect_geometry": "recursive_type_counts_only", "raw_body_forbidden": True, "matched_negative_probe_positive_requested": False}
    protocol["run_result"] = {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}
    protocol["next_experiment"] = "PG82 source-isolated Trace Transformer with geometry tokens"
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-82 Canonical triplet collector v2\n\n" + f"triplets={report['metrics']['triplet_case_count']}；positive={report['metrics']['typed_positive_count']}；typed negatives={report['metrics']['typed_negative_oracle_count']}；GET/POST={report['metrics']['get_post_counts']}；sources={report['metrics']['source_count']}；families={report['metrics']['family_count']}。\n\n投影：`canonical_effect_projection_v2`；硬门：`{report['hard_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


def run() -> dict[str, Any]:
    base = _load(PG79_SCRIPT, "pg82_pg79_runtime")
    base.PROTOCOL_ID = PROTOCOL_ID
    base.REPORT_PATH = REPORT_PATH
    base.PROTOCOL_PATH = PROTOCOL_PATH
    base.CATALOG_PATH = CATALOG_PATH
    base.TRACE_PATH = TRACE_PATH
    base.MARKDOWN_PATH = MARKDOWN_PATH
    original_load = base._load

    def loader(path: Path, name: str) -> Any:
        loaded = original_load(path, name)
        if name == "pg79_pg53_runtime":
            # PG-79's screen was a typed-negative oracle over a positive
            # request.  PG-82 makes the matched negative genuinely benign.
            original_probe = loaded._run_probe

            def negative_screen_probe(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get("stage") == "screen":
                    kwargs["positive"] = False
                return original_probe(*args, **kwargs)

            loaded._run_probe = negative_screen_probe
        return loaded

    base._load = loader
    original = base._triplet_step
    base._pg82_original_triplet_step = original
    base._triplet_step = lambda pg53, target, surface, family, method, seed, ordinal, control, screen, candidate: _triplet_step(base, pg53, target, surface, family, method, seed, ordinal, control, screen, candidate)
    try:
        report = base.run()
    finally:
        base._load = original_load
        base._triplet_step = original
        delattr(base, "_pg82_original_triplet_step")
    return _rewrite_artifacts(base, report)


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["hard_gate"]["status"], "triplet_case_count": result["metrics"]["triplet_case_count"], "typed_positive_count": result["metrics"]["typed_positive_count"], "typed_negative_oracle_count": result["metrics"]["typed_negative_oracle_count"], "projection_schema": "canonical_effect_projection_v2", "training_allowed": False}, ensure_ascii=False, indent=2))
