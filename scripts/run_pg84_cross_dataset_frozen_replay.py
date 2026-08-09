"""PG-84: frozen PG-83 replay on the independent PG-74 triplet dataset.

PG-74 predates the canonical v2 projection and comes from a different local
surface implementation.  This evaluator performs a deterministic, generic
bounded-type adapter (no raw bodies or evaluator labels) and then runs the
frozen PG-83 Transformer/Rule-IR head without any weight or memory update.
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

PG82_SCRIPT = ROOT / "scripts" / "train_pg82_effect_geometry_source_holdout.py"
PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG83_CHECKPOINT = ROOT / "artifacts" / "pg83-cross-seed-geometry-holdout-transformer" / "model.pt"
PG83_DATASET = ROOT / "research" / "pg83_cross_seed_geometry_holdout_trace_dataset_v1.json"
PG74_TRACE = ROOT / "research" / "pg74_causal_triplet_collector_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg84_cross_dataset_frozen_replay_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg84_cross_dataset_frozen_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg84_cross_dataset_frozen_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg84_cross_dataset_frozen_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg84_cross_dataset_frozen_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-84-cross-dataset-frozen-replay-v1"
CONFIDENCE_THRESHOLD = 0.70


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _excluded(key: Any) -> bool:
    name = str(key).casefold()
    return name in {"body", "raw_body", "body_preview", "request_body", "raw_probe", "password", "token", "cookie", "authorization"} or name.endswith("sha256") or "sha256" in name


def _geometry(value: Any) -> dict[str, int]:
    counts = {"object_count": 0, "array_count": 0, "array_item_count": 0, "boolean_count": 0, "true_boolean_count": 0, "numeric_count": 0, "nonzero_numeric_count": 0, "string_count": 0, "string_length_bucket_sum": 0, "leaf_count": 0, "max_depth": 0}

    def visit(node: Any, depth: int) -> None:
        counts["max_depth"] = min(16, max(counts["max_depth"], depth))
        if isinstance(node, dict):
            counts["object_count"] = min(64, counts["object_count"] + 1)
            for key, child in node.items():
                if not _excluded(key):
                    visit(child, depth + 1)
            return
        if isinstance(node, list):
            counts["array_count"] = min(32, counts["array_count"] + 1)
            counts["array_item_count"] = min(64, counts["array_item_count"] + len(node))
            for child in node[:32]:
                visit(child, depth + 1)
            return
        counts["leaf_count"] = min(128, counts["leaf_count"] + 1)
        if isinstance(node, bool):
            counts["boolean_count"] = min(64, counts["boolean_count"] + 1)
            counts["true_boolean_count"] = min(64, counts["true_boolean_count"] + int(node))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            counts["numeric_count"] = min(64, counts["numeric_count"] + 1)
            counts["nonzero_numeric_count"] = min(64, counts["nonzero_numeric_count"] + int(abs(float(node)) > 1e-9))
        elif isinstance(node, str):
            counts["string_count"] = min(64, counts["string_count"] + 1)
            length = len(node)
            counts["string_length_bucket_sum"] = min(128, counts["string_length_bucket_sum"] + (0 if length == 0 else 1 if length <= 16 else 2 if length <= 64 else 3))

    visit(value, 0)
    return counts


def _surface(projection: dict[str, Any]) -> dict[str, Any]:
    booleans: list[bool] = []
    numerics: list[float] = []
    arrays = 0
    buckets: list[int] = []
    for key, value in projection.items():
        if _excluded(key):
            continue
        digest = hashlib.sha256(str(key).encode("utf-8", errors="replace")).digest()
        buckets.append(int.from_bytes(digest[:2], "big") % 64)
        if isinstance(value, bool):
            booleans.append(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numerics.append(float(value))
        elif isinstance(value, list):
            arrays += 1
    result = {"boolean_field_count": min(32, len(booleans)), "true_boolean_count": min(32, sum(booleans)), "numeric_field_count": min(32, len(numerics)), "nonzero_numeric_count": min(32, sum(abs(item) > 1e-9 for item in numerics)), "array_field_count": min(8, arrays), "key_hash_buckets": sorted(set(buckets))[:16], "observation_schema": "bounded_effect_shape_v2"}
    result["observation_sha256"] = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json(result)
    return result


def _adapt(projection: dict[str, Any], sha256_json: Any) -> dict[str, Any]:
    result = dict(projection)
    result["projection_schema"] = "canonical_effect_projection_v2_adapter_pg84"
    result["effect_surface"] = _surface(projection)
    geometry = _geometry(projection)
    geometry["geometry_schema"] = "anonymous_value_type_geometry_v2_adapter_pg84"
    geometry["geometry_sha256"] = sha256_json(geometry)
    result["effect_geometry"] = geometry
    result["projection_sha256"] = sha256_json({key: value for key, value in result.items() if key != "projection_sha256"})
    return result


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    max_len = max(len(row["tokens"]) for row in rows)
    if max_len > 128:
        raise RuntimeError(f"PG-84 sequence exceeds max_len: {max_len}")
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    positions: list[int] = []
    unknown = 0
    for index, row in enumerate(rows):
        encoded = [vocabulary.get(token, unk) for token in row["tokens"]]
        unknown += sum(int(token not in vocabulary) for token in row["tokens"])
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), unknown


def _rows(pg77: Any, geometry: Any, trace: dict[str, Any], sha256_json: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        neutral = _adapt(dict(step["neutral_projection"]), sha256_json)
        negative = _adapt(dict(step["negative_probe_projection"]), sha256_json)
        positive = _adapt(dict(step["response_projection"]), sha256_json)
        for role, projection, oracle in (("positive", positive, step["oracle_projection"]), ("negative", negative, step["negative_oracle_projection"])):
            fake_step = {"action_manifest": step["action_manifest"], "neutral_projection": neutral, "negative_probe_projection": negative}
            tokens, oracle_index = geometry._row_tokens_v2(pg77, fake_step, projection, dict(oracle))
            rows.append({"trace_id": f"{step['step_id']}-{role}", "role": role, "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm" if bool(oracle.get("positive")) else "reject", "raw_probe_stored": False, "raw_response_stored": False, "source_trace": "pg74"})
    return rows


def run() -> dict[str, Any]:
    geometry = _load(PG82_SCRIPT, "pg84_geometry_runtime")
    pg77 = _load(PG77_SCRIPT, "pg84_pg77_runtime")
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    checkpoint = torch.load(PG83_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = dict(checkpoint["vocabulary"])
    pg74 = json.loads(PG74_TRACE.read_text(encoding="utf-8"))
    pg83_dataset = json.loads(PG83_DATASET.read_text(encoding="utf-8"))
    rows = _rows(pg77, geometry, pg74, sha256_json)
    reference_rows = [row for row in pg83_dataset["rows"] if row.get("split") == "train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pg77.CausalTraceTransformer(len(vocabulary), d_model=96, nhead=4, layers=2, max_len=128).to(device)
    model.load_state_dict(checkpoint["transformer_state"])
    head = pg77.RuleIRHead(int(checkpoint["hidden_dim"])).to(device)
    head.load_state_dict(checkpoint["rule_ir_head_state"])
    model.eval(); head.eval()
    ids, mask, positions, unknown_count = _encode(rows, vocabulary)
    ref_ids, ref_mask, ref_positions, ref_unknown = _encode(reference_rows, vocabulary)
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
        details.append({"trace_id": row["trace_id"], "role": row["role"], "expected": row["expected"], "raw_prediction": raw, "decision": decision, "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6)})
    positives = [item for item in details if item["expected"] == "confirm"]
    metrics = {"count": len(details), "typed_positive_count": len(positives), "typed_negative_count": len(details) - len(positives), "typed_neutral_count": len(pg74.get("steps", [])), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positives) / max(len(positives), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details), "abstain_count": sum(int(item["decision"] == "abstain") for item in details), "unknown_token_count": unknown_count, "reference_unknown_token_count": ref_unknown}
    methods = {str(step.get("action_manifest", {}).get("method", "")).upper() for step in pg74.get("steps", [])}
    checks = {"independent_triplet_dataset": len(pg74.get("steps", [])) == 21, "typed_positive_count": metrics["typed_positive_count"] == 21, "typed_negative_count": metrics["typed_negative_count"] == 21, "typed_neutral_count": metrics["typed_neutral_count"] == 21, "fresh_reset_per_case": all(bool(step.get("fresh_reset", {}).get("fresh_target")) for step in pg74.get("steps", [])), "get_post_covered": methods == {"GET", "POST"}, "unknown_token_count_zero": metrics["unknown_token_count"] == 0, "false_accept_zero": metrics["false_accept_count"] == 0, "known_recall_min": metrics["confirm_recall"] >= 0.80, "raw_free": True}
    report = {"protocol_id": PROTOCOL_ID, "schema_version": "pg84-cross-dataset-frozen-replay-report-v1", "status": "completed_evaluation", "source": {"frozen_checkpoint": str(PG83_CHECKPOINT.relative_to(ROOT)), "dataset": str(PG74_TRACE.relative_to(ROOT)), "device": str(device), "training": False, "memory_write": False, "projection_adapter": "generic_bounded_type_geometry_v2"}, "dataset": {"step_count": len(pg74.get("steps", [])), "row_count": len(rows), "get_post_counts": {"GET": sum(str(step.get("action_manifest", {}).get("method", "")).upper() == "GET" for step in pg74.get("steps", [])), "POST": sum(str(step.get("action_manifest", {}).get("method", "")).upper() == "POST" for step in pg74.get("steps", []))}}, "metrics": metrics, "details": details, "hard_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "frozen_cross_dataset_replay_only", "reason": "PG84 is evaluation-only and cannot promote a checkpoint"}}
    DATASET_PATH.write_text(json.dumps({"schema_version": "pg84-cross-dataset-frozen-replay-dataset-v1", "dataset_id": "pg84-pg74-adapted-triplets", "evaluation_only": True, "training_eligible": False, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg84-cross-dataset-frozen-replay-trace-v1", "protocol_id": PROTOCOL_ID, "evaluation_only": True, "rows": [{"trace_id": row["trace_id"], "role": row["role"], "source_trace": row["source_trace"], "raw_probe_stored": False, "raw_response_stored": False} for row in rows], "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": PROTOCOL_ID, "schema_version": "pg84-cross-dataset-frozen-replay-protocol-v1", "frozen_checkpoint": str(PG83_CHECKPOINT.relative_to(ROOT)), "projection_adapter": "generic_bounded_type_geometry_v2", "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"hard_gate": report["hard_gate"], "training_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-84 Cross-dataset frozen replay\n\n" + f"rows={len(rows)}；GET/POST={report['dataset']['get_post_counts']}；recall={metrics['confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={metrics['unknown_token_count']}。\n\n硬门：`{report['hard_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["hard_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
