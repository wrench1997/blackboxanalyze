"""PG-78: evaluate PG-77 on an existing fresh multi-source/multi-family holdout.

PG-53 already collected 270 loopback cases across five independently written
implementations, eight effect families, three seeds and both GET/POST.  This
adapter maps only its bounded response-shape projections into the PG-77
abstract triplet contract.  It never uses family/source labels as features and
does not retrain the model.  Missing screen observations are explicitly
counted, so a compatibility failure cannot be hidden as a model success.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


PG77_PATH = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG77_CHECKPOINT = ROOT / "artifacts" / "pg77-real-triplet-transformer" / "model.pt"
PG53_REPORT = ROOT / "research" / "pg53_cross_source_typed_replay_report_v1.json"
REPORT_PATH = ROOT / "research" / "pg78_multisource_triplet_holdout_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg78_multisource_triplet_holdout_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg78_multisource_triplet_holdout_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg78_multisource_triplet_holdout_report_v1.md"
SEED = 20780403


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_projection(phase: dict[str, Any] | None) -> dict[str, Any]:
    """Map PG-53 bounded geometry into PG-77's anonymous shape vocabulary."""

    phase = dict(phase or {})
    response = dict(phase.get("response") or {})
    surface = dict(phase.get("surface_observation") or {})
    geometry = dict(phase.get("generic_effect_geometry") or {})
    shape = dict(response.get("shape") or {})
    response["shape"] = {
        "kind": str(shape.get("kind", "object")),
        "key_count": int(shape.get("key_count", geometry.get("object_count", 0)) or 0),
        "scalar_count": int(shape.get("scalar_count", geometry.get("leaf_count", 0)) or 0),
        "array_count": int(shape.get("array_count", geometry.get("array_count", 0)) or 0),
    }
    # These are anonymous bounded counts, not HTML/SQL/family indicators.
    response["html_tag_count"] = int(geometry.get("leaf_count", 0) or 0)
    response["form_count"] = int(geometry.get("object_count", 0) or 0)
    response["input_count"] = int(surface.get("boolean_field_count", 0) or 0)
    response["script_count"] = int(surface.get("numeric_field_count", 0) or 0)
    response["result_row_count"] = int(geometry.get("array_item_count", 0) or 0)
    response["marker_reflected"] = bool(surface.get("true_boolean_count", 0))
    response["marker_count"] = int(surface.get("true_boolean_count", 0) or 0)
    response["dom_shape"] = {
        "node_count": int(geometry.get("leaf_count", 0) or 0),
        "svg_count": int(geometry.get("array_count", 0) or 0),
        "event_handler_attribute_count": int(surface.get("true_boolean_count", 0) or 0),
        "form_count": int(geometry.get("object_count", 0) or 0),
        "input_count": int(surface.get("boolean_field_count", 0) or 0),
        "script_count": int(surface.get("numeric_field_count", 0) or 0),
    }
    response["location_origin"] = "none"
    return response


def _step_from_pg53(pg77: Any, row: dict[str, Any]) -> dict[str, Any]:
    control = _canonical_projection(row.get("control"))
    screen_phase = row.get("screen") or row.get("control")
    screen = _canonical_projection(screen_phase)
    candidate = _canonical_projection(row.get("candidate"))
    oracle = dict((row.get("candidate") or {}).get("oracle") or {})
    method = str(row.get("method", "GET")).upper()
    return {
        "step_id": str(row["sample_id"]),
        "sampling_seed": int(row.get("sampling_seed", 0)),
        "action_manifest": {"method": method, "placement": "query" if method == "GET" else "form"},
        "neutral_projection": control,
        "negative_probe_projection": screen,
        "response_projection": candidate,
        "oracle_projection": oracle,
        "family_label": str(row.get("family", "unknown")),
        "source_label": str(row.get("source_id", "unknown")),
        "implementation_label": str(row.get("implementation", "unknown")),
        "variant_label": str(row.get("variant", "unknown")),
        "screen_present": row.get("screen") is not None,
        "fresh_reset": row.get("fresh_reset"),
        "expected": "confirm" if bool(oracle.get("positive")) else "reject",
    }


def _rows(pg77: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(PG53_REPORT.read_text(encoding="utf-8"))
    steps = [_step_from_pg53(pg77, row) for row in report.get("rows", [])]
    rows: list[dict[str, Any]] = []
    for step in steps:
        tokens, oracle_index = pg77._row_tokens(step, step["response_projection"], step["oracle_projection"])
        rows.append({"trace_id": step["step_id"], "split": "pg53_multisource_holdout", "role": "positive" if step["expected"] == "confirm" else "negative", "tokens": tokens, "oracle_index": oracle_index, "expected": step["expected"], "source_label": step["source_label"], "implementation_label": step["implementation_label"], "variant_label": step["variant_label"], "family_label": step["family_label"], "method": step["action_manifest"]["method"], "screen_present": step["screen_present"], "raw_probe_stored": False, "raw_response_stored": False})
    diagnostics = {"source_report": str(PG53_REPORT.relative_to(ROOT)), "case_count": len(steps), "fresh_reset_count": int(report.get("metrics", {}).get("fresh_reset_count", 0)), "screen_present_count": sum(int(step["screen_present"]) for step in steps), "screen_missing_count": sum(int(not step["screen_present"]) for step in steps), "source_count": len({step["source_label"] for step in steps}), "code_implementation_count": len({step["implementation_label"] for step in steps}), "implementation_count": len({(step["implementation_label"], step["variant_label"]) for step in steps}), "family_count": len({step["family_label"] for step in steps}), "method_counts": dict(Counter(step["action_manifest"]["method"] for step in steps)), "source_report_sha256": hashlib.sha256(PG53_REPORT.read_bytes()).hexdigest()}
    return rows, diagnostics


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    max_len = max(len(row["tokens"]) for row in rows)
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    positions: list[int] = []
    unknown_token_count = 0
    for index, row in enumerate(rows):
        encoded = []
        for token in row["tokens"]:
            if token not in vocabulary:
                unknown_token_count += 1
            encoded.append(vocabulary.get(token, unk))
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), torch.tensor([unknown_token_count], dtype=torch.long)


def run() -> dict[str, Any]:
    if not PG77_CHECKPOINT.exists():
        raise RuntimeError("PG-78 requires the PG-77 candidate checkpoint")
    pg77 = _load(PG77_PATH, "pg78_pg77_runtime")
    rows, diagnostics = _rows(pg77)
    checkpoint = torch.load(PG77_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = dict(checkpoint["vocabulary"])
    ids, mask, positions, unknown_token_count = _encode(rows, vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTraceTransformer(len(vocabulary), d_model=96, nhead=4, layers=2, max_len=int(checkpoint.get("max_len", 128))).to(device)
    model.load_state_dict(checkpoint["transformer_state"])
    model.eval()
    head = pg77.RuleIRHead(int(checkpoint["hidden_dim"])).to(device)
    head.load_state_dict(checkpoint["rule_ir_head_state"])
    head.eval()
    with torch.no_grad():
        hidden = pg77._hidden_at_oracle(model, ids.to(device), mask.to(device), positions.to(device)).detach()
    # Reconstruct the PG-77 training reference without using PG-53 labels.
    train_rows, _, _, _ = pg77._make_rows()
    train_ids, train_mask, train_positions = pg77._encode(train_rows, vocabulary)
    with torch.no_grad():
        train_hidden = pg77._hidden_at_oracle(model, train_ids.to(device), train_mask.to(device), train_positions.to(device)).detach()
    threshold = float(checkpoint["ood_distance_threshold"])
    metrics, details = pg77._evaluate_head(head, hidden, rows, train_hidden, threshold, unknown=False)
    by_impl: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    for group_key, key in (("implementation_label", "by_implementation"), ("family_label", "by_family")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row, detail in zip(rows, details):
            grouped[str(row[group_key])].append(detail)
        target = by_impl if key == "by_implementation" else by_family
        for group, items in grouped.items():
            positives = [item for item in items if item["expected"] == "confirm"]
            target[group] = {"count": len(items), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positives) / max(len(positives), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in items), "abstain_count": sum(int(item["decision"] == "abstain") for item in items)}
    checks = {"typed_positive_and_negative_rows": len(rows) == 270, "fresh_reset_attested": diagnostics["fresh_reset_count"] >= len(rows), "get_post_covered": diagnostics["method_counts"] == {"GET": 135, "POST": 135}, "multi_implementation": diagnostics["source_count"] >= 5, "multi_family": diagnostics["family_count"] >= 8, "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows), "unknown_token_count_zero": int(unknown_token_count.item()) == 0, "false_accept_zero": int(metrics.get("false_accept_count", 1)) == 0, "known_recall_min": float(metrics.get("confirm_recall", 0.0)) >= 0.80, "screen_contract_complete": diagnostics["screen_missing_count"] == 0}
    report = {"protocol_id": "pg-pk-78-multisource-triplet-holdout-v1", "schema_version": "pg78-multisource-triplet-holdout-report-v1", "status": "completed_evaluation", "source": {"pg53_report": str(PG53_REPORT.relative_to(ROOT)), "pg53_report_sha256": diagnostics["source_report_sha256"], "candidate_checkpoint": str(PG77_CHECKPOINT.relative_to(ROOT)), "candidate_checkpoint_sha256": hashlib.sha256(PG77_CHECKPOINT.read_bytes()).hexdigest(), "model_retrained_on_pg53": False, "family_in_tokens": False, "source_in_tokens": False, "oracle_in_tokens_before_target": False, "device": str(device)}, "dataset": diagnostics, "metrics": metrics, "by_implementation": by_impl, "by_family": by_family, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "holdout_evaluation_only", "reason": "PG-53 is a broad source holdout; screen contract is incomplete and the candidate must not be retrained on it"}, "formal_claim": {"allowed": False, "reason": "external holdout is diagnostic and projection contract is incomplete"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg78-multisource-triplet-holdout-trace-v1", "evaluation_only": True, "training_eligible": False, "rows": [{key: row[key] for key in ("trace_id", "split", "role", "expected", "source_label", "implementation_label", "family_label", "method", "screen_present", "raw_probe_stored", "raw_response_stored")} for row in rows], "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-78-multisource-triplet-holdout-v1", "schema_version": "pg78-multisource-triplet-holdout-protocol-v1", "source_contract": {"source_report": str(PG53_REPORT.relative_to(ROOT)), "fresh_reset_required": True, "loopback_only": True, "multi_implementation_required": True, "multi_family_required": True, "get_post_required": True}, "model_contract": {"checkpoint": str(PG77_CHECKPOINT.relative_to(ROOT)), "family_in_tokens": False, "source_in_tokens": False, "oracle_after_target_only": True, "retraining_forbidden": True}, "required_gates": {"known_recall_min": 0.80, "false_accept_zero": True, "screen_contract_complete": True, "unknown_token_count_zero": True, "raw_free": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG79 fresh unified triplet collector over multiple independent fixtures"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-78 多实现多族 triplet holdout\n\n" + f"cases={len(rows)}；implementations={diagnostics['implementation_count']}；families={diagnostics['family_count']}；GET/POST={diagnostics['method_counts']}；screen missing={diagnostics['screen_missing_count']}。\n\nknown recall={metrics.get('confirm_recall', 0.0)}；false accepts={metrics.get('false_accept_count', 0)}；能力门=`{report['capability_gate']['status']}`。\n\ntraining/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "case_count": result["dataset"]["case_count"], "confirm_recall": result["metrics"]["confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "screen_missing_count": result["dataset"]["screen_missing_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
