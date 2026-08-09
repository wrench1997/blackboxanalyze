"""PG-89: frozen PG-86 replay on the independent PG-88 matrix.

This evaluator answers the important question after PG-88 collection: does
the current surface-signal Rule IR head transfer to fresh seeds without any
weight update, oracle-conditioned input, or memory write?  It consumes only
bounded projections from the PG-88 trace and uses the same vocabulary,
threshold and model checkpoint that PG-86 already froze.
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
CHECKPOINT_PATH = ROOT / "artifacts" / "pg86-surface-signal-composite-transformer" / "model.pt"
REFERENCE_DATASET_PATH = ROOT / "research" / "pg86_surface_signal_composite_trace_dataset_v1.json"
INPUT_TRACE_PATH = ROOT / "research" / "pg88_independent_html_dom_matrix_trace_v1.json"
DATASET_PATH = ROOT / "research" / "pg89_pg86_frozen_html_dom_replay_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg89_pg86_frozen_html_dom_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg89_pg86_frozen_html_dom_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg89_pg86_frozen_html_dom_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg89_pg86_frozen_html_dom_replay_report_v1.md"
PROTOCOL_ID = "pg-pk-89-pg86-frozen-html-dom-replay-v1"
CONFIDENCE_THRESHOLD = 0.70


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    if not rows:
        raise ValueError("cannot encode an empty row list")
    actual_max = max(len(row["tokens"]) for row in rows)
    if actual_max > max_len:
        raise RuntimeError(f"PG-89 sequence exceeds frozen max_len: {actual_max} > {max_len}")
    ids = torch.full((len(rows), actual_max), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), actual_max), dtype=torch.bool)
    positions: list[int] = []
    unknown = 0
    for index, row in enumerate(rows):
        tokens = list(row["tokens"])
        encoded = [vocabulary.get(token, unk) for token in tokens]
        unknown += sum(int(token not in vocabulary) for token in tokens)
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), unknown


def _trace_sha256() -> str:
    return hashlib.sha256(INPUT_TRACE_PATH.read_bytes()).hexdigest()


def _replay_rows(pg86: Any, pg77: Any, trace: dict[str, Any], sha256_json: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        # Adapter fields are bounded and do not carry body, payload, cookie or
        # evaluator state.  The oracle is only used after ORACLE_TARGET by the
        # tokenizer, matching the PG-86 input contract.
        neutral = pg86._adapter(dict(step.get("neutral_projection") or step.get("baseline_projection") or {}), sha256_json)
        negative = pg86._adapter(dict(step.get("negative_probe_projection") or neutral), sha256_json)
        positive = pg86._adapter(dict(step.get("response_projection") or {}), sha256_json)
        for role, candidate, oracle in (("positive", positive, dict(step.get("oracle_projection") or {})), ("negative", negative, dict(step.get("negative_oracle_projection") or {}))):
            tokens, oracle_index = pg86._row_tokens(pg77, {"action_manifest": step.get("action_manifest"), "neutral_projection": neutral, "negative_probe_projection": negative}, candidate, oracle)
            rows.append({
                "trace_id": f"{step['step_id']}-{role}",
                "step_id": str(step["step_id"]),
                "seed": int(step.get("sampling_seed", 0)),
                "target_instance_id": str(step.get("target_instance_id", "")),
                "family": str(step.get("hypothesis", "unknown")),
                "surface": str((step.get("action_manifest") or {}).get("route_template_id", "unknown")),
                "method": str((step.get("action_manifest") or {}).get("method", "GET")).upper(),
                "role": role,
                "tokens": tokens,
                "oracle_index": oracle_index,
                "expected": "confirm" if bool(oracle.get("positive")) else "reject",
                "raw_probe_stored": False,
                "raw_response_stored": False,
            })
    return rows


def _evaluate(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], checkpoint: dict[str, Any], pg77: Any) -> tuple[dict[str, Any], list[dict[str, Any]], str, int, int]:
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
        details.append({
            "trace_id": row["trace_id"], "step_id": row["step_id"], "seed": row["seed"], "role": row["role"],
            "method": row["method"], "expected": row["expected"], "raw_prediction": raw, "decision": decision,
            "confidence": round(float(confidence), 6), "ood_distance": round(distance, 6),
        })
    positives = [item for item in details if item["expected"] == "confirm"]
    negatives = [item for item in details if item["expected"] == "reject"]
    by_seed: dict[int, dict[str, Any]] = {}
    for seed in sorted({int(item["seed"]) for item in details}):
        subset = [item for item in details if int(item["seed"]) == seed]
        pos = [item for item in subset if item["expected"] == "confirm"]
        by_seed[seed] = {
            "count": len(subset), "positive_count": len(pos),
            "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in pos) / max(len(pos), 1), 6),
            "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in subset),
            "abstain_count": sum(int(item["decision"] == "abstain") for item in subset),
        }
    metrics = {
        "row_count": len(details), "typed_positive_count": len(positives), "typed_negative_count": len(negatives),
        "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positives) / max(len(positives), 1), 6),
        "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in details),
        "abstain_count": sum(int(item["decision"] == "abstain") for item in details),
        "ood_distance_threshold": threshold, "confidence_threshold": CONFIDENCE_THRESHOLD,
        "unknown_token_count": unknown_count, "reference_unknown_token_count": reference_unknown,
        "seed_metrics": by_seed,
        "seed_min_confirm_recall": min((item["confirm_recall"] for item in by_seed.values()), default=0.0),
        "device": str(device),
    }
    return metrics, details, str(device), unknown_count, reference_unknown


def run() -> dict[str, Any]:
    pg86 = _load(PG86_SCRIPT, "pg89_pg86_runtime")
    pg77 = _load(PG77_SCRIPT, "pg89_pg77_runtime")
    sha256_json = __import__("app.trace_aligned_dataset", fromlist=["sha256_json"]).sha256_json
    trace = json.loads(INPUT_TRACE_PATH.read_text(encoding="utf-8"))
    reference_dataset = json.loads(REFERENCE_DATASET_PATH.read_text(encoding="utf-8"))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    rows = _replay_rows(pg86, pg77, trace, sha256_json)
    reference_rows = [row for row in reference_dataset.get("rows", []) if row.get("split") == "train"]
    metrics, details, device, unknown_count, reference_unknown = _evaluate(rows, reference_rows, checkpoint, pg77)
    methods = {str(step.get("action_manifest", {}).get("method", "")).upper() for step in trace.get("steps", [])}
    steps = trace.get("steps", [])
    expected_steps = 28
    checks = {
        "independent_pg88_trace_complete": len(steps) == expected_steps and len(rows) == expected_steps * 2,
        "typed_positive_count": metrics["typed_positive_count"] == expected_steps,
        "typed_negative_count": metrics["typed_negative_count"] == expected_steps,
        "fresh_target_per_case": len({str(step.get("target_instance_id", "")) for step in steps}) == expected_steps and all(bool(step.get("fresh_reset", {}).get("fresh_target")) for step in steps),
        "get_post_covered": methods == {"GET", "POST"},
        "unknown_token_count_zero": unknown_count == 0 and reference_unknown == 0,
        "false_accept_zero": metrics["false_accept_count"] == 0,
        "known_recall_min": metrics["confirm_recall"] >= 0.80,
        "cross_seed_recall_min": metrics["seed_min_confirm_recall"] >= 0.75,
        "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows),
    }
    status = "passed" if all(checks.values()) else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg89-pg86-frozen-html-dom-replay-report-v1",
        "status": "completed_evaluation",
        "source": {
            "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)),
            "input_trace_sha256": _trace_sha256(),
            "reference_dataset": str(REFERENCE_DATASET_PATH.relative_to(ROOT)),
            "device": device, "training": False, "memory_write": False,
            "projection_adapter": "pg86_surface_signal_composite_adapter_v3",
            "oracle_after_target_only": True,
        },
        "dataset": {"step_count": len(steps), "row_count": len(rows), "get_post_counts": {"GET": sum(int(step.get("action_manifest", {}).get("method", "").upper() == "GET") for step in steps), "POST": sum(int(step.get("action_manifest", {}).get("method", "").upper() == "POST") for step in steps)}},
        "metrics": metrics,
        "details": details,
        "capability_gate": {"status": status, "checks": checks, "blocking_reasons": [key for key, value in checks.items() if not value], "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "frozen_replay_only", "reason": "PG89 is a frozen independent-target evaluation; no weight or memory update is permitted"},
        "artifacts": {"report": str(REPORT_PATH.relative_to(ROOT)), "protocol": str(PROTOCOL_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT))},
    }
    dataset = {"schema_version": "pg89-pg86-frozen-html-dom-replay-dataset-v1", "dataset_id": "pg89-pg86-frozen-pg88-matrix", "evaluation_only": True, "training_eligible": False, "rows": rows, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "long_term_memory_write": False}
    trace_out = {"schema_version": "pg89-pg86-frozen-html-dom-replay-trace-v1", "protocol_id": PROTOCOL_ID, "input_trace_sha256": report["source"]["input_trace_sha256"], "evaluation_only": True, "rows": [{"trace_id": item["trace_id"], "decision": item["decision"], "expected": item["expected"], "seed": item["seed"], "raw_probe_stored": False, "raw_response_stored": False} for item in details], "online_weight_update": False, "long_term_memory_write": False}
    protocol = {"protocol_id": PROTOCOL_ID, "schema_version": "pg89-pg86-frozen-html-dom-replay-protocol-v1", "frozen_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "input_trace": str(INPUT_TRACE_PATH.relative_to(ROOT)), "input_trace_sha256": report["source"]["input_trace_sha256"], "oracle_after_target_only": True, "raw_persistence_forbidden": True, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG90 cross-seed/implementation Codex review"}
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-89 Frozen PG-86 replay\n\n" + f"rows={len(rows)}；recall={metrics['confirm_recall']}；seed_min_recall={metrics['seed_min_confirm_recall']}；false_accept={metrics['false_accept_count']}；unknown_tokens={unknown_count}。\n\n能力门：`{status}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "status": result["capability_gate"]["status"], "confirm_recall": result["metrics"]["confirm_recall"], "seed_min_confirm_recall": result["metrics"]["seed_min_confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "device": result["source"]["device"], "training_allowed": False}, ensure_ascii=False, indent=2))
