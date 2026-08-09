"""PG-80: replay the frozen PG-77 Trace Transformer on PG-79 triplets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


PG77_SCRIPT = ROOT / "scripts" / "run_pg77_real_triplet_transformer.py"
PG77_CHECKPOINT = ROOT / "artifacts" / "pg77-real-triplet-transformer" / "model.pt"
PG79_TRACE = ROOT / "research" / "pg79_fresh_unified_triplet_collector_trace_v1.json"
REPORT_PATH = ROOT / "research" / "pg80_frozen_trace_transformer_replay_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg80_frozen_trace_transformer_replay_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg80_frozen_trace_transformer_replay_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg80_frozen_trace_transformer_replay_report_v1.md"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(pg77: Any) -> list[dict[str, Any]]:
    trace = json.loads(PG79_TRACE.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        for role, projection, oracle in (("positive", step["response_projection"], step["oracle_projection"]), ("negative", step["negative_probe_projection"], step["negative_oracle_projection"])):
            tokens, oracle_index = pg77._row_tokens(step, dict(projection), dict(oracle))
            rows.append({"trace_id": f"{step['step_id']}-{role}", "tokens": tokens, "oracle_index": oracle_index, "expected": "confirm" if role == "positive" and bool(oracle.get("positive")) else "reject", "role": role, "seed": step["sampling_seed"], "raw_probe_stored": False, "raw_response_stored": False})
    return rows


def _encode(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pad, unk = vocabulary["<PAD>"], vocabulary["<UNK>"]
    max_len = max(len(row["tokens"]) for row in rows)
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    positions: list[int] = []
    unknown = 0
    for index, row in enumerate(rows):
        encoded = []
        for token in row["tokens"]:
            unknown += int(token not in vocabulary)
            encoded.append(vocabulary.get(token, unk))
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
        positions.append(int(row["oracle_index"]))
    return ids, mask, torch.tensor(positions, dtype=torch.long), unknown


def run() -> dict[str, Any]:
    if not PG77_CHECKPOINT.exists():
        raise RuntimeError("PG-80 requires the PG-77 checkpoint")
    pg77 = _load(PG77_SCRIPT, "pg80_pg77_runtime")
    checkpoint = torch.load(PG77_CHECKPOINT, map_location="cpu", weights_only=False)
    rows = _rows(pg77)
    vocabulary = dict(checkpoint["vocabulary"])
    ids, mask, positions, unknown_count = _encode(rows, vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTraceTransformer(len(vocabulary), d_model=96, nhead=4, layers=2, max_len=int(checkpoint.get("max_len", 128))).to(device)
    model.load_state_dict(checkpoint["transformer_state"])
    model.eval()
    head = pg77.RuleIRHead(int(checkpoint["hidden_dim"])).to(device)
    head.load_state_dict(checkpoint["rule_ir_head_state"])
    head.eval()
    train_rows, _, _, _ = pg77._make_rows()
    train_ids, train_mask, train_positions = pg77._encode(train_rows, vocabulary)
    with torch.no_grad():
        hidden = pg77._hidden_at_oracle(model, ids.to(device), mask.to(device), positions.to(device)).detach()
        reference = pg77._hidden_at_oracle(model, train_ids.to(device), train_mask.to(device), train_positions.to(device)).detach()
    threshold = float(checkpoint["ood_distance_threshold"])
    metrics, details = pg77._evaluate_head(head, hidden, rows, reference, threshold, unknown=False)
    checks = {"trace_triplet_count": len(rows) == 540, "raw_free": all(not row["raw_probe_stored"] and not row["raw_response_stored"] for row in rows), "unknown_token_count_zero": unknown_count == 0, "false_accept_zero": int(metrics.get("false_accept_count", 1)) == 0, "known_recall_min": float(metrics.get("confirm_recall", 0.0)) >= 0.80}
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        subset = [detail for row, detail in zip(rows, details) if int(row["seed"]) == seed]
        positive = [item for item in subset if item["expected"] == "confirm"]
        by_seed[str(seed)] = {"count": len(subset), "confirm_recall": round(sum(int(item["decision"] == "confirm") for item in positive) / max(len(positive), 1), 6), "false_accept_count": sum(int(item["expected"] == "reject" and item["decision"] == "confirm") for item in subset), "abstain_count": sum(int(item["decision"] == "abstain") for item in subset)}
    report = {"protocol_id": "pg-pk-80-frozen-trace-transformer-replay-v1", "schema_version": "pg80-frozen-trace-transformer-replay-report-v1", "status": "completed_evaluation", "source": {"triplet_trace": str(PG79_TRACE.relative_to(ROOT)), "triplet_trace_sha256": hashlib.sha256(PG79_TRACE.read_bytes()).hexdigest(), "candidate_checkpoint": str(PG77_CHECKPOINT.relative_to(ROOT)), "candidate_checkpoint_sha256": hashlib.sha256(PG77_CHECKPOINT.read_bytes()).hexdigest(), "retrained": False, "device": str(device)}, "dataset": {"step_count": len(rows) // 2, "positive_negative_row_count": len(rows), "seed_count": len(by_seed), "get_post_covered": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}, "metrics": metrics | {"by_seed": by_seed, "unknown_token_count": unknown_count}, "details": details, "capability_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "frozen_replay_only", "reason": "PG-80 is a frozen model replay; no weights or memory are updated"}, "formal_claim": {"allowed": False, "reason": "frozen candidate replay is not broad vulnerability capability"}}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps({"schema_version": "pg80-frozen-trace-transformer-replay-trace-v1", "evaluation_only": True, "training_eligible": False, "rows": [{"trace_id": row["trace_id"], "expected": row["expected"], "role": row["role"], "raw_probe_stored": False, "raw_response_stored": False} for row in rows], "online_weight_update": False, "long_term_memory_write": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps({"protocol_id": "pg-pk-80-frozen-trace-transformer-replay-v1", "schema_version": "pg80-frozen-trace-transformer-replay-protocol-v1", "model_contract": {"checkpoint": str(PG77_CHECKPOINT.relative_to(ROOT)), "retraining_forbidden": True, "memory_write_forbidden": True}, "input_contract": {"family_in_tokens": False, "oracle_before_target_forbidden": True, "raw_persistence_forbidden": True}, "required_gates": {"trace_triplet_count": True, "false_accept_zero": True, "known_recall_min": 0.80, "unknown_token_count_zero": True}, "run_result": {"capability_gate": report["capability_gate"], "training_allowed": False, "memory_promotion_allowed": False}, "next_experiment": "PG81 canonical projection schema version and retraining-isolated ablation"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("# PG-80 Frozen Trace Transformer replay\n\n" + f"triplet steps={len(rows)//2}；positive/negative rows={len(rows)}；device={device}；recall={metrics.get('confirm_recall', 0.0)}；false accepts={metrics.get('false_accept_count', 0)}；unknown tokens={unknown_count}。\n\n能力门：`{report['capability_gate']['status']}`；training/memory promotion=`false`。\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"protocol_id": result["protocol_id"], "status": result["capability_gate"]["status"], "step_count": result["dataset"]["step_count"], "confirm_recall": result["metrics"]["confirm_recall"], "false_accept_count": result["metrics"]["false_accept_count"], "unknown_token_count": result["metrics"]["unknown_token_count"], "training_allowed": False}, ensure_ascii=False, indent=2))
