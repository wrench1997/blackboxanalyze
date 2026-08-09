"""PG-232: strict source-heldout audit for the PG-231 adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg230_next_token_quality_funnel import FrozenXXLNextTokenAdapter, LANES, LANE_INDEX, REPAIR_INDEX, build_vocabulary, digest  # noqa: E402


RESEARCH = ROOT / "research"
PG231_DATASET = RESEARCH / "pg231_feedback_trajectory_dataset_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg232_source_holdout_audit_report_v1.json"
TRACE = RESEARCH / "pg232_source_holdout_audit_trace_v1.json"
PROTOCOL = RESEARCH / "pg232_source_holdout_audit_protocol_v1.json"
MARKDOWN = RESEARCH / "pg232_source_holdout_audit_report_v1.md"


def _load_pg231() -> Any:
    path = ROOT / "scripts" / "run_pg231_feedback_trajectory_training.py"
    spec = importlib.util.spec_from_file_location("pg231_training_for_pg232", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-231 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG231 = _load_pg231()


def _metrics(model: FrozenXXLNextTokenAdapter, context: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor) -> dict[str, Any]:
    return PG231._evaluate(model, context, targets, positions)


def _train_fold(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], base: Any, input_vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    target_vocab = build_vocabulary(train_rows)
    train_ids, train_targets, train_lane, train_repair = PG231._encode(train_rows, input_vocab, target_vocab, device)
    hold_ids, hold_targets, hold_lane, hold_repair = PG231._encode(holdout_rows, input_vocab, target_vocab, device)
    with torch.no_grad():
        train_context = base.base.body.encode(train_ids, train_ids.ne(0)).detach().clone()
        hold_context = base.base.body.encode(hold_ids, hold_ids.ne(0)).detach().clone()
    train_positions = PG231._positions(train_rows, train_context.shape[1], device)
    hold_positions = PG231._positions(holdout_rows, hold_context.shape[1], device)
    torch.manual_seed(232 + len(train_rows) + len(holdout_rows))
    model = FrozenXXLNextTokenAdapter(d_model=int(train_context.shape[-1]), hidden_dim=64, vocab_size=len(target_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lane_counts = torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)
    repair_counts = torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0)
    lane_weights = (lane_counts.sum() / lane_counts).to(device)
    repair_weights = (repair_counts.sum() / repair_counts).to(device)
    for _ in range(60):
        model.train()
        output = model(train_context, classification_positions=train_positions)
        token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_targets.reshape(-1), ignore_index=0)
        loss = token_loss + 0.30 * nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {"train": _metrics(model, train_context, (train_targets, train_lane, train_repair), train_positions), "holdout": _metrics(model, hold_context, (hold_targets, hold_lane, hold_repair), hold_positions), "target_vocabulary_size": len(target_vocab)}


def main() -> int:
    dataset = json.loads(PG231_DATASET.read_text(encoding="utf-8-sig"))
    usable = [dict(record) for record in dataset.get("records", []) if record.get("lane") not in {"quarantine", "reject"}]
    sources = sorted({str(record["source"]) for record in usable})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    rows: list[dict[str, Any]] = []
    for source in sources:
        train_rows = [record for record in usable if str(record["source"]) != source]
        holdout_rows = [record for record in usable if str(record["source"]) == source]
        result = _train_fold(train_rows, holdout_rows, base, input_vocab, device)
        rows.append({"heldout_source": source, "train_rows": len(train_rows), "holdout_rows": len(holdout_rows), "holdout_lane_counts": dict(Counter(str(record["lane"]) for record in holdout_rows)), **result})
    body_after = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    strict_pass = all(
        float(row["holdout"]["lane_accuracy"]) >= 0.80
        and float(row["holdout"]["repair_accuracy"]) >= 0.80
        and (int(row["holdout"]["self_error_count"]) == 0 or float(row["holdout"]["self_error_recall"]) >= 0.80)
        for row in rows
    )
    report = {"protocol_id": "pg-pk-232-source-holdout-audit-v1", "schema_version": "pg232-source-holdout-audit-v1", "status": "completed_strict_source_holdout_audit", "device": str(device), "source_dataset": str(PG231_DATASET.relative_to(ROOT)), "usable_records": len(usable), "heldout_source_count": len(rows), "folds": rows, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_after, "frozen_body_changed": body_before != body_after, "strict_source_holdout_pass": strict_pass, "promotion": {"memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "training_promotion_allowed": False}, "honesty": {"source_holdout_is_stricter_than_pg231": True, "small_source_folds_are_reported": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg232-source-holdout-audit-protocol-v1", "source_group_is_never_in_training": True, "quarantine_excluded": True, "lane_repair_target_leakage_guard": True, "minimum_lane_accuracy": 0.80, "minimum_repair_accuracy": 0.80, "minimum_self_error_recall_when_present": 0.80, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write(REPORT, report)
    _write(TRACE, {"schema_version": "pg232-source-holdout-audit-trace-v1", "folds": rows, "strict_source_holdout_pass": strict_pass, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-232 strict source-heldout audit", "", f"usable={len(usable)}; source_folds={len(rows)}; strict_pass={strict_pass}", *[f"{row['heldout_source']}: holdout={row['holdout_rows']}; lane={row['holdout']['lane_accuracy']}; repair={row['holdout']['repair_accuracy']}; self_error_recall={row['holdout']['self_error_recall']}" for row in rows], "", "任何一个 source 留出折叠不过门，就不提升长期记忆或漏洞结论。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "strict_source_holdout_pass": strict_pass, "folds": rows, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

