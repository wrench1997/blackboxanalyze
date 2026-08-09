"""PG-235: train an abstention-aware policy on wire-grounded trajectories."""

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

from app.pg230_next_token_quality_funnel import LANES, LANE_INDEX, REPAIR_INDEX, build_vocabulary, digest  # noqa: E402
from app.pg235_failure_conditioned_policy import ACTION_CLASSES, ACTION_INDEX, FrozenXXLFailurePolicy, PG235_SCHEMA, action_target, prepare_wire_record  # noqa: E402


RESEARCH = ROOT / "research"
PG233_DATASET = RESEARCH / "pg233_cross_family_capacity_dataset_v1.json"
PG234_CATALOG = RESEARCH / "pg234_pikachu_payload_wire_catalog_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg235_failure_conditioned_policy_training_report_v1.json"
DATASET = RESEARCH / "pg235_failure_conditioned_policy_dataset_v1.json"
TRACE = RESEARCH / "pg235_failure_conditioned_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg235_failure_conditioned_policy_protocol_v1.json"
MARKDOWN = RESEARCH / "pg235_failure_conditioned_policy_report_v1.md"


def _load_pg231() -> Any:
    path = ROOT / "scripts" / "run_pg231_feedback_trajectory_training.py"
    spec = importlib.util.spec_from_file_location("pg231_training_for_pg235", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-231 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG231 = _load_pg231()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = json.loads(PG233_DATASET.read_text(encoding="utf-8-sig"))
    base_records = [dict(record) for record in base.get("records", [])]
    catalog = json.loads(PG234_CATALOG.read_text(encoding="utf-8-sig"))
    wire_records = [prepare_wire_record(row) for row in catalog.get("rows", [])]
    return base_records, wire_records


def _encode(rows: list[dict[str, Any]], input_vocab: dict[str, int], target_vocab: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return PG231._encode(rows, input_vocab, target_vocab, device)


def _positions(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(row.get("classification_position", 0)), 0), max(width - 1, 0)) for row in rows], dtype=torch.long, device=device)


def _evaluate(model: FrozenXXLFailurePolicy, context: torch.Tensor, encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    ids, token_targets, lane_targets, repair_targets = encoded
    action_targets = torch.tensor([ACTION_INDEX[action_target(row)] for row in rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    valid = token_targets.ne(0)
    token_pred = output["token"].argmax(-1)
    lane_pred = output["lane"].argmax(-1)
    repair_pred = output["repair"].argmax(-1)
    action_pred = output["action"].argmax(-1)
    token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_targets.reshape(-1), ignore_index=0)
    abstain_mask = action_targets == ACTION_INDEX["abstain"]
    send_mask = action_targets == ACTION_INDEX["send_candidate"]
    return {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "lane_accuracy": round(float((lane_pred == lane_targets).float().mean().item()), 8),
        "repair_accuracy": round(float((repair_pred == repair_targets).float().mean().item()), 8),
        "action_accuracy": round(float((action_pred == action_targets).float().mean().item()), 8),
        "abstain_recall": round(float(((action_pred == ACTION_INDEX["abstain"]) & abstain_mask).sum().item() / max(int(abstain_mask.sum().item()), 1)), 8),
        "false_send_count": int(((action_pred == ACTION_INDEX["send_candidate"]) & ~send_mask).sum().item()),
        "send_count": int(send_mask.sum().item()),
        "abstain_count": int(abstain_mask.sum().item()),
    }


def main() -> int:
    base_records, wire_records = _load_records()
    all_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in base_records + wire_records:
        key = str(record.get("trajectory_hash", digest(record.get("tokens", []))))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        all_records.append(record)
    holdout_family = {"dom", "redirect"}
    train_rows = [record for record in all_records if record.get("family_class") not in holdout_family and record.get("lane") not in {"quarantine", "reject"}]
    holdout_rows = [record for record in all_records if record.get("family_class") in holdout_family]
    if not train_rows or not holdout_rows:
        raise RuntimeError("PG-235 requires non-empty known-family train and unseen-family holdout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    target_vocab = build_vocabulary(train_rows)
    train_encoded = _encode(train_rows, input_vocab, target_vocab, device)
    holdout_encoded = _encode(holdout_rows, input_vocab, target_vocab, device)
    with torch.no_grad():
        train_context = base.base.body.encode(train_encoded[0], train_encoded[0].ne(0)).detach().clone()
        holdout_context = base.base.body.encode(holdout_encoded[0], holdout_encoded[0].ne(0)).detach().clone()
    train_positions = _positions(train_rows, train_context.shape[1], device)
    holdout_positions = _positions(holdout_rows, holdout_context.shape[1], device)
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: FrozenXXLFailurePolicy | None = None
    for hidden_dim in (64, 128, 256):
        torch.manual_seed(235 + hidden_dim)
        model = FrozenXXLFailurePolicy(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        train_lane, train_repair = train_encoded[2], train_encoded[3]
        train_action = torch.tensor([ACTION_INDEX[action_target(row)] for row in train_rows], dtype=torch.long, device=device)
        lane_counts = torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)
        repair_counts = torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0)
        action_counts = torch.bincount(train_action, minlength=len(ACTION_CLASSES)).float().clamp_min(1.0)
        lane_weights = (lane_counts.sum() / lane_counts).to(device)
        repair_weights = (repair_counts.sum() / repair_counts).to(device)
        action_weights = (action_counts.sum() / action_counts).to(device)
        for _ in range(90):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_encoded[1].reshape(-1), ignore_index=0)
            loss = token_loss + 0.30 * nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights) + 0.40 * nn.functional.cross_entropy(output["action"], train_action, weight=action_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": _evaluate(model, train_context, train_encoded, train_positions, train_rows, device), "unseen_family_holdout": _evaluate(model, holdout_context, holdout_encoded, holdout_positions, holdout_rows, device)}
        variants.append(result)
        hold = result["unseen_family_holdout"]
        key = (hold["false_send_count"], -hold["abstain_recall"], -hold["action_accuracy"], hold["token_loss"])
        old_key = None if selected is None else (selected["unseen_family_holdout"]["false_send_count"], -selected["unseen_family_holdout"]["abstain_recall"], -selected["unseen_family_holdout"]["action_accuracy"], selected["unseen_family_holdout"]["token_loss"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    body_after = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    if selected is None or selected_model is None:
        raise RuntimeError("PG-235 no selected policy")
    hold = selected["unseen_family_holdout"]
    strict_pass = hold["false_send_count"] == 0 and hold["abstain_recall"] >= 0.80
    artifact_dir = ROOT / "artifacts" / "pg235-failure-conditioned-policy-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_policy_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": PG235_SCHEMA, "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "action_classes": list(ACTION_CLASSES), "token_vocabulary": target_vocab, "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg235-failure-conditioned-policy-dataset-v1", "source_datasets": [str(PG233_DATASET.relative_to(ROOT)), str(PG234_CATALOG.relative_to(ROOT))], "records": all_records, "counts": {"base_records": len(base_records), "wire_records": len(wire_records), "unique_records": len(all_records), "duplicate_records": duplicate_count, "train_rows": len(train_rows), "unseen_family_holdout_rows": len(holdout_rows), "train_action_counts": dict(Counter(action_target(row) for row in train_rows)), "holdout_action_counts": dict(Counter(action_target(row) for row in holdout_rows)), "holdout_family_counts": dict(Counter(str(row.get("family_class")) for row in holdout_rows))}, "contract": {"unseen_family_holdout": True, "action_target_is_abstain_or_grounded_send": True, "typed_result_required_for_send_candidate": True, "next_token_loss_not_promotion_gate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    report = {"protocol_id": "pg-pk-235-failure-conditioned-policy-v1", "schema_version": PG235_SCHEMA, "status": "completed_failure_conditioned_abstention_training", "device": str(device), "counts": dataset["counts"], "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "variants": variants, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_after, "frozen_body_changed": body_before != body_after, "strict_unseen_family_abstain_pass": strict_pass, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"unseen_family_holdout_is_projection_based": True, "send_candidate_requires_typed_result": True, "large_body_is_frozen": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg235-failure-conditioned-policy-protocol-v1", "train_families": ["sql", "generic", "logic"], "holdout_families": ["dom", "redirect"], "action_head": list(ACTION_CLASSES), "send_candidate_requires_typed_result": True, "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg235-failure-conditioned-policy-trace-v1", "selected": selected, "variants": variants, "strict_unseen_family_abstain_pass": strict_pass, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    hold_metrics = selected["unseen_family_holdout"]
    MARKDOWN.write_text("\n".join(["# PG-235 failure-conditioned policy", "", f"device={device}; train={len(train_rows)}; unseen_family_holdout={len(holdout_rows)}", f"selected hidden={selected['hidden_dim']}; token={hold_metrics['next_token_accuracy']}; lane={hold_metrics['lane_accuracy']}; repair={hold_metrics['repair_accuracy']}; abstain_recall={hold_metrics['abstain_recall']}; false_send={hold_metrics['false_send_count']}; strict_pass={strict_pass}", "", "未见 DOM/redirect family 只能选择 abstain；只有 SQL typed+result 才允许 send_candidate 标签。next-token loss 不单独晋级。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "strict_pass": strict_pass, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

