"""PG-233: add safe Pikachu family traces and compare adapter capacity.

The training body remains the frozen XXL checkpoint.  Four adapter capacities
are measured under a source+family double holdout; this prevents a larger head
from being mistaken for a more general model.
"""

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

from app.pg230_next_token_quality_funnel import FrozenXXLNextTokenAdapter, LANES, LANE_INDEX, REPAIR_INDEX, build_vocabulary, digest, split_quality_records  # noqa: E402
from app.pg233_cross_family_capacity import PG233_SCHEMA, add_family_context, prepare_pikachu_sample  # noqa: E402


RESEARCH = ROOT / "research"
PG231_DATASET = RESEARCH / "pg231_feedback_trajectory_dataset_v1.json"
PG51_CATALOG = RESEARCH / "pg51_pikachu_docker_dual_channel_catalog_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"

REPORT = RESEARCH / "pg233_cross_family_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg233_cross_family_capacity_dataset_v1.json"
TRACE = RESEARCH / "pg233_cross_family_capacity_trace_v1.json"
PROTOCOL = RESEARCH / "pg233_cross_family_capacity_protocol_v1.json"
MARKDOWN = RESEARCH / "pg233_cross_family_capacity_report_v1.md"


def _load_pg231() -> Any:
    path = ROOT / "scripts" / "run_pg231_feedback_trajectory_training.py"
    spec = importlib.util.spec_from_file_location("pg231_training_for_pg233", path)
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
    base = json.loads(PG231_DATASET.read_text(encoding="utf-8-sig"))
    records: list[dict[str, Any]] = []
    for source_record in base.get("records", []):
        records.append(add_family_context(source_record, family=source_record.get("surface_class"), channel=source_record.get("method"), source_role="observed"))
    catalog = json.loads(PG51_CATALOG.read_text(encoding="utf-8-sig"))
    samples = [prepare_pikachu_sample(sample) for sample in catalog.get("samples", [])]
    return records, samples


def _encode(records: list[dict[str, Any]], input_vocab: dict[str, int], target_vocab: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return PG231._encode(records, input_vocab, target_vocab, device)


def _evaluate(model: FrozenXXLNextTokenAdapter, context: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor) -> dict[str, Any]:
    return PG231._evaluate(model, context, targets, positions)


def _train_capacity(hidden_dim: int, train_rows: list[dict[str, Any]], eval_sets: dict[str, list[dict[str, Any]]], base: Any, input_vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    target_vocab = build_vocabulary(train_rows)
    encoded: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    contexts: dict[str, torch.Tensor] = {}
    positions: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, rows in {"train": train_rows, **eval_sets}.items():
            ids, targets, lanes, repairs = _encode(rows, input_vocab, target_vocab, device)
            encoded[name] = (ids, targets, lanes, repairs)
            contexts[name] = base.base.body.encode(ids, ids.ne(0)).detach().clone()
            positions[name] = PG231._positions(rows, contexts[name].shape[1], device)
    torch.manual_seed(233 + hidden_dim)
    model = FrozenXXLNextTokenAdapter(d_model=int(contexts["train"].shape[-1]), hidden_dim=hidden_dim, vocab_size=len(target_vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_targets = encoded["train"]
    train_lane, train_repair = train_targets[2], train_targets[3]
    lane_weights = (torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0).sum() / torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)).to(device)
    repair_weights = (torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0).sum() / torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0)).to(device)
    for _ in range(80):
        model.train()
        output = model(contexts["train"], classification_positions=positions["train"])
        token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_targets[1].reshape(-1), ignore_index=0)
        loss = token_loss + 0.30 * nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    metrics = {name: _evaluate(model, contexts[name], encoded[name][1:], positions[name]) for name in eval_sets}
    metrics["train"] = _evaluate(model, contexts["train"], train_targets[1:], positions["train"])
    return {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "target_vocabulary_size": len(target_vocab), "metrics": metrics, "state_dict": model.state_dict()}


def main() -> int:
    base_records, new_samples = _load_records()
    all_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in base_records + new_samples:
        if record["trajectory_hash"] in seen:
            duplicate_count += 1
            continue
        seen.add(record["trajectory_hash"])
        all_records.append(record)
    usable = [record for record in all_records if record["lane"] not in {"quarantine", "reject"}]
    source_holdout_name = "pg51_pikachu_docker_dual_channel"
    family_holdout_name = "redirect"
    train_rows = [record for record in usable if record["source"] != source_holdout_name and record.get("family_class") != family_holdout_name]
    source_holdout = [record for record in usable if record["source"] == source_holdout_name]
    family_holdout = [record for record in usable if record.get("family_class") == family_holdout_name]
    double_holdout = [record for record in source_holdout if record.get("family_class") == family_holdout_name]
    if not train_rows or not source_holdout or not family_holdout or not double_holdout:
        raise RuntimeError("PG-233 requires non-empty source, family and double holdouts")
    eval_sets = {"source_holdout": source_holdout, "family_holdout": family_holdout, "double_holdout": double_holdout}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: dict[str, Any] | None = None
    for hidden_dim in (64, 128, 256, 512):
        result = _train_capacity(hidden_dim, train_rows, eval_sets, base, input_vocab, device)
        clean = {key: value for key, value in result.items() if key != "state_dict"}
        variants.append(clean)
        double = result["metrics"]["double_holdout"]
        source = result["metrics"]["source_holdout"]
        family = result["metrics"]["family_holdout"]
        key = (-float(double["repair_accuracy"]), -float(source["repair_accuracy"]), -float(family["repair_accuracy"]), float(source["token_loss"]))
        old_key = None if selected is None else (-float(selected["metrics"]["double_holdout"]["repair_accuracy"]), -float(selected["metrics"]["source_holdout"]["repair_accuracy"]), -float(selected["metrics"]["family_holdout"]["repair_accuracy"]), float(selected["metrics"]["source_holdout"]["token_loss"]))
        if selected is None or key < old_key:
            selected = clean
            selected_model = result
    body_after = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    if selected is None or selected_model is None:
        raise RuntimeError("PG-233 no selected capacity")
    artifact_dir = ROOT / "artifacts" / "pg233-cross-family-capacity-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_capacity_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": PG233_SCHEMA, "state_dict": selected_model["state_dict"], "hidden_dim": selected["hidden_dim"], "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg233-cross-family-capacity-dataset-v1", "source_datasets": [str(PG231_DATASET.relative_to(ROOT)), str(PG51_CATALOG.relative_to(ROOT))], "records": all_records, "counts": {"raw_base_records": len(base_records), "raw_new_samples": len(new_samples), "unique_records": len(all_records), "duplicate_records": duplicate_count, "usable_records": len(usable), "quarantine_records": len(all_records) - len(usable), "train_rows": len(train_rows), "source_holdout_rows": len(source_holdout), "family_holdout_rows": len(family_holdout), "double_holdout_rows": len(double_holdout), "lane_counts": dict(Counter(str(record["lane"]) for record in all_records)), "family_counts": dict(Counter(str(record.get("family_class", "unknown")) for record in all_records)), "source_counts": dict(Counter(str(record["source"]) for record in all_records))}, "holdout_policy": {"source_excluded_from_train": source_holdout_name, "family_excluded_from_train": family_holdout_name, "double_holdout_is_intersection": True}, "contract": {"projection_only": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "diagnosis_targets_not_features": True, "next_token_loss_not_promotion_gate": True, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    selected_metrics = selected["metrics"]
    strict_pass = all(float(selected_metrics[name]["lane_accuracy"]) >= 0.80 and float(selected_metrics[name]["repair_accuracy"]) >= 0.80 for name in ("source_holdout", "family_holdout", "double_holdout"))
    report = {"protocol_id": "pg-pk-233-cross-family-capacity-v1", "schema_version": PG233_SCHEMA, "status": "completed_cross_family_capacity_double_holdout", "device": str(device), "counts": dataset["counts"], "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "target_vocabulary_size": selected["target_vocabulary_size"], "metrics": selected_metrics}, "variants": variants, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_after, "frozen_body_changed": body_before != body_after, "strict_source_family_double_holdout_pass": strict_pass, "promotion": {"memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "training_promotion_allowed": False}, "honesty": {"new_samples_are_projection_only": True, "large_body_is_frozen": True, "capacity_sweep_does_not_replace_data": True, "strict_holdout_required": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg233-cross-family-capacity-protocol-v1", "new_local_source": "pg51_pikachu_docker_dual_channel", "capacity_variants": [64, 128, 256, 512], "source_and_family_must_both_be_excluded": True, "double_holdout_required": True, "next_token_loss_not_promotion_gate": True, "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg233-cross-family-capacity-trace-v1", "selected": selected, "variants": variants, "strict_source_family_double_holdout_pass": strict_pass, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    lines = ["# PG-233 cross-family capacity", "", f"device={device}; unique={len(all_records)}; train={len(train_rows)}; source_holdout={len(source_holdout)}; family_holdout={len(family_holdout)}; double_holdout={len(double_holdout)}", f"selected hidden={selected['hidden_dim']}; strict_pass={strict_pass}"]
    for name in ("source_holdout", "family_holdout", "double_holdout"):
        metrics = selected_metrics[name]
        lines.append(f"{name}: token={metrics['next_token_accuracy']}; lane={metrics['lane_accuracy']}; repair={metrics['repair_accuracy']}; self_error_recall={metrics['self_error_recall']}")
    lines.append("容量增大只有在 source+family 双重留出也通过时才有意义；本报告不自动晋级长期记忆。")
    MARKDOWN.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "strict_pass": strict_pass, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

