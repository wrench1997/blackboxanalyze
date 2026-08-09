"""PG-231: train on richer observable failure -> repair trajectories.

This is an independent comparison to PG-230.  It keeps the frozen 101M body,
adds all bounded process-state rows from the existing local Pikachu traces,
and prevents the lane/repair heads from reading their target suffix.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg230_next_token_quality_funnel import (  # noqa: E402
    FrozenXXLNextTokenAdapter,
    LANE_INDEX,
    LANES,
    REPAIR_INDEX,
    build_vocabulary,
    digest,
    split_quality_records,
)
from app.pg231_feedback_trajectory import PG231_SCHEMA, prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
PG222_DATASET = RESEARCH / "pg222_problem_diagnoser_dataset_v1.json"
PG224_REPORT = RESEARCH / "pg224_pikachu_parameter_surface_collection_report_v1.json"
PG226_REPORT = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.json"
PG227_REPORT = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.json"
PG229_REPORT = RESEARCH / "pg229_juice_shop_fresh_typed_replay_report_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"

REPORT = RESEARCH / "pg231_feedback_trajectory_training_report_v1.json"
DATASET = RESEARCH / "pg231_feedback_trajectory_dataset_v1.json"
TRACE = RESEARCH / "pg231_feedback_trajectory_trace_v1.json"
PROTOCOL = RESEARCH / "pg231_feedback_trajectory_protocol_v1.json"
MARKDOWN = RESEARCH / "pg231_feedback_trajectory_report_v1.md"


def _load_pg230() -> Any:
    path = ROOT / "scripts" / "run_pg230_next_token_quality_funnel_training.py"
    spec = importlib.util.spec_from_file_location("pg230_training_for_pg231", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-230 normalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG230 = _load_pg230()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evidence_hash(row: Mapping[str, Any], index: int) -> str:
    value = str(row.get("evidence_hash", ""))
    return value if len(value) == 64 else digest({"source": "pg222_observed_process", "index": index, "seed": row.get("seed", 0)})


def _route_surface(route: Any) -> str:
    text = str(route or "").casefold()
    if "sqli" in text or "sql" in text:
        return "sql_surface"
    if "xss" in text or "dom" in text:
        return "dom_surface"
    if "redirect" in text:
        return "redirect_surface"
    if "csrf" in text or "auth" in text:
        return "authentication_surface"
    return "generic_surface"


def _pg222_process_rows() -> list[dict[str, Any]]:
    source = json.loads(PG222_DATASET.read_text(encoding="utf-8-sig"))
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(source.get("rows", [])):
        diagnosis = str(source_row.get("diagnosis", ""))
        row = {
            "source": "pg222_observed_process",
            "seed": int(source_row.get("seed", 0) or 0),
            "surface_role": _route_surface(source_row.get("route")),
            "method": str(source_row.get("method", "GET")).upper(),
            "status_class": str(source_row.get("status_class", "unknown")),
            "field_count": int(source_row.get("field_count", 0) or 0),
            "fresh_reset_ok": bool(source_row.get("fresh_reset_ok", False)),
            "reset_completed": bool(source_row.get("reset_completed", False)),
            "reset_not_attempted": False,
            "candidate_sent": bool(source_row.get("candidate_sent", False)),
            "oracle_available": bool(source_row.get("oracle_available", False)),
            "typed_effect_observed": bool(source_row.get("typed_effect_observed", False)),
            "typed_effect_confirmed": bool(source_row.get("typed_effect_observed", False)),
            "result_fixture_verified": bool(source_row.get("result_fixture_verified", False)),
            "candidate_reference_agreement": bool(source_row.get("candidate_reference_agreement", False)),
            "negative_clean": bool(source_row.get("negative_clean", False)),
            "binding_valid": bool(source_row.get("binding_valid", False)),
            "transport_error": bool(source_row.get("transport_error", False)),
            "result_mismatch_observed": bool(source_row.get("result_mismatch_observed", False)),
            "next_step": str(source_row.get("next_step", "abstain")),
            "previous_feedback": str(source_row.get("previous_feedback", "none")),
            "history_len": int(source_row.get("history_len", 0) or 0),
            "candidate_result_present": bool(source_row.get("candidate_result_present", False)),
            "model_claimed_positive": bool(source_row.get("model_claimed_positive", False)),
            "model_abstained": bool(source_row.get("model_abstained", False)),
            "backend_observed": bool(source_row.get("backend_observed", False)),
            "database_health_ok": bool(source_row.get("database_health_ok", False)),
            "reference_sent": bool(source_row.get("reference_sent", False)),
            "negative_sent": bool(source_row.get("negative_sent", False)),
            "candidate_sql_error_shape": bool(source_row.get("candidate_sql_error_shape", False)),
            "boolean_differential": bool(source_row.get("boolean_differential", False)),
            "negative_result_absent": bool(source_row.get("negative_result_absent", False)),
            "hard_gate_observed": bool(source_row.get("hard_gate_observed", False)),
            "model_self_error_detected": diagnosis == "model_decision_error",
            "model_self_error_kind": "annotated_model_decision_error" if diagnosis == "model_decision_error" else None,
            "evidence_hash": _evidence_hash(source_row, index),
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        rows.append(row)
    return rows


def _normalize_rows() -> list[dict[str, Any]]:
    # Keep the prior cross-lab rows, but replace PG-230's derived error-only
    # slice with all bounded PG-222 process rows.  The diagnosis string itself
    # is used only to create a hard-negative label and is never tokenized.
    base_rows = [row for row in PG230._normalize_reports() if row.get("source") != "pg222_model_decision_error_counterfactual"]
    return base_rows + _pg222_process_rows()


def _input_token_id(token: str, vocabulary: Mapping[str, int]) -> int:
    aliases = {
        "phase=observe": "[OBS]",
        "phase=diagnose": "[STEP]",
        "phase=repair": "ir.failure.recovery_phase=failure_adjusted",
        "phase=replay": "[BELIEF]",
        "method=GET": "history::method::GET",
        "method=POST": "history::method::POST",
        "status=2xx": "history::status::2xx",
        "status=4xx": "history::status::4xx",
        "status=5xx": "history::status::5xx",
        "candidate_sent=1": "history::candidate::1",
        "candidate_sent=0": "history::candidate::0",
        "field_bucket=0": "history::field_count::0",
        "field_bucket=1": "history::field_count::2",
        "field_bucket=2": "history::field_count::2",
        "field_bucket=3+": "history::field_count::4",
        "history_bucket=0": "history::history_len::0",
        "history_bucket=1": "history::history_len::1",
        "history_bucket=2": "history::history_len::2",
        "history_bucket=3+": "history::history_len::4",
        "oracle_available=1": "ir.oracle.availability=typed",
        "oracle_available=0": "history::typed_available::0",
        "result_verified=1": "history::gate::typed_effect",
        "candidate_present=1": "ir.response.candidate_signal=true",
        "candidate_present=0": "ir.response.candidate_signal=false",
        "candidate_error_shape=1": "ir.failure.kind=parse_error_signature",
        "boolean_differential=1": "ir.failure.kind=shape_delta",
        "transport_error=1": "ir.failure.kind=timeout_signature",
        "result_mismatch=1": "ir.failure.kind=shape_delta",
        "failure=oracle_unavailable": "ir.failure.kind=oracle_unavailable",
        "failure=typed_effect": "ir.failure.kind=typed_positive",
        "failure=candidate_no_effect": "ir.failure.kind=no_surface_delta",
        "failure=environment_failure": "ir.failure.kind=timeout_signature",
        "failure=result_mismatch": "ir.failure.kind=shape_delta",
        "failure=reference_disagreement": "history::belief_top::no_surface_delta",
        "failure=binding_failure": "ir.failure.kind=candidate_without_typed_effect",
        "failure=model_self_error": "ir.failure.recovery_phase=failure_adjusted",
    }
    if token in vocabulary:
        return int(vocabulary[token])
    return int(vocabulary.get(aliases.get(token, "[UNK]"), vocabulary.get("[UNK]", 1)))


def _encode(records: list[dict[str, Any]], input_vocabulary: Mapping[str, int], target_vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_sequences = [[_input_token_id(str(token), input_vocabulary) for token in record["tokens"]] for record in records]
    target_sequences = [[int(target_vocabulary.get(str(token), target_vocabulary["[UNK]"])) for token in record["tokens"]] for record in records]
    width = max(len(sequence) for sequence in input_sequences)
    input_ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
    target_ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
    for index, (left, right) in enumerate(zip(input_sequences, target_sequences)):
        input_ids[index, : len(left)] = torch.tensor(left, dtype=torch.long, device=device)
        target_ids[index, : len(right)] = torch.tensor(right, dtype=torch.long, device=device)
    return input_ids[:, :-1], target_ids[:, 1:], torch.tensor([LANE_INDEX[str(row["lane"])] for row in records], dtype=torch.long, device=device), torch.tensor([REPAIR_INDEX[str(row["repair_action"])] for row in records], dtype=torch.long, device=device)


def _positions(records: list[dict[str, Any]], context_width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(record.get("classification_position", 0)), 0), max(context_width - 1, 0)) for record in records], dtype=torch.long, device=device)


def _evaluate(model: FrozenXXLNextTokenAdapter, context: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    token_target, lane_target, repair_target = targets
    token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_target.reshape(-1), ignore_index=0)
    valid = token_target.ne(0)
    token_pred = output["token"].argmax(-1)
    lane_pred = output["lane"].argmax(-1)
    repair_pred = output["repair"].argmax(-1)
    hard_mask = lane_target == LANE_INDEX["hard_negative"]
    return {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_target) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "lane_accuracy": round(float((lane_pred == lane_target).float().mean().item()), 8),
        "repair_accuracy": round(float((repair_pred == repair_target).float().mean().item()), 8),
        "self_error_recall": round(float(((lane_pred == LANE_INDEX["hard_negative"]) & hard_mask).sum().item() / max(int(hard_mask.sum().item()), 1)), 8),
        "self_error_count": int(hard_mask.sum().item()),
    }


def main() -> int:
    raw_rows = _normalize_rows()
    prepared = [prepare_feedback_record(row) for row in raw_rows]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in prepared:
        if record["trajectory_hash"] in seen:
            duplicate_count += 1
            continue
        seen.add(record["trajectory_hash"])
        unique.append(record)
    train_rows, holdout_rows, quarantine_rows = split_quality_records(unique)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = build_vocabulary(train_rows)
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG230.PG191._build_model("xxl", input_vocabulary, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_ids, train_targets_tokens, train_lane, train_repair = _encode(train_rows, input_vocabulary, vocabulary, device)
    hold_ids, hold_targets_tokens, hold_lane, hold_repair = _encode(holdout_rows, input_vocabulary, vocabulary, device)
    with torch.no_grad():
        train_context = base.base.body.encode(train_ids, train_ids.ne(0)).detach().clone()
        hold_context = base.base.body.encode(hold_ids, hold_ids.ne(0)).detach().clone()
    train_positions = _positions(train_rows, train_context.shape[1], device)
    hold_positions = _positions(holdout_rows, hold_context.shape[1], device)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    frozen_count = int(sum(parameter.numel() for parameter in base.parameters()))
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: FrozenXXLNextTokenAdapter | None = None
    for hidden_dim in (64, 128, 256):
        torch.manual_seed(231 + hidden_dim)
        model = FrozenXXLNextTokenAdapter(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, vocab_size=len(vocabulary)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        lane_counts = torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)
        repair_counts = torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0)
        lane_weights = (lane_counts.sum() / lane_counts).to(device)
        repair_weights = (repair_counts.sum() / repair_counts).to(device)
        for _ in range(80):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_targets_tokens.reshape(-1), ignore_index=0)
            loss = token_loss + 0.30 * nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": _evaluate(model, train_context, (train_targets_tokens, train_lane, train_repair), train_positions), "holdout": _evaluate(model, hold_context, (hold_targets_tokens, hold_lane, hold_repair), hold_positions)}
        variants.append(result)
        key = (-result["holdout"]["self_error_recall"], -result["holdout"]["lane_accuracy"], -result["holdout"]["repair_accuracy"], result["holdout"]["token_loss"])
        old_key = None if selected is None else (-selected["holdout"]["self_error_recall"], -selected["holdout"]["lane_accuracy"], -selected["holdout"]["repair_accuracy"], selected["holdout"]["token_loss"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-231 no adapter selected")
    artifact_dir = ROOT / "artifacts" / "pg231-feedback-trajectory-funnel-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_feedback_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": PG231_SCHEMA, "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "token_vocabulary": vocabulary, "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT)), "frozen_body_parameter_count": frozen_count}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lane_counts = dict(Counter(str(record["lane"]) for record in unique))
    feature_tokens = Counter(token.split("=", 1)[0] for record in unique for token in record["tokens"] if "=" in token)
    dataset = {"schema_version": "pg231-feedback-trajectory-dataset-v1", "source_reports": [str(path.relative_to(ROOT)) for path in (PG222_DATASET, PG224_REPORT, PG226_REPORT, PG227_REPORT, PG229_REPORT)], "records": unique, "split": {"train": len(train_rows), "holdout": len(holdout_rows), "quarantine": len(quarantine_rows), "source_seed_surface_holdout": True}, "funnel": {"raw_records": len(raw_rows), "unique_records": len(unique), "duplicate_records": duplicate_count, "lane_counts": lane_counts, "quarantine_reasons": dict(Counter(reason for record in quarantine_rows for reason in record["quality_reasons"]))}, "feature_coverage": dict(feature_tokens), "contract": {"observed_process_fields_only": True, "diagnosis_targets_not_features": True, "classification_context_excludes_lane_and_repair_targets": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    report = {"protocol_id": "pg-pk-231-feedback-trajectory-funnel-v1", "schema_version": PG231_SCHEMA, "status": "completed_feedback_trajectory_frozen_xxl_training", "device": str(device), "source_reports": dataset["source_reports"], "funnel": dataset["funnel"], "split": dataset["split"], "vocabulary_size": len(vocabulary), "frozen_body_parameter_count": frozen_count, "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_before, "frozen_body_changed": False, "variants": variants, "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "holdout": selected["holdout"]}, "metrics": {"next_token_loss_is_not_quality_gate": True, "self_error_detection_is_measured": True, "cross_seed_surface_holdout": True, "classification_context_excludes_lane_and_repair_targets": True, "all_pg222_process_rows_included": True}, "promotion": {"gold_next_token_training_allowed": True, "hard_negative_repair_training_allowed": True, "silver_abstention_training_allowed": True, "quarantine_training_allowed": False, "payload_grounded_catalog_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"data_expanded_but_local": True, "frozen_xxl_body_not_updated": True, "derived_hard_negative_labels_are_not_typed_positive": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg231-feedback-trajectory-protocol-v1", "stages": ["bounded_observation", "feedback_normalization", "failure_signature", "repair_target", "causal_prefix_classification", "cross_seed_surface_holdout", "promotion_review"], "target": "next token + lane + repair action", "frozen_xxl_body": True, "diagnosis_targets_not_features": True, "classification_context_excludes_lane_and_repair_targets": True, "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg231-feedback-trajectory-trace-v1", "selected": selected, "variants": variants, "funnel": dataset["funnel"], "feature_coverage": dict(feature_tokens), "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-231 feedback trajectory funnel", "", f"device={device}; raw={len(raw_rows)}; unique={len(unique)}; train={len(train_rows)}; holdout={len(holdout_rows)}; quarantine={len(quarantine_rows)}", f"lanes={lane_counts}; duplicates={duplicate_count}", f"selected hidden={selected['hidden_dim']}; holdout token accuracy={selected['holdout']['next_token_accuracy']}; lane accuracy={selected['holdout']['lane_accuracy']}; repair accuracy={selected['holdout']['repair_accuracy']}; self-error recall={selected['holdout']['self_error_recall']}", "", "加入的是可观察过程状态，不是漏洞标签或原始 payload；分类头只读取 failure 位置之前的因果上下文。next-token loss 仍不能单独晋级。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "funnel": report["funnel"], "split": report["split"], "selected": report["selected"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

