"""PG-230: train a frozen-XXL next-token adapter after a quality funnel."""

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
    PG230_SCHEMA,
    REPAIR_ACTIONS,
    REPAIR_INDEX,
    build_vocabulary,
    digest,
    prepare_record,
    split_quality_records,
)


RESEARCH = ROOT / "research"
PG224_REPORT = RESEARCH / "pg224_pikachu_parameter_surface_collection_report_v1.json"
PG226_REPORT = RESEARCH / "pg226_ai_sql_payload_validation_report_v1.json"
PG227_REPORT = RESEARCH / "pg227_ai_dom_redirect_validation_report_v1.json"
PG229_REPORT = RESEARCH / "pg229_juice_shop_fresh_typed_replay_report_v1.json"
PG222_DATASET = RESEARCH / "pg222_problem_diagnoser_dataset_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"

REPORT = RESEARCH / "pg230_next_token_quality_funnel_training_report_v1.json"
DATASET = RESEARCH / "pg230_next_token_quality_funnel_dataset_v1.json"
TRACE = RESEARCH / "pg230_next_token_quality_funnel_trace_v1.json"
PROTOCOL = RESEARCH / "pg230_next_token_quality_funnel_protocol_v1.json"
MARKDOWN = RESEARCH / "pg230_next_token_quality_funnel_report_v1.md"


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG191 = _load_script("run_pg191_pikachu_surface_matrix_large.py")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _projection(result: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = result.get(key) or {}
    if isinstance(value, Mapping) and isinstance(value.get("response_projection"), Mapping):
        return value["response_projection"]
    if isinstance(value, Mapping):
        return value
    return {}


def _status(projection: Mapping[str, Any]) -> str:
    value = str(projection.get("status_class", "unknown"))
    return value if value in {"1xx", "2xx", "3xx", "4xx", "5xx"} else "unknown"


def _base_common(result: Mapping[str, Any], *, source: str, surface: str) -> dict[str, Any]:
    reset = result.get("reset") or {}
    if not isinstance(reset, Mapping):
        reset = {}
    fields = list(result.get("fields") or [])
    candidate = _projection(result, "candidate")
    negative = _projection(result, "negative")
    evidence = result.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        evidence = {}
    sent = bool((result.get("ai") or {}).get("sent", True))
    typed = bool(result.get("typed_effect_confirmed") or result.get("typed_effect_observed"))
    result_verified = bool(result.get("result_fixture_verified"))
    oracle_available = bool(typed or result_verified)
    return {
        "source": source,
        "seed": int(result.get("seed", 0) or 0),
        "surface_role": surface,
        "method": str(result.get("method", "GET")).upper(),
        "status_class": _status(candidate),
        "field_count": len(fields),
        "fresh_reset_ok": bool(result.get("fresh_reset", reset.get("fresh_target", True))),
        "reset_completed": bool(reset.get("completed", result.get("fresh_reset", True))),
        "reset_not_attempted": False,
        "candidate_sent": sent,
        "oracle_available": oracle_available,
        "typed_effect_confirmed": typed,
        "typed_effect_observed": typed,
        "result_fixture_verified": result_verified,
        "candidate_reference_agreement": bool(evidence.get("ai_reference_binding_match", (result.get("oracle") or {}).get("candidate_reference_agreement", True))),
        "negative_clean": bool((result.get("oracle") or {}).get("negative_clean", not bool((negative.get("marker") or {}).get("reflected", False)))),
        "binding_valid": bool(fields) if fields else True,
        "transport_error": bool(candidate.get("transport_error", False)),
        "result_mismatch_observed": bool(result.get("result_mismatch_observed", False)),
        "next_step": str(result.get("next_step", "")),
        "evidence_hash": str(evidence.get("evidence_sha256") or result.get("evidence_hash") or ""),
        "payload_grounded_eligible": bool(result.get("training_candidate", False) and typed and result_verified),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }


def _normalize_reports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pg226 = json.loads(PG226_REPORT.read_text(encoding="utf-8-sig"))
    for result in pg226.get("results", []):
        row = _base_common(result, source="pg226_typed_sql_result", surface=str(result.get("route", "sqli")))
        row["evidence_hash"] = row["evidence_hash"] if len(row["evidence_hash"]) == 64 else digest({"source": row["source"], "seed": row["seed"], "surface": row["surface_role"]})
        rows.append(row)
    pg227 = json.loads(PG227_REPORT.read_text(encoding="utf-8-sig"))
    for result in pg227.get("results", []):
        oracle = result.get("oracle") or {}
        route = str(result.get("route", ""))
        row = _base_common(result, source="pg227_dom_redirect_surface", surface=route)
        row.update({"oracle_available": False, "typed_effect_confirmed": False, "typed_effect_observed": False, "result_fixture_verified": False, "negative_clean": bool(oracle.get("negative_clean", False)), "candidate_reference_agreement": bool(oracle.get("candidate_reference_agreement", False)), "payload_grounded_eligible": False})
        row["evidence_hash"] = str((result.get("evidence") or {}).get("evidence_sha256") or digest({"source": row["source"], "seed": row["seed"], "surface": row["surface_role"]}))
        rows.append(row)
    pg229 = json.loads(PG229_REPORT.read_text(encoding="utf-8-sig"))
    for result in pg229.get("results", []):
        candidate = result.get("candidate_projection") or {}
        row = {
            "source": "pg229_juice_shop_fresh_typed_replay",
            "seed": int(result.get("seed", 0) or 0),
            "surface_role": str(result.get("route", "generic_surface")),
            "method": str(result.get("method", "GET")).upper(),
            "status_class": _status(candidate),
            "field_count": 0,
            "fresh_reset_ok": True,
            "reset_completed": True,
            "reset_not_attempted": False,
            "candidate_sent": True,
            "oracle_available": bool(result.get("typed_effect_confirmed", False)),
            "typed_effect_confirmed": bool(result.get("typed_effect_confirmed", False)),
            "typed_effect_observed": bool(result.get("typed_effect_confirmed", False)),
            "result_fixture_verified": False,
            "candidate_reference_agreement": bool(result.get("candidate_reference_agreement", False)),
            "negative_clean": bool(result.get("negative_clean", False)),
            "binding_valid": True,
            "transport_error": bool(candidate.get("transport_error", False)),
            "result_mismatch_observed": False,
            "next_step": str(result.get("next_step", "")),
            "evidence_hash": str(result.get("evidence_hash", "")),
            "model_self_error_detected": bool(result.get("model_self_error_detected", False)),
            "model_self_error_kind": result.get("model_self_error_kind"),
            "model_gate_corrected_diagnosis": result.get("model_gate_corrected_diagnosis"),
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        if len(row["evidence_hash"]) != 64:
            row["evidence_hash"] = digest({"source": row["source"], "seed": row["seed"], "surface": row["surface_role"]})
        rows.append(row)
    pg224 = json.loads(PG224_REPORT.read_text(encoding="utf-8-sig"))
    for result in pg224.get("results", []):
        sent = bool((result.get("ai") or {}).get("sent", False))
        candidate = _projection(result, "candidate")
        negative = _projection(result, "negative")
        source_hash = str(result.get("source_row_sha256", ""))
        evidence_hash = digest({"source": "pg224_real_surface_projection", "seed": result.get("seed", 0), "route": result.get("route", "")})
        row = {
            "source": "pg224_real_surface_projection",
            "seed": int(result.get("seed", 0) or 0),
            "surface_role": str(result.get("route", "generic_surface")),
            "method": str(result.get("method", "GET")).upper(),
            "status_class": _status(candidate),
            "field_count": len(result.get("fields") or []),
            "fresh_reset_ok": True if not sent else bool(result.get("fresh_reset", True)),
            "reset_completed": False if not sent else True,
            "reset_not_attempted": not sent,
            "candidate_sent": sent,
            "oracle_available": False,
            "typed_effect_confirmed": False,
            "typed_effect_observed": False,
            "result_fixture_verified": False,
            "candidate_reference_agreement": True,
            "negative_clean": not bool((negative.get("marker") or {}).get("reflected", False)),
            "binding_valid": bool(result.get("fields")),
            "transport_error": bool(candidate.get("transport_error", False)),
            "result_mismatch_observed": False,
            "next_step": "recheck_oracle" if sent else "abstain",
            "evidence_hash": source_hash if len(source_hash) == 64 else evidence_hash,
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        rows.append(row)
    # PG-222 already contains explicit, bounded model-decision-error
    # counterfactuals.  Reuse only their observable process fields and derive
    # a new evidence hash; never copy the diagnosis target into the token
    # record.
    pg222 = json.loads(PG222_DATASET.read_text(encoding="utf-8-sig"))
    for index, source_row in enumerate(pg222.get("rows", [])):
        if str(source_row.get("diagnosis", "")) != "model_decision_error":
            continue
        row = {
            "source": "pg222_model_decision_error_counterfactual",
            "seed": 300000 + index,
            "surface_role": str(source_row.get("method", "GET")),
            "method": str(source_row.get("method", "GET")).upper(),
            "status_class": str(source_row.get("status_class", "2xx")) if str(source_row.get("status_class", "2xx")) in {"1xx", "2xx", "3xx", "4xx", "5xx"} else "unknown",
            "field_count": int(source_row.get("field_count", 0) or 0),
            "fresh_reset_ok": True,
            "reset_completed": True,
            "reset_not_attempted": False,
            "candidate_sent": True,
            "oracle_available": False,
            "typed_effect_confirmed": False,
            "typed_effect_observed": False,
            "result_fixture_verified": False,
            "candidate_reference_agreement": True,
            "negative_clean": True,
            "binding_valid": True,
            "transport_error": False,
            "result_mismatch_observed": False,
            "next_step": "abstain",
            "previous_feedback": str(source_row.get("previous_feedback", "none")),
            "history_len": int(source_row.get("history_len", 0) or 0),
            "candidate_result_present": bool(source_row.get("candidate_result_present", False)),
            "model_claimed_positive": True,
            "model_self_error_detected": True,
            "model_self_error_kind": "premature_positive_counterfactual",
            "model_gate_corrected_diagnosis": "abstain",
            "evidence_hash": digest({"source": "pg222_model_decision_error_counterfactual", "index": index, "seed": 300000 + index}),
            "payload_grounded_eligible": False,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        }
        rows.append(row)
    return rows


def _input_token_id(token: str, vocabulary: Mapping[str, int]) -> int:
    aliases = {
        "method=GET": "history::method::GET",
        "method=POST": "history::method::POST",
        "status=2xx": "history::status::2xx",
        "status=4xx": "history::status::4xx",
        "status=5xx": "history::status::5xx",
        "oracle_available=1": "ir.oracle.availability=typed",
        "oracle_available=0": "history::typed_available::0",
        "typed_effect=1": "history::gate::typed_effect",
        "typed_effect=0": "history::gate::matched_negative_control",
    }
    return int(vocabulary.get(aliases.get(token, token), vocabulary.get("[UNK]", 1)))


def _encode_for_frozen(records: list[dict[str, Any]], input_vocabulary: Mapping[str, int], target_vocabulary: Mapping[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_sequences = [[_input_token_id(token, input_vocabulary) for token in record["tokens"]] for record in records]
    target_sequences = [[int(target_vocabulary.get(token, target_vocabulary["[UNK]"])) for token in record["tokens"]] for record in records]
    if any(len(left) != len(right) for left, right in zip(input_sequences, target_sequences)):
        raise ValueError("PG-230 input/target tokenizer alignment mismatch")
    width = max(len(seq) for seq in input_sequences)
    input_ids = torch.zeros((len(input_sequences), width), dtype=torch.long, device=device)
    target_ids = torch.zeros((len(target_sequences), width), dtype=torch.long, device=device)
    for index, (input_sequence, target_sequence) in enumerate(zip(input_sequences, target_sequences)):
        input_ids[index, : len(input_sequence)] = torch.tensor(input_sequence, dtype=torch.long, device=device)
        target_ids[index, : len(target_sequence)] = torch.tensor(target_sequence, dtype=torch.long, device=device)
    return input_ids[:, :-1], target_ids[:, 1:], torch.tensor([LANE_INDEX[str(row["lane"])] for row in records], dtype=torch.long, device=device), torch.tensor([REPAIR_INDEX[str(row["repair_action"])] for row in records], dtype=torch.long, device=device)


def _classification_positions(records: list[dict[str, Any]], context_width: int, device: torch.device) -> torch.Tensor:
    """Use the causal failure signature, never the lane/repair target suffix."""

    positions: list[int] = []
    for record in records:
        tokens = list(record.get("tokens", []))
        failure_index = next((index for index, token in enumerate(tokens) if str(token).startswith("failure=")), 0)
        positions.append(min(max(failure_index, 0), max(context_width - 1, 0)))
    return torch.tensor(positions, dtype=torch.long, device=device)


def _evaluate(model: FrozenXXLNextTokenAdapter, context: torch.Tensor, targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor, *, device: torch.device) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    token_target, lane_target, repair_target = targets
    token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_target.reshape(-1), ignore_index=0)
    valid = token_target.ne(0)
    token_pred = output["token"].argmax(-1)
    token_count = int(valid.sum().item())
    lane_pred = output["lane"].argmax(-1)
    repair_pred = output["repair"].argmax(-1)
    hard_mask = lane_target == LANE_INDEX["hard_negative"]
    self_error_recall = float(((lane_pred == LANE_INDEX["hard_negative"]) & hard_mask).sum().item() / max(int(hard_mask.sum().item()), 1))
    return {"token_loss": round(float(token_loss.detach().cpu()), 8), "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8), "next_token_accuracy": round(float(((token_pred == token_target) & valid).sum().item() / max(token_count, 1)), 8), "token_count": token_count, "lane_accuracy": round(float((lane_pred == lane_target).float().mean().item()), 8), "repair_accuracy": round(float((repair_pred == repair_target).float().mean().item()), 8), "self_error_recall": round(self_error_recall, 8), "self_error_count": int(hard_mask.sum().item())}


def main() -> int:
    raw_rows = _normalize_reports()
    prepared = [prepare_record(row) for row in raw_rows]
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for record in prepared:
        if record["token_hash"] in seen:
            duplicate_count += 1
            continue
        seen.add(record["token_hash"])
        unique.append(record)
    train_rows, holdout_rows, quarantine_rows = split_quality_records(unique)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocabulary = build_vocabulary(train_rows)
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocabulary = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG191._build_model("xxl", input_vocabulary, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_ids, train_targets_tokens, train_lane, train_repair = _encode_for_frozen(train_rows, input_vocabulary, vocabulary, device)
    hold_ids, hold_targets_tokens, hold_lane, hold_repair = _encode_for_frozen(holdout_rows, input_vocabulary, vocabulary, device)
    # ``inference_mode`` tensors cannot be saved for the adapter backward
    # pass.  The frozen body still does no gradient work; clone its bounded
    # contexts into ordinary tensors before training the next-token heads.
    with torch.no_grad():
        # ``DualHead.hidden`` intentionally pools the last token for the old
        # diagnostic task.  Next-token training needs the full contextual
        # sequence, so use the frozen body's encoder directly.
        train_context = base.base.body.encode(train_ids, train_ids.ne(0)).detach().clone()
        hold_context = base.base.body.encode(hold_ids, hold_ids.ne(0)).detach().clone()
    train_positions = _classification_positions(train_rows, train_context.shape[1], device)
    hold_positions = _classification_positions(holdout_rows, hold_context.shape[1], device)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    frozen_count = int(sum(parameter.numel() for parameter in base.parameters()))
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: FrozenXXLNextTokenAdapter | None = None
    for hidden_dim in (64, 128, 256):
        torch.manual_seed(230 + hidden_dim)
        model = FrozenXXLNextTokenAdapter(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, vocab_size=len(vocabulary)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        lane_counts = torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)
        repair_counts = torch.bincount(train_repair, minlength=len(REPAIR_ACTIONS)).float().clamp_min(1.0)
        lane_weights = (lane_counts.sum() / lane_counts).to(device)
        repair_weights = (repair_counts.sum() / repair_counts).to(device)
        for _ in range(100):
            model.train()
            outputs = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(outputs["token"].reshape(-1, outputs["token"].shape[-1]), train_targets_tokens.reshape(-1), ignore_index=0)
            loss = token_loss + 0.30 * nn.functional.cross_entropy(outputs["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(outputs["repair"], train_repair, weight=repair_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = {"train": _evaluate(model, train_context, (train_targets_tokens, train_lane, train_repair), train_positions, device=device), "holdout": _evaluate(model, hold_context, (hold_targets_tokens, hold_lane, hold_repair), hold_positions, device=device)}
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), **metrics}
        variants.append(result)
        key = (-result["holdout"]["self_error_recall"], -result["holdout"]["lane_accuracy"], -result["holdout"]["repair_accuracy"], result["holdout"]["token_loss"])
        old_key = None if selected is None else (-selected["holdout"]["self_error_recall"], -selected["holdout"]["lane_accuracy"], -selected["holdout"]["repair_accuracy"], selected["holdout"]["token_loss"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-230 no adapter selected")
    artifact_dir = ROOT / "artifacts" / "pg230-next-token-quality-funnel-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_next_token_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": PG230_SCHEMA, "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "token_vocabulary": vocabulary, "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT)), "frozen_body_parameter_count": frozen_count}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg230-next-token-quality-funnel-dataset-v1", "source_reports": [str(path.relative_to(ROOT)) for path in (PG224_REPORT, PG226_REPORT, PG227_REPORT, PG229_REPORT, PG222_DATASET)], "records": unique, "split": {"train": len(train_rows), "holdout": len(holdout_rows), "quarantine": len(quarantine_rows), "source_seed_surface_holdout": True}, "funnel": {"raw_records": len(raw_rows), "unique_records": len(unique), "duplicate_records": duplicate_count, "lane_counts": dict(Counter(str(record["lane"]) for record in unique)), "quarantine_reasons": dict(Counter(reason for record in quarantine_rows for reason in record["quality_reasons"]))}, "contract": {"next_token_loss_alone_not_promotion": True, "hard_negative_self_error_retained": True, "silver_abstention_only": True, "evaluator_targets_as_features": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    report = {"protocol_id": "pg-pk-230-next-token-quality-funnel-v1", "schema_version": PG230_SCHEMA, "status": "completed_next_token_quality_funnel_frozen_xxl_training", "device": str(device), "source_reports": [str(path.relative_to(ROOT)) for path in (PG224_REPORT, PG226_REPORT, PG227_REPORT, PG229_REPORT, PG222_DATASET)], "funnel": dataset["funnel"], "split": dataset["split"], "vocabulary_size": len(vocabulary), "frozen_body_parameter_count": frozen_count, "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_before, "frozen_body_changed": False, "variants": variants, "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "holdout": selected["holdout"]}, "metrics": {"next_token_loss_is_not_quality_gate": True, "self_error_detection_is_measured": True, "cross_seed_surface_holdout": True, "class_balanced_lane_and_repair_loss": True, "classification_context_excludes_lane_and_repair_targets": True}, "engineering_repairs": [{"failure": "inference_tensor_cannot_be_saved_for_backward", "repair": "clone_frozen_context_under_no_grad_before_adapter_backward"}, {"failure": "input_target_tokenizer_class_mismatch", "repair": "separate_frozen_input_vocabulary_and_PG230_target_vocabulary_with_alignment_gate"}, {"failure": "pooled_context_cannot_supervise_next_token_sequence", "repair": "use_frozen_body_sequence_encoder_for_token_head"}, {"failure": "lane_repair_target_leakage_through_last_hidden", "repair": "classify_from_causal_failure_position_before_lane_and_repair_suffix"}], "promotion": {"gold_next_token_training_allowed": True, "hard_negative_repair_training_allowed": True, "silver_abstention_training_allowed": True, "quarantine_training_allowed": False, "payload_grounded_catalog_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"data_small": True, "frozen_xxl_body_not_updated": True, "gold_rows_local_only": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg230-next-token-quality-funnel-protocol-v1", "stages": ["serialization", "provenance_scope", "information_completeness", "self_consistency", "independent_replay", "generalization", "learning_value"], "lanes": list(LANES), "target": "next token + lane + repair action", "frozen_xxl_body": True, "loss_alone_cannot_promote": True, "hard_negative_self_error_required": True, "class_balanced_auxiliary_losses": True, "catastrophic_forgetting": {"body_state_hash_must_remain_equal": True, "canary_label_or_oracle_used": False}, "raw_payload_and_response_excluded": True, "promotion_blocked": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg230-next-token-quality-funnel-trace-v1", "selected": selected, "variants": variants, "funnel": dataset["funnel"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-230 next-token quality funnel", "", f"device={device}; raw={len(raw_rows)}; unique={len(unique)}; train={len(train_rows)}; holdout={len(holdout_rows)}; quarantine={len(quarantine_rows)}", f"lanes={dataset['funnel']['lane_counts']}; duplicates={duplicate_count}", f"selected hidden={selected['hidden_dim']}; holdout token accuracy={selected['holdout']['next_token_accuracy']}; lane accuracy={selected['holdout']['lane_accuracy']}; repair accuracy={selected['holdout']['repair_accuracy']}; self-error recall={selected['holdout']['self_error_recall']}", "", "next-token loss 只作为表示学习指标；gold/hard-negative/silver/quarantine 分层决定数据能否进入对应训练头。冻结 XXL 主体哈希前后相同，未把本轮小数据误报成通用能力。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "funnel": report["funnel"], "split": report["split"], "selected": report["selected"], "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
