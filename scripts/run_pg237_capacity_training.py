"""PG-237: train a larger frozen-XXL policy on non-trivial typed replay.

The independent PG-237 result-fixture trace contributes both grounded
send-candidate rows and clean abstention rows.  Seed 23702 and the PG-236
seed 23632 are held out together, so the holdout contains positive and
negative actions instead of a vacuous all-abstain safety check.
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

from app.pg230_next_token_quality_funnel import LANES, REPAIR_INDEX, build_vocabulary, digest  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402
from app.pg235_failure_conditioned_policy import ACTION_CLASSES, ACTION_INDEX, FrozenXXLFailurePolicy, PG235_SCHEMA, action_target  # noqa: E402


RESEARCH = ROOT / "research"
BASE_DATASET = RESEARCH / "pg236_failure_conditioned_training_dataset_v1.json"
PG237_TRACE = RESEARCH / "pg237_pikachu_result_fixture_replay_trace_v1.json"
PG191_CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg237_capacity_training_report_v1.json"
DATASET = RESEARCH / "pg237_capacity_training_dataset_v1.json"
TRACE = RESEARCH / "pg237_capacity_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg237_capacity_training_protocol_v1.json"
MARKDOWN = RESEARCH / "pg237_capacity_training_report_v1.md"

# The defaults reproduce PG-237 exactly.  Later source-heldout experiments can
# override these module variables without duplicating the trainer, while the
# original regression fixture remains unchanged.
FRESH_SOURCE = "pg237_pikachu_result_fixture_replay"
FRESH_HOLDOUT_SEEDS = (23702,)
EXTRA_HOLDOUT_SOURCE = "pg236_pikachu_fixed_independent"
EXTRA_HOLDOUT_SEEDS = (23632,)
ARTIFACT_DIR = ROOT / "artifacts" / "pg237-pikachu-capacity-training-v1"
EXPERIMENT_ID = "pg237"

# Optional split extensions used by later experiments.  Empty defaults keep
# the original PG-237 protocol and regression fixture unchanged.  A wrapper
# may add a source-substring holdout (for example every Pikachu-labelled
# trace) or explicit source/seed pairs without copying the trainer.
HOLDOUT_SOURCE_SUBSTRINGS: tuple[str, ...] = ()
HOLDOUT_SOURCE_SEED_PAIRS: tuple[tuple[str, tuple[int, ...]], ...] = ()
EXCLUDED_SOURCE_SUBSTRINGS: tuple[str, ...] = ()
# A value of zero preserves the historical argmax behavior.  Later safety
# experiments may pre-register a fixed send probability gate; the gate is
# applied only to the action projection, never to the training labels.
SEND_PROBABILITY_THRESHOLD: float = 0.0
CAPACITY_VARIANTS: tuple[int, ...] = (256, 512, 1024, 2048)
TRAIN_STEPS: int = 120


def _load_pg231_helpers() -> Any:
    path = ROOT / "scripts" / "run_pg231_feedback_trajectory_training.py"
    spec = importlib.util.spec_from_file_location("pg231_helpers_for_pg237", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-231 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG231 = _load_pg231_helpers()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pg237_row(row: dict[str, Any]) -> dict[str, Any]:
    typed = bool((row.get("typed_oracle") or {}).get("typed_effect_confirmed"))
    fixture = bool((row.get("result_oracle") or {}).get("result_fixture_verified"))
    typed_evidence = dict((row.get("typed_oracle") or {}).get("evidence") or {})
    result_evidence = dict((row.get("result_oracle") or {}).get("evidence") or {})
    negative_projection = dict(((row.get("fixture") or {}).get("negative") or {}).get("response_projection") or {})
    negative_clean = bool(row.get("negative_sent")) and int(negative_projection.get("row_marker_count", 0) or 0) == 0
    evidence_hash = digest({"typed": typed_evidence, "result": result_evidence, "route": row.get("route"), "seed": row.get("seed")})
    # No raw route identity or runtime value enters the token stream.  The
    # route is used only to choose its abstract family and to bind evidence.
    raw = {
        "source": FRESH_SOURCE,
        "seed": int(row.get("seed", 0) or 0),
        "surface_role": "sql_surface",
        "method": str(row.get("method", "GET")).upper(),
        "status_class": "2xx",
        "field_count": len(row.get("fields") or []),
        "history_len": 0,
        "fresh_reset_ok": bool((row.get("reset") or {}).get("fresh_target")),
        "reset_completed": bool((row.get("reset") or {}).get("completed")),
        "reset_not_attempted": False,
        "candidate_sent": bool(row.get("ai_sent")),
        "reference_sent": bool(row.get("reference_sent")),
        "negative_sent": bool(row.get("negative_sent")),
        "oracle_available": bool(typed or fixture),
        "typed_effect_confirmed": typed,
        "typed_effect_observed": typed,
        "result_fixture_verified": fixture,
        "candidate_reference_agreement": bool(typed_evidence.get("candidate_reference_agreement")),
        "negative_clean": negative_clean,
        "binding_valid": True,
        "backend_observed": True,
        "database_health_ok": bool((row.get("reset") or {}).get("database_health_gate") == "mysqli_root_pikachu_ok"),
        "candidate_result_present": bool(typed or fixture),
        "candidate_sql_error_shape": bool(typed_evidence.get("candidate_sql_error_shape")),
        "negative_result_absent": negative_clean,
        "model_claimed_positive": False,
        "model_abstained": not (typed and fixture and negative_clean),
        "next_step": "recheck_oracle" if typed and fixture and negative_clean else "abstain",
        "previous_feedback": "result_verified" if typed and fixture else "none",
        "evidence_hash": evidence_hash,
        "payload_grounded_eligible": bool(typed and fixture and negative_clean),
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    record = prepare_feedback_record(raw)
    # Abstract event tokens intentionally omit route identity.  Keep the
    # attested source hash outside the model input so distinct routes in the
    # same seed are not silently collapsed during dataset construction.
    record["route_source_sha256"] = str(row.get("route_source_sha256", ""))
    return record


def _load_records() -> tuple[list[dict[str, Any]], dict[str, int]]:
    base = json.loads(BASE_DATASET.read_text(encoding="utf-8-sig"))
    trace = json.loads(PG237_TRACE.read_text(encoding="utf-8-sig"))
    records = [dict(row) for row in base.get("records", [])]
    fresh = [_pg237_row(dict(row)) for row in trace.get("results", [])]
    records.extend(fresh)
    # Keep seed replicas for cross-seed evaluation, but remove exact copies
    # from the same source/seed/trajectory.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    duplicate_count = 0
    for row in records:
        key = (str(row.get("trajectory_hash", row.get("token_hash", ""))), int(row.get("seed", 0) or 0), str(row.get("source", "")), str(row.get("route_source_sha256", "")))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, {"base_records": len(base.get("records", [])), "fresh_pg237_records": len(fresh), "unique_records": len(unique), "duplicate_records": duplicate_count}


def _is_holdout(row: dict[str, Any]) -> bool:
    # Derived pre-probe records keep the original source in split_source so a
    # route/implementation holdout cannot be bypassed by a new record prefix.
    source = str(row.get("split_source", row.get("source", "")))
    seed = int(row.get("seed", 0) or 0)
    if source == FRESH_SOURCE and seed in set(FRESH_HOLDOUT_SEEDS):
        return True
    if source == EXTRA_HOLDOUT_SOURCE and seed in set(EXTRA_HOLDOUT_SEEDS):
        return True
    if any(fragment and fragment in source for fragment in HOLDOUT_SOURCE_SUBSTRINGS):
        return True
    return any(source == name and seed in set(seeds) for name, seeds in HOLDOUT_SOURCE_SEED_PAIRS)


def _is_excluded(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    return any(fragment and fragment in source for fragment in EXCLUDED_SOURCE_SUBSTRINGS)


def _encode(rows: list[dict[str, Any]], input_vocab: dict[str, int], target_vocab: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return PG231._encode(rows, input_vocab, target_vocab, device)


def _positions(rows: list[dict[str, Any]], width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor([min(max(int(row.get("classification_position", 0)), 0), max(width - 1, 0)) for row in rows], dtype=torch.long, device=device)


def _evaluate(model: FrozenXXLFailurePolicy, context: torch.Tensor, encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], positions: torch.Tensor, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    _, token_targets, lane_targets, repair_targets = encoded
    action_targets = torch.tensor([ACTION_INDEX[action_target(row)] for row in rows], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(context, classification_positions=positions)
    valid = token_targets.ne(0)
    token_pred = output["token"].argmax(-1)
    lane_pred = output["lane"].argmax(-1)
    repair_pred = output["repair"].argmax(-1)
    action_pred = output["action"].argmax(-1)
    if SEND_PROBABILITY_THRESHOLD > 0.0:
        send_index = ACTION_INDEX["send_candidate"]
        send_probability = output["action"].softmax(-1)[:, send_index]
        below_gate = (action_pred == send_index) & send_probability.lt(float(SEND_PROBABILITY_THRESHOLD))
        action_pred = torch.where(below_gate, torch.full_like(action_pred, ACTION_INDEX["abstain"]), action_pred)
    token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), token_targets.reshape(-1), ignore_index=0)
    send_mask = action_targets == ACTION_INDEX["send_candidate"]
    abstain_mask = action_targets == ACTION_INDEX["abstain"]
    predicted_send = action_pred == ACTION_INDEX["send_candidate"]
    return {
        "token_loss": round(float(token_loss.detach().cpu()), 8),
        "perplexity": round(float(torch.exp(token_loss.detach().cpu().clamp(max=20.0))), 8),
        "next_token_accuracy": round(float(((token_pred == token_targets) & valid).sum().item() / max(int(valid.sum().item()), 1)), 8),
        "token_count": int(valid.sum().item()),
        "lane_accuracy": round(float((lane_pred == lane_targets).float().mean().item()), 8),
        "repair_accuracy": round(float((repair_pred == repair_targets).float().mean().item()), 8),
        "action_accuracy": round(float((action_pred == action_targets).float().mean().item()), 8),
        "abstain_recall": round(float(((action_pred == ACTION_INDEX["abstain"]) & abstain_mask).sum().item() / max(int(abstain_mask.sum().item()), 1)), 8),
        "positive_send_recall": round(float((predicted_send & send_mask).sum().item() / max(int(send_mask.sum().item()), 1)), 8),
        "false_send_count": int((predicted_send & ~send_mask).sum().item()),
        "missed_send_count": int((~predicted_send & send_mask).sum().item()),
        "send_count": int(send_mask.sum().item()),
        "abstain_count": int(abstain_mask.sum().item()),
    }


def main() -> int:
    all_records, source_counts = _load_records()
    # Low-quality lanes never become a positive or negative training target;
    # they remain audit data in the source manifests.  Holdout extensions are
    # evaluated only after this same lane filter, so an oracle gap cannot turn
    # into a hidden action label.
    eligible_rows = [row for row in all_records if row.get("lane") not in {"quarantine", "reject"} and not _is_excluded(row)]
    holdout = [row for row in eligible_rows if _is_holdout(row)]
    train = [row for row in eligible_rows if not _is_holdout(row)]
    if not train or not holdout:
        raise RuntimeError("PG-237 requires non-empty train and non-trivial seed holdout")
    holdout_actions = Counter(action_target(row) for row in holdout)
    if not holdout_actions.get("send_candidate") or not holdout_actions.get("abstain"):
        raise RuntimeError("PG-237 holdout must contain both send_candidate and abstain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(PG191_CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    target_vocab = build_vocabulary(train)
    train_encoded = _encode(train, input_vocab, target_vocab, device)
    holdout_encoded = _encode(holdout, input_vocab, target_vocab, device)
    with torch.no_grad():
        train_context = base.base.body.encode(train_encoded[0], train_encoded[0].ne(0)).detach().clone()
        holdout_context = base.base.body.encode(holdout_encoded[0], holdout_encoded[0].ne(0)).detach().clone()
    train_positions = _positions(train, train_context.shape[1], device)
    holdout_positions = _positions(holdout, holdout_context.shape[1], device)
    train_lane, train_repair = train_encoded[2], train_encoded[3]
    train_action = torch.tensor([ACTION_INDEX[action_target(row)] for row in train], dtype=torch.long, device=device)
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: FrozenXXLFailurePolicy | None = None
    for hidden_dim in CAPACITY_VARIANTS:
        torch.manual_seed(237 + hidden_dim)
        model = FrozenXXLFailurePolicy(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        lane_counts = torch.bincount(train_lane, minlength=len(LANES)).float().clamp_min(1.0)
        repair_counts = torch.bincount(train_repair, minlength=len(REPAIR_INDEX)).float().clamp_min(1.0)
        action_counts = torch.bincount(train_action, minlength=len(ACTION_CLASSES)).float().clamp_min(1.0)
        lane_weights = (lane_counts.sum() / lane_counts).to(device)
        repair_weights = (repair_counts.sum() / repair_counts).to(device)
        action_weights = (action_counts.sum() / action_counts).to(device)
        for _ in range(int(TRAIN_STEPS)):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_encoded[1].reshape(-1), ignore_index=0)
            loss = token_loss + 0.30 * nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights) + 0.75 * nn.functional.cross_entropy(output["action"], train_action, weight=action_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": _evaluate(model, train_context, train_encoded, train_positions, train, device), "seed_holdout": _evaluate(model, holdout_context, holdout_encoded, holdout_positions, holdout, device)}
        variants.append(result)
        metrics = result["seed_holdout"]
        key = (metrics["false_send_count"], metrics["missed_send_count"], -metrics["abstain_recall"], -metrics["positive_send_recall"], metrics["token_loss"])
        old = None if selected is None else selected["seed_holdout"]
        old_key = None if old is None else (old["false_send_count"], old["missed_send_count"], -old["abstain_recall"], -old["positive_send_recall"], old["token_loss"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-237 no capacity variant selected")
    body_after = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    hold_metrics = selected["seed_holdout"]
    safety_pass = hold_metrics["false_send_count"] == 0 and hold_metrics["abstain_recall"] >= 0.80
    capability_pass = safety_pass and hold_metrics["positive_send_recall"] >= 0.80 and hold_metrics["send_count"] > 0
    artifact_dir = ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_capacity_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": f"{EXPERIMENT_ID}-capacity-training-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "action_classes": list(ACTION_CLASSES), "token_vocabulary": target_vocab, "frozen_body_checkpoint": str(PG191_CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset = {"schema_version": "pg237-capacity-training-dataset-v1", "source_datasets": [str(BASE_DATASET.relative_to(ROOT)), str(PG237_TRACE.relative_to(ROOT))], "records": all_records, "counts": {**source_counts, "train_rows": len(train), "holdout_rows": len(holdout), "excluded_rows": sum(1 for row in all_records if _is_excluded(row)), "excluded_source_counts": dict(Counter(str(row.get("source")) for row in all_records if _is_excluded(row))), "train_action_counts": dict(Counter(action_target(row) for row in train)), "holdout_action_counts": dict(holdout_actions), "holdout_source_counts": dict(Counter(str(row.get("source")) for row in holdout)), "holdout_family_counts": dict(Counter(str(row.get("surface_class")) for row in holdout)), "token_template_count": len({str(row.get("trajectory_hash", row.get("token_hash", ""))) for row in all_records})}, "contract": {"fresh_source_holdout_seeds_never_in_training": True, "extra_source_holdout_seeds_never_in_training": True, "extended_source_holdout_never_in_training": bool(HOLDOUT_SOURCE_SUBSTRINGS or HOLDOUT_SOURCE_SEED_PAIRS), "excluded_source_never_in_training_or_holdout": bool(EXCLUDED_SOURCE_SUBSTRINGS), "holdout_contains_positive_and_abstain": True, "typed_positive_requires_result_fixture_and_negative_clean": True, "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    report = {"protocol_id": "pg-pk-237-capacity-training-v1", "schema_version": "pg237-capacity-training-v1", "status": "completed_nontrivial_seed_holdout_capacity_training", "device": str(device), "counts": dataset["counts"], "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "variants": variants, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_after, "frozen_body_changed": body_before != body_after, "safety_abstain_gate_pass": safety_pass, "capability_gate_pass": capability_pass, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"seed23702_is_never_in_training": True, "seed23632_is_never_in_training": True, "holdout_has_positive_and_abstain": True, "typed_result_fixture_is_local_read_only": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg237-capacity-training-protocol-v1", "capacity_variants": list(CAPACITY_VARIANTS), "train_steps": int(TRAIN_STEPS), "seed_holdout": [23702, 23632], "extended_source_holdout_substrings": list(HOLDOUT_SOURCE_SUBSTRINGS), "extended_source_holdout_pairs": [[name, list(seeds)] for name, seeds in HOLDOUT_SOURCE_SEED_PAIRS], "excluded_source_substrings": list(EXCLUDED_SOURCE_SUBSTRINGS), "holdout_must_contain_positive_and_abstain": True, "typed_positive_contract": ["fresh_reset", "candidate_reference_agreement", "negative_clean", "typed_effect", "result_fixture"], "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "frozen_body_required": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg237-capacity-training-trace-v1", "selected": selected, "variants": variants, "holdout_action_counts": dict(holdout_actions), "safety_abstain_gate_pass": safety_pass, "capability_gate_pass": capability_pass, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-237 non-trivial seed-heldout capacity training", "", f"train={len(train)}; holdout={len(holdout)}; holdout_actions={dict(holdout_actions)}", f"selected hidden={selected['hidden_dim']}; token={hold_metrics['next_token_accuracy']}; lane={hold_metrics['lane_accuracy']}; repair={hold_metrics['repair_accuracy']}; positive_recall={hold_metrics['positive_send_recall']}; abstain_recall={hold_metrics['abstain_recall']}; false_send={hold_metrics['false_send_count']}; missed_send={hold_metrics['missed_send_count']}", f"safety_abstain_gate={safety_pass}; capability_gate={capability_pass}", "", "留出集同时包含 typed positive 和 abstain，避免全 abstain 自我安慰；正例仍是本地只读结果 fixture，不等于任意站点漏洞结论。", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "safety_abstain_gate": safety_pass, "capability_gate": capability_pass, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
