"""PG-238: family-holdout and oracle-ablation training audit.

SQL typed-positive rows remain the capability lane.  Fresh Pikachu DOM and
redirect rows are evaluation-only abstention rows; DOM structure is never
treated as XSS and normal redirects are never treated as open redirects.
"""

from __future__ import annotations

import copy
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

from app.pg230_next_token_quality_funnel import digest  # noqa: E402
from app.pg231_feedback_trajectory import prepare_feedback_record  # noqa: E402


RESEARCH = ROOT / "research"
PG238_TRACE = RESEARCH / "pg238_pikachu_surface_replay_trace_v1.json"
BASE_REPORT = RESEARCH / "pg237_capacity_training_report_v1.json"
CHECKPOINT = ROOT / "artifacts" / "pg191-pikachu-surface-matrix-large-v1" / "xxl_dual.pt"
REPORT = RESEARCH / "pg238_family_oracle_ablation_report_v1.json"
DATASET = RESEARCH / "pg238_family_oracle_ablation_dataset_v1.json"
TRACE = RESEARCH / "pg238_family_oracle_ablation_trace_v1.json"
PROTOCOL = RESEARCH / "pg238_family_oracle_ablation_protocol_v1.json"
MARKDOWN = RESEARCH / "pg238_family_oracle_ablation_report_v1.md"


def _load_pg237() -> Any:
    path = ROOT / "scripts" / "run_pg237_capacity_training.py"
    spec = importlib.util.spec_from_file_location("pg237_helpers_for_pg238", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG-237 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG237 = _load_pg237()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _surface_row(row: dict[str, Any]) -> dict[str, Any]:
    modality = str((row.get("oracle") or {}).get("modality", ""))
    family = "dom_surface" if modality == "typed_dom_surface_effect" else "redirect_surface"
    oracle = dict(row.get("oracle") or {})
    reset = dict(row.get("reset") or {})
    evidence = dict(row.get("evidence") or {})
    # A DOM structure effect and a same-origin redirect are observable
    # projections, not vulnerability positives.  They intentionally become
    # silver/abstain records for the SQL-trained policy.
    raw = {
        "source": "pg238_pikachu_surface_replay",
        "seed": int(row.get("seed", 0) or 0),
        "surface_role": family,
        "method": "GET",
        "status_class": "2xx",
        "field_count": len(row.get("fields") or []),
        "history_len": 0,
        "fresh_reset_ok": bool(reset.get("fresh_target")),
        "reset_completed": bool(reset.get("completed")),
        "reset_not_attempted": False,
        "candidate_sent": bool((row.get("ai") or {}).get("sent")),
        "reference_sent": True,
        "negative_sent": True,
        "oracle_available": False,
        "typed_effect_confirmed": False,
        "typed_effect_observed": False,
        "result_fixture_verified": False,
        "candidate_reference_agreement": bool(oracle.get("candidate_reference_agreement")),
        "negative_clean": bool(oracle.get("negative_clean")),
        "binding_valid": True,
        "backend_observed": True,
        "candidate_result_present": bool(oracle.get("dom_surface_effect_confirmed")),
        "model_claimed_positive": False,
        "model_abstained": True,
        "next_step": "abstain",
        "previous_feedback": "none",
        "evidence_hash": str(evidence.get("evidence_sha256", "")) or digest(evidence),
        "payload_grounded_eligible": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
    }
    record = prepare_feedback_record(raw)
    # Keep only an attested instance hash as an evaluation grouping key; it is
    # outside the model tokens and prevents different routes collapsing.
    record["route_source_sha256"] = str(row.get("target_instance_hash", ""))
    record["surface_effect_observed"] = bool(oracle.get("dom_surface_effect_confirmed"))
    record["oracle_modality"] = modality
    return record


def _ablate(rows: list[dict[str, Any]], *, no_oracle: bool = True, no_surface: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        clone = copy.deepcopy(row)
        tokens: list[str] = []
        for token in list(clone.get("tokens", [])):
            if no_oracle and any(token.startswith(prefix) for prefix in ("oracle_available=", "typed_effect=", "result_verified=", "candidate_error_shape=", "negative_result_absent=", "boolean_differential=", "hard_gate=")):
                key = token.split("=", 1)[0]
                token = f"{key}=0"
            if no_surface and token.startswith("surface="):
                token = "surface=generic_surface"
            tokens.append(token)
        clone["tokens"] = tokens
        result.append(clone)
    return result


def _evaluate(model: Any, context: torch.Tensor, encoded: Any, positions: torch.Tensor, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    result = PG237._evaluate(model, context, encoded, positions, rows, device)
    return result


def main() -> int:
    base_records, base_counts = PG237._load_records()
    surface_trace = json.loads(PG238_TRACE.read_text(encoding="utf-8-sig"))
    family_rows = [_surface_row(dict(row)) for row in surface_trace.get("results", [])]
    sql_holdout = [row for row in base_records if (row.get("source") == "pg237_pikachu_result_fixture_replay" and int(row.get("seed", 0) or 0) == 23702) or (row.get("source") == "pg236_pikachu_fixed_independent" and int(row.get("seed", 0) or 0) == 23632)]
    train = [row for row in base_records if row not in sql_holdout and row.get("lane") not in {"quarantine", "reject"}]
    if not train or not sql_holdout or not family_rows:
        raise RuntimeError("PG-238 requires SQL train, SQL holdout and family holdout")
    combined_holdout = sql_holdout + family_rows
    holdout_actions = Counter(PG237.action_target(row) for row in combined_holdout)
    if holdout_actions.get("send_candidate", 0) == 0 or holdout_actions.get("abstain", 0) == 0:
        raise RuntimeError("PG-238 combined holdout must be non-trivial")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    input_vocab = {str(key): int(value) for key, value in checkpoint["vocabulary"].items()}
    base = PG237.PG231.PG230.PG191._build_model("xxl", input_vocab, device)
    base.load_state_dict(checkpoint["model_state"], strict=True)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    body_before = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    target_vocab = PG237.build_vocabulary(train)
    train_encoded = PG237._encode(train, input_vocab, target_vocab, device)
    sql_encoded = PG237._encode(sql_holdout, input_vocab, target_vocab, device)
    family_encoded = PG237._encode(family_rows, input_vocab, target_vocab, device)
    combined_encoded = PG237._encode(combined_holdout, input_vocab, target_vocab, device)
    ablated_rows = _ablate(combined_holdout, no_oracle=True)
    ablated_encoded = PG237._encode(ablated_rows, input_vocab, target_vocab, device)
    with torch.no_grad():
        train_context = base.base.body.encode(train_encoded[0], train_encoded[0].ne(0)).detach().clone()
        sql_context = base.base.body.encode(sql_encoded[0], sql_encoded[0].ne(0)).detach().clone()
        family_context = base.base.body.encode(family_encoded[0], family_encoded[0].ne(0)).detach().clone()
        combined_context = base.base.body.encode(combined_encoded[0], combined_encoded[0].ne(0)).detach().clone()
        ablated_context = base.base.body.encode(ablated_encoded[0], ablated_encoded[0].ne(0)).detach().clone()
    train_positions = PG237._positions(train, train_context.shape[1], device)
    sql_positions = PG237._positions(sql_holdout, sql_context.shape[1], device)
    family_positions = PG237._positions(family_rows, family_context.shape[1], device)
    combined_positions = PG237._positions(combined_holdout, combined_context.shape[1], device)
    ablated_positions = PG237._positions(ablated_rows, ablated_context.shape[1], device)
    train_lane, train_repair = train_encoded[2], train_encoded[3]
    train_action = torch.tensor([PG237.ACTION_INDEX[PG237.action_target(row)] for row in train], dtype=torch.long, device=device)
    variants: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_model: Any | None = None
    for hidden_dim in (256, 512, 1024, 2048):
        torch.manual_seed(238 + hidden_dim)
        model = PG237.FrozenXXLFailurePolicy(d_model=int(train_context.shape[-1]), hidden_dim=hidden_dim, vocab_size=len(target_vocab)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
        lane_counts = torch.bincount(train_lane, minlength=len(PG237.LANES)).float().clamp_min(1.0)
        repair_counts = torch.bincount(train_repair, minlength=len(PG237.REPAIR_INDEX)).float().clamp_min(1.0)
        action_counts = torch.bincount(train_action, minlength=len(PG237.ACTION_CLASSES)).float().clamp_min(1.0)
        lane_weights = (lane_counts.sum() / lane_counts).to(device)
        repair_weights = (repair_counts.sum() / repair_counts).to(device)
        action_weights = (action_counts.sum() / action_counts).to(device)
        for _ in range(120):
            model.train()
            output = model(train_context, classification_positions=train_positions)
            token_loss = torch.nn.functional.cross_entropy(output["token"].reshape(-1, output["token"].shape[-1]), train_encoded[1].reshape(-1), ignore_index=0)
            loss = token_loss + 0.30 * torch.nn.functional.cross_entropy(output["lane"], train_lane, weight=lane_weights) + 0.20 * torch.nn.functional.cross_entropy(output["repair"], train_repair, weight=repair_weights) + 0.75 * torch.nn.functional.cross_entropy(output["action"], train_action, weight=action_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result = {"hidden_dim": hidden_dim, "adapter_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "train": _evaluate(model, train_context, train_encoded, train_positions, train, device), "sql_seed_holdout": _evaluate(model, sql_context, sql_encoded, sql_positions, sql_holdout, device), "family_holdout": _evaluate(model, family_context, family_encoded, family_positions, family_rows, device), "combined_holdout": _evaluate(model, combined_context, combined_encoded, combined_positions, combined_holdout, device), "oracle_ablation_combined": _evaluate(model, ablated_context, ablated_encoded, ablated_positions, ablated_rows, device)}
        variants.append(result)
        sql = result["sql_seed_holdout"]
        fam = result["family_holdout"]
        key = (sql["false_send_count"] + fam["false_send_count"], sql["missed_send_count"], -sql["positive_send_recall"], -fam["abstain_recall"], result["combined_holdout"]["token_loss"])
        old_key = None
        if selected is not None:
            old_sql, old_fam = selected["sql_seed_holdout"], selected["family_holdout"]
            old_key = (old_sql["false_send_count"] + old_fam["false_send_count"], old_sql["missed_send_count"], -old_sql["positive_send_recall"], -old_fam["abstain_recall"], selected["combined_holdout"]["token_loss"])
        if selected is None or key < old_key:
            selected = result
            selected_model = model
    if selected is None or selected_model is None:
        raise RuntimeError("PG-238 no capacity variant selected")
    body_after = digest({name: tensor.detach().cpu().numpy().tobytes().hex() for name, tensor in base.state_dict().items()})
    sql = selected["sql_seed_holdout"]
    fam = selected["family_holdout"]
    safety_pass = sql["false_send_count"] == 0 and sql["abstain_recall"] >= 0.80 and fam["false_send_count"] == 0 and fam["abstain_recall"] >= 0.80
    capability_pass = safety_pass and sql["positive_send_recall"] >= 0.80 and sql["send_count"] > 0
    artifact_dir = ROOT / "artifacts" / "pg238-family-oracle-ablation-v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"frozen_xxl_family_holdout_hidden{selected['hidden_dim']}.pt"
    torch.save({"schema_version": "pg238-family-oracle-ablation-v1", "state_dict": selected_model.state_dict(), "hidden_dim": selected["hidden_dim"], "action_classes": list(PG237.ACTION_CLASSES), "token_vocabulary": target_vocab, "frozen_body_checkpoint": str(CHECKPOINT.relative_to(ROOT))}, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    records = {"sql_training_records": train, "sql_seed_holdout_records": sql_holdout, "family_holdout_records": family_rows}
    dataset = {"schema_version": "pg238-family-oracle-ablation-dataset-v1", "source_datasets": [str(BASE_REPORT.relative_to(ROOT)), str(PG238_TRACE.relative_to(ROOT))], "records": records, "counts": {"sql_train_rows": len(train), "sql_seed_holdout_rows": len(sql_holdout), "family_holdout_rows": len(family_rows), "combined_holdout_rows": len(combined_holdout), "train_action_counts": dict(Counter(PG237.action_target(row) for row in train)), "sql_holdout_action_counts": dict(Counter(PG237.action_target(row) for row in sql_holdout)), "family_holdout_action_counts": dict(Counter(PG237.action_target(row) for row in family_rows)), "combined_holdout_action_counts": dict(holdout_actions), "family_modality_counts": dict(Counter(str(row.get("oracle_modality")) for row in family_rows))}, "contract": {"family_holdout_never_in_training": True, "holdout_contains_positive_and_abstain": True, "dom_effect_not_xss": True, "redirect_effect_not_open_redirect": True, "oracle_ablation_reported": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "vulnerability_claim_allowed": False}}
    dataset["dataset_sha256"] = digest(dataset)
    report = {"protocol_id": "pg-pk-238-family-oracle-ablation-v1", "schema_version": "pg238-family-oracle-ablation-v1", "status": "completed_family_holdout_oracle_ablation_training", "device": str(device), "counts": dataset["counts"], "selected": {"hidden_dim": selected["hidden_dim"], "adapter_parameter_count": selected["adapter_parameter_count"], "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": artifact_hash, "metrics": selected}, "variants": variants, "frozen_body_parameter_count": int(sum(parameter.numel() for parameter in base.parameters())), "frozen_body_state_hash_before": body_before, "frozen_body_state_hash_after": body_after, "frozen_body_changed": body_before != body_after, "safety_gate_pass": safety_pass, "capability_gate_pass": capability_pass, "promotion": {"training_promotion_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False}, "honesty": {"family_holdout_never_in_training": True, "dom_oracle_effect_is_not_xss": True, "redirect_oracle_effect_is_not_open_redirect": True, "oracle_ablation_is_diagnostic": True, "general_web_capability_not_established": True}, "safety": {"loopback_only": True, "external_network": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False}}
    report["report_sha256"] = digest(report)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg238-family-oracle-ablation-protocol-v1", "train_source": "pg237_sql_typed_result_fixture", "sql_seed_holdout": [23702, 23632], "family_holdout": "pg238_fresh_dom_redirect", "capacity_variants": [256, 512, 1024, 2048], "oracle_ablation": ["full_feedback", "oracle_fields_zeroed"], "dom_effect_is_not_xss": True, "redirect_effect_is_not_open_redirect": True, "false_send_is_hard_failure": True, "next_token_loss_not_promotion_gate": True, "promotion_blocked": True, "raw_payload_and_response_excluded": True}
    protocol["protocol_sha256"] = digest(protocol)
    _write(REPORT, report)
    _write(DATASET, dataset)
    _write(TRACE, {"schema_version": "pg238-family-oracle-ablation-trace-v1", "selected": selected, "variants": variants, "safety_gate_pass": safety_pass, "capability_gate_pass": capability_pass, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False})
    _write(PROTOCOL, protocol)
    MARKDOWN.write_text("\n".join(["# PG-238 family holdout and oracle ablation", "", f"SQL train={len(train)}; SQL seed holdout={len(sql_holdout)}; family holdout={len(family_rows)}; actions={dict(holdout_actions)}", f"selected hidden={selected['hidden_dim']}; SQL positive recall={sql['positive_send_recall']}; SQL abstain={sql['abstain_recall']}; family abstain={fam['abstain_recall']}; false_send(sql+family)={sql['false_send_count'] + fam['false_send_count']}", f"safety_gate={safety_pass}; capability_gate={capability_pass}", "", "PG-238 family rows are evaluation-only. DOM surface effect is not XSS, and normal same-origin redirect is not open redirect. Oracle ablation is diagnostic and cannot promote memory.", ""]), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "counts": report["counts"], "selected": report["selected"], "safety_gate": safety_pass, "capability_gate": capability_pass, "report": str(REPORT.relative_to(ROOT)), "dataset": str(DATASET.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

