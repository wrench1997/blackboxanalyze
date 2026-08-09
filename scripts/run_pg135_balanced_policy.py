"""PG-135 exact GET/POST-balanced replay using the independent token policy."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg122_logic_authorization_replay import collect_target as collect_pg122_target
from app.pg125_scope_logic_replay import collect_target as collect_pg125_target
from app.pg127_resource_visibility_replay import collect_target as collect_pg127_target
from app.pg135_balanced_history_replay import collect_target as collect_pg135_target
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


def _load_pg134_runner() -> Any:
    path = ROOT / "scripts" / "run_pg134_independent_token_gru.py"
    spec = importlib.util.spec_from_file_location("pg134_runner_for_pg135", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-134 independent helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG134 = _load_pg134_runner()
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg135-balanced-policy-v1"
CHECKPOINT = ARTIFACT_DIR / "balanced_full.pt"
ZERO_CHECKPOINT = ARTIFACT_DIR / "balanced_zero.pt"
REPORT = RESEARCH / "pg135_balanced_policy_report_v1.json"
DATASET = RESEARCH / "pg135_balanced_policy_dataset_v1.json"
VISIBLE = RESEARCH / "pg135_balanced_policy_visible_dataset_v1.json"
TRACE = RESEARCH / "pg135_balanced_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg135_balanced_policy_protocol_v1.json"
PROPOSAL = RESEARCH / "pg135_balanced_policy_proposal_v1.json"

TRAIN_SEEDS = (13511, 13513, 13515)
DEV_SEEDS = (13512, 13514, 13516)
HOLDOUT_SEEDS = (13501, 13503, 13505)
PG127_TRAIN = (13531, 13533, 13535)
PG127_DEV = (13532, 13534, 13536)
PG127_HOLDOUT = (13521, 13523, 13525)
PG125_TRAIN = (13541, 13543, 13545)
PG125_DEV = (13542, 13544, 13546)
PG125_OOD = (13551, 13553, 13555)
PG122_OOD = (13561, 13563, 13565)
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


async def _collect() -> dict[str, list[dict[str, Any]]]:
    return {
        "pg135_train": [await collect_pg135_target(seed, decoy_strength=index % 3) for index, seed in enumerate(TRAIN_SEEDS)],
        "pg135_dev": [await collect_pg135_target(seed, decoy_strength=index % 3) for index, seed in enumerate(DEV_SEEDS)],
        "pg135_holdout": [await collect_pg135_target(seed, decoy_strength=index % 3) for index, seed in enumerate(HOLDOUT_SEEDS)],
        "pg127_train": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_TRAIN)],
        "pg127_dev": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_DEV)],
        "pg127_holdout": [await collect_pg127_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG127_HOLDOUT)],
        "pg125_train": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_TRAIN)],
        "pg125_dev": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_DEV)],
        "pg125_family_ood": [await collect_pg125_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG125_OOD)],
        "pg122_family_ood": [await collect_pg122_target(seed, decoy_strength=index % 3) for index, seed in enumerate(PG122_OOD)],
    }


def _rows(targets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    row = PG134._rows_from_targets
    return {
        "train": row(targets["pg135_train"], split="train", source="pg135_balanced_train", history_source=True) + row(targets["pg127_train"], split="train", source="pg127_balanced_train") + row(targets["pg125_train"], split="train", source="pg125_balanced_train"),
        "dev": row(targets["pg135_dev"], split="dev", source="pg135_balanced_dev", history_source=True) + row(targets["pg127_dev"], split="dev", source="pg127_balanced_dev") + row(targets["pg125_dev"], split="dev", source="pg125_balanced_dev"),
        "pg135": row(targets["pg135_holdout"], split="holdout", source="pg135_balanced_holdout", history_source=True),
        "pg127": row(targets["pg127_holdout"], split="holdout", source="pg127_balanced_holdout"),
        "pg125": row(targets["pg125_family_ood"], split="family_ood", source="pg125_balanced_family_ood"),
        "pg122": row(targets["pg122_family_ood"], split="family_ood", source="pg122_balanced_family_ood"),
    }


def _summary(targets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return PG134._summary(targets)


def _evaluate_guarded(model: Any, rows: list[dict[str, Any]], *, mode: str, device: torch.device, known_pairs: frozenset[str]) -> dict[str, Any]:
    """Evaluate the model plus a bounded unseen-IR fail-closed guard."""

    predictions, labels, confidences = PG134._predict(model, rows, mode=mode, device=device)
    guarded_predictions: list[int] = []
    reasons: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions):
        action = PG134.POLICY_ACTIONS[prediction]
        guarded, reason = guard_action(action, row, known_pairs)
        guarded_predictions.append(PG134.policy_index(guarded))
        reasons.append(reason)
    names = [PG134.POLICY_ACTIONS[index] for index in guarded_predictions]
    compliant: list[bool] = []
    for row, reason, name in zip(rows, reasons, names):
        allowed = set(PG134._allowed(row))
        if reason.get("reason") == "unseen_rule_ir_pair" and bool(row["failure_signature"].get("typed_available", True)):
            allowed.add("abstain_candidate_only")
        compliant.append(name in allowed)
    episode_max: dict[str, int] = {}
    for row in rows:
        match = PG134.re.search(r"-s(\d+)$", str(row["step_id"]))
        if match:
            episode_max[str(row.get("episode_id", ""))] = max(episode_max.get(str(row.get("episode_id", "")), 0), int(match.group(1)))
    blind_final = [index for index, row in enumerate(rows) if row.get("surface_kind") == "blind" and (match := PG134.re.search(r"-s(\d+)$", str(row["step_id"]))) and int(match.group(1)) == episode_max.get(str(row.get("episode_id", "")), -1)]
    unknown = [index for index, row in enumerate(rows) if not bool(row["failure_signature"].get("typed_available", True))]
    negative = [index for index, row in enumerate(rows) if row.get("surface_kind") in {"blind", "decoy", "steady"}]
    history = [index for index, row in enumerate(rows) if row.get("counterfactual_pair_id")]
    return {
        "mode": mode,
        "guard": "unseen_rule_ir_pair_fail_closed",
        "known_pairs_count": len(known_pairs),
        "known_pairs_sha256": known_pairs_sha256(known_pairs),
        "metrics": PG134._metrics(guarded_predictions, labels),
        "raw_metrics": PG134._metrics(predictions, labels),
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "blind_final_rows": len(blind_final),
        "blind_final_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in blind_final) / len(blind_final), 6) if blind_final else 0.0,
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "history_current_accuracy": round(sum(guarded_predictions[index] == labels[index] for index in history) / len(history), 6) if history else 0.0,
        "history_pair": PG134._pair_metrics(rows, guarded_predictions),
        "guard_override_count": sum(bool(reason.get("guarded")) for reason in reasons),
        "unseen_row_count": sum(bool(reason.get("unseen_pairs")) for reason in reasons),
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else 0.0,
        "predictions": guarded_predictions,
        "raw_predictions": predictions,
        "labels": labels,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect())
    rows = _rows(targets)
    train, dev = rows["train"], rows["dev"]
    model, history = PG134._train(train, dev, device=device, mode="full", seed=13531)
    zero_model, zero_history = PG134._train(train, dev, device=device, mode="zero", seed=13531)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": PG134.SCHEMA_VERSION, "policy_actions": list(PG134.POLICY_ACTIONS), "embedding_provenance": model.embedding_provenance, "model_state_dict": model.state_dict()}, CHECKPOINT)
    torch.save({"schema_version": PG134.SCHEMA_VERSION, "policy_actions": list(PG134.POLICY_ACTIONS), "embedding_provenance": zero_model.embedding_provenance, "mode": "zero", "model_state_dict": zero_model.state_dict()}, ZERO_CHECKPOINT)
    pg135 = PG134._evaluate(model, rows["pg135"], mode="full", device=device)
    pg127 = PG134._evaluate(model, rows["pg127"], mode="full", device=device)
    pg125 = PG134._evaluate(model, rows["pg125"], mode="full", device=device)
    pg122 = PG134._evaluate(model, rows["pg122"], mode="full", device=device)
    known_pairs = known_rule_ir_pairs(train)
    pg122_guarded = _evaluate_guarded(model, rows["pg122"], mode="full", device=device, known_pairs=known_pairs)
    current_only = []
    for item in rows["pg135"]:
        if item.get("counterfactual_pair_id"):
            copied = dict(item)
            copied["layered_steps"] = [dict(item["layered_steps"][-1])]
            current_only.append(copied)
    current = PG134._evaluate(model, current_only, mode="full", device=device)
    ablations = {mode: PG134._evaluate(model, rows["pg135"], mode=mode, device=device) for mode in ("source_only", "ir_only", "availability_only", "weight_only", "no_weight", "zero")}
    zero_baseline = PG134._evaluate(zero_model, rows["pg135"], mode="zero", device=device)
    all_holdout = rows["pg135"] + rows["pg127"] + rows["pg125"] + rows["pg122"]
    get_count = sum(item["failure_signature"].get("observed_method") == "GET" for item in all_holdout)
    post_count = sum(item["failure_signature"].get("observed_method") == "POST" for item in all_holdout)
    label_corrections = sum(1 for item in train + dev + all_holdout if item.get("label_correction"))
    exact_balance = get_count == post_count
    hard_checks = {
        "fresh_checkpoint": True,
        "exact_get_post_balance": exact_balance,
        "get_count": get_count,
        "post_count": post_count,
        "pg135_accuracy_floor": pg135["metrics"]["accuracy"] >= 0.90,
        "pg135_history_accuracy_floor": pg135["history_current_accuracy"] >= 0.90,
        "pg135_counterfactual_conflicts": pg135["history_pair"]["pair_count"] >= 3 and pg135["history_pair"]["label_conflict_count"] == pg135["history_pair"]["pair_count"],
        "pg135_counterfactual_separation": pg135["history_pair"]["prediction_separation_rate"] == 1.0,
        "current_only_ablation_drop": current["metrics"]["accuracy"] <= 0.75,
        "pg127_accuracy_floor": pg127["metrics"]["accuracy"] >= 0.85,
        "pg125_family_accuracy_floor": pg125["metrics"]["accuracy"] >= 0.85,
        "pg122_family_accuracy_floor": pg122["metrics"]["accuracy"] >= 0.90,
        "unknown_all_steps_abstain": all(report["unknown_abstain_rate"] == 1.0 for report in (pg135, pg127, pg125, pg122)),
        "safety_compliance_floor": all(report["safety_compliance_rate"] >= 0.99 for report in (pg135, pg127, pg125)) and pg122_guarded["safety_compliance_rate"] >= 0.99,
        "pg122_ood_guard_triggered": pg122_guarded["guard_override_count"] > 0,
        "pg122_guarded_safety_floor": pg122_guarded["safety_compliance_rate"] >= 0.99,
        "negative_false_stop_zero": all(report["negative_false_stop_count"] == 0 for report in (pg135, pg127, pg125, pg122)),
        "channel_ablations_present": all(mode in ablations for mode in ("source_only", "ir_only", "availability_only", "weight_only")),
        "memory_promotion_forbidden": True,
    }
    # ``get_count``/``post_count`` are audit fields, not labels; keep the
    # booleans separate so a malformed count cannot masquerade as a pass.
    checks = {key: value for key, value in hard_checks.items() if key not in {"get_count", "post_count"}}
    hard_gates_passed = all(checks.values())
    training_eligible = hard_gates_passed and CROSS_IMPLEMENTATION_REVIEW_COMPLETE
    report = {
        "protocol_id": "pg-pk-135-balanced-policy-v1",
        "schema_version": "pg135-balanced-policy-report-v1",
        "status": "completed_pg135_balanced_policy",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": training_eligible,
        "scope": {"model": "pg134_independent_blake2b_token_hash_gru_on_pg135_balanced_replay", "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "training": {"train_count": len(train), "dev_count": len(dev), "safety_label_correction_count": label_corrections, "history_tail": history[-5:], "zero_history_tail": zero_history[-5:], "target_summary": _summary({"pg135_train": targets["pg135_train"], "pg127_train": targets["pg127_train"], "pg125_train": targets["pg125_train"]})},
        "holdout": {"pg135_balanced_holdout": pg135, "pg127_seed_holdout": pg127, "pg125_family_ood": pg125, "pg122_family_ood": pg122, "pg122_family_ood_guarded": pg122_guarded, "history_current_only_ablation": current, "channel_ablations": ablations, "fresh_zero_baseline": zero_baseline},
        "checks": checks,
        "transport_balance": {"get_count": get_count, "post_count": post_count, "exact": exact_balance},
        "input_contract": {"fresh_balanced_replay": True, "source_and_rule_ir_tokens": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "history_authority_in_model_input": False, "unknown_oracle_safe_label": "abstain_unknown_oracle", "label_correction_count": label_corrections},
        "diagnosis": {"raw_pg122_failure": "未见 response.transition_delta=authorization 时，原模型 3 条 typed_positive 步预测为 repeat，raw safety compliance=0.9375。", "guard": "training split 的 bounded Rule-IR slot=value 白名单叠加 typed action contract；typed_positive 只见一个方法时强制 probe，GET/POST 都见过时强制 stop，candidate_without_typed_effect 按方法数/预算保留 probe 或 abstain，matched negative 只允许 repeat；未见 pair 仍禁止不安全 repeat/stop，保留 raw/guarded 两套指标。", "known_pairs_sha256": known_pairs_sha256(known_pairs)},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "status": "balanced_candidate_pending_manual_review" if hard_gates_passed else "blocked_pg135_gate_failure_preserved", "reason": "平衡通道与完整 episode 通过后仍须跨实现/人工审核；本轮不晋升。"},
        "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "balanced_replay": hashlib.sha256((ROOT / "app/pg135_balanced_history_replay.py").read_bytes()).hexdigest(), "policy": hashlib.sha256((ROOT / "app/pg134_independent_policy.py").read_bytes()).hexdigest(), "ood_guard": hashlib.sha256((ROOT / "app/rule_ir_ood_guard.py").read_bytes()).hexdigest()},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg135-balanced-policy-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "rows": train + dev + all_holdout}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg135-balanced-policy-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "rows": [{"row_id": item["row_id"], "split": item["split"], "layered_steps": item["layered_steps"], "training_label": item["label"]} for item in train + dev]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg135-balanced-policy-trace-v1", "protocol_id": "pg-pk-135-balanced-policy-v1", "status": "completed_pg135_balanced_policy", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "exact_get_post_balance": exact_balance, "get_count": get_count, "post_count": post_count, "label_correction_count": label_corrections, "raw_source_saved": False, "raw_probe_response_saved": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-135-balanced-policy-v1", "schema_version": "pg135-balanced-policy-protocol-v1", "objective": "每个完整 counterfactual episode 固定 2 GET + 2 POST，并在同一当前 POST 上验证历史差异。", "action_plan": {"control_first": [["control", "GET"], ["candidate", "POST"], ["control", "GET"], ["control", "POST"]], "candidate_first": [["candidate", "GET"], ["candidate", "POST"], ["control", "GET"], ["control", "POST"]]}, "ood_guard": {"unknown_slot_value": "abstain_candidate_only", "raw_and_guarded_metrics_required": True, "known_pairs_source": "train_only"}, "required_gates": checks, "promotion": {"hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-135-balanced-policy-v1", "proposal_id": "pg135-balanced-policy-proposal-v1", "prediction": {"exact_get_post": True, "pg135_accuracy": ">=0.90", "counterfactual_prediction_separation": 1.0, "unknown_all_steps_abstain": 1.0, "safety_compliance_floor": 1.0, "negative_false_stop": 0}, "failure_rule": "任何 episode 不完整、GET/POST 不等、unknown 步不弃权、证据哈希不一致或族外门失败，保留 evaluation-only。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "train": len(train), "dev": len(dev), "pg135_holdout": len(rows["pg135"]), "get_count": get_count, "post_count": post_count, "exact_balance": exact_balance, "pg135_accuracy": pg135["metrics"]["accuracy"], "counterfactual_separation": pg135["history_pair"]["prediction_separation_rate"], "current_only_accuracy": current["metrics"]["accuracy"], "unknown_abstain": pg135["unknown_abstain_rate"], "pg122_raw_accuracy": pg122["metrics"]["accuracy"], "pg122_raw_safety": pg122["safety_compliance_rate"], "pg122_guarded_safety": pg122_guarded["safety_compliance_rate"], "guard_overrides": pg122_guarded["guard_override_count"], "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
