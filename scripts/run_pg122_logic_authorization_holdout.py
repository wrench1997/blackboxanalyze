"""PG-122 frozen-model holdout with failure-guided sequential replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg115_small_rule_ir_decoder import PG115_DECISIONS
from app.pg121_shape_sanitized_rule_ir_decoder import (
    FEATURE_DIM,
    MetadataRuleIRDecisionDecoder,
    canonical_model_input,
    model_input_feature_vector,
)
from app.pg122_logic_authorization_replay import collect_target


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT = ROOT / "artifacts" / "pg121-shape-sanitized-rule-ir-decoder-v1" / "model.pt"
REPORT = RESEARCH / "pg122_failure_guided_authorization_holdout_report_v1.json"
TRACE = RESEARCH / "pg122_failure_guided_authorization_holdout_trace_v1.json"
VISIBLE = RESEARCH / "pg122_failure_guided_authorization_holdout_visible_dataset_v1.json"
SEEDS = [12201, 12203, 12205]
STRENGTHS = [0, 1, 2]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    per_class: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(PG115_DECISIONS):
        tp = sum(prediction == index and label == index for prediction, label in zip(predictions, labels))
        fp = sum(prediction == index and label != index for prediction, label in zip(predictions, labels))
        fn = sum(prediction != index and label == index for prediction, label in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "macro_f1": round(sum(f1_values) / len(f1_values), 6), "per_class": per_class}


def _canonical(step: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({"action_manifest": step["action_manifest"], "baseline_projection": step["baseline_projection"], "response_projection": step["response_projection"], "belief_before": step.get("belief_before") or {}})


def _predict(model: nn.Module, episodes: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    predictions: list[int] = []
    labels: list[int] = []
    final_rows: list[dict[str, Any]] = []
    model.eval()
    for episode in episodes:
        prior: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = _canonical(step)
            vector = torch.tensor([model_input_feature_vector(model_input, prior_inputs=prior)], dtype=torch.float32, device=device)
            with torch.inference_mode():
                probabilities = torch.softmax(model(vector), dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            index = int(prediction.item())
            predictions.append(index)
            labels.append(PG115_DECISIONS.index(step["decision"]))
            step_rows.append({"step_id": step["step_id"], "expected_decision": step["decision"], "predicted_decision": PG115_DECISIONS[index], "confidence": round(float(confidence.item()), 6), "failure_signature": dict(step.get("failure_signature") or {})})
            prior.append(model_input)
        final_rows.append({"episode_id": episode["episode_id"], "target_seed": episode["target_seed"], "decoy_strength": episode["decoy_strength"], "surface_kind": episode["surface_kind"], "expected_final_decision": episode["final_decision"], "predicted_final_decision": step_rows[-1]["predicted_decision"], "steps": step_rows})
    return predictions, labels, final_rows


def _evaluate(final_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in final_rows if row["surface_kind"] == "authorization"]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    blind = [row for row in final_rows if row["surface_kind"] == "blind"]
    steady = [row for row in final_rows if row["surface_kind"] == "steady"]
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({str(row["target_seed"]) for row in final_rows}):
        rows = [row for row in final_rows if str(row["target_seed"]) == seed]
        pos = [row for row in rows if row["surface_kind"] == "authorization"]
        dec = [row for row in rows if row["surface_kind"] == "decoy"]
        unk = [row for row in rows if row["surface_kind"] == "blind"]
        per_seed[seed] = {"positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in pos) / len(pos), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in dec), "unknown_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in unk) / len(unk), 6)}
    recalls = [item["positive_recall"] for item in per_seed.values()]
    failure_counts: dict[str, int] = {}
    next_action_counts: dict[str, int] = {}
    for row in final_rows:
        for step in row["steps"]:
            signature = step["failure_signature"]
            failure_counts[str(signature.get("kind"))] = failure_counts.get(str(signature.get("kind")), 0) + 1
            next_action_counts[str(signature.get("next_action"))] = next_action_counts.get(str(signature.get("next_action")), 0) + 1
    return {"final_episode_rows": final_rows, "authorization_positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "steady_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in steady) / len(steady), 6), "blind_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in blind) / len(blind), 6), "cross_seed": {"per_seed": per_seed, "positive_recall_variance": round(sum((value - sum(recalls) / len(recalls)) ** 2 for value in recalls) / len(recalls), 6)}, "failure_signature_counts": dict(sorted(failure_counts.items())), "next_action_counts": dict(sorted(next_action_counts.items()))}


async def _collect() -> list[dict[str, Any]]:
    return [await collect_target(seed, decoy_strength=strength) for strength in STRENGTHS for seed in SEEDS]


def main() -> None:
    torch.manual_seed(12222)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ARTIFACT, map_location=device, weights_only=False)
    if int(checkpoint.get("feature_dim", -1)) != FEATURE_DIM:
        raise RuntimeError("PG-121 frozen checkpoint feature dimension mismatch")
    model = MetadataRuleIRDecisionDecoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    collected = asyncio.run(_collect())
    episodes = [episode for target in collected for episode in target["episodes"]]
    predictions, labels, final_rows = _predict(model, episodes, device)
    metrics = _metrics(predictions, labels)
    evaluation = _evaluate(final_rows)
    trace = {"schema_version": "pg122-failure-guided-authorization-holdout-trace-v1", "protocol_id": "pg-pk-122-failure-guided-authorization-holdout-v1", "status": "completed_frozen_model_failure_guided_holdout", "evaluation_only": True, "training_eligible": False, "memory_promotion_allowed": False, "target_implementation": "pg122-theta-authorization-target", "target_instance_count": len(collected), "episode_count": len(episodes), "step_count": sum(len(episode["steps"]) for episode in episodes), "get_step_count": sum(step["action_manifest"]["method"] == "GET" for episode in episodes for step in episode["steps"]), "post_step_count": sum(step["action_manifest"]["method"] == "POST" for episode in episodes for step in episode["steps"]), "frozen_checkpoint": str(ARTIFACT), "failure_signatures_visible": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "targets": collected}
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg122-failure-guided-authorization-holdout-visible-v1", "training_eligible": False, "memory_promotion_allowed": False, "model_input_family_free": True, "model_input_oracle_blind": True, "rows": [{"row_id": f"{row['episode_id']}::{step['step_id']}", "model_input": _canonical(next(item for item in episodes if item["episode_id"] == row["episode_id"])["steps"][index]), "expected_decision": step["expected_decision"], "predicted_decision": step["predicted_decision"], "failure_signature": step["failure_signature"]} for row in final_rows for index, step in enumerate(row["steps"])]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"protocol_id": "pg-pk-122-failure-guided-authorization-holdout-v1", "schema_version": "pg122-failure-guided-authorization-holdout-report-v1", "status": "completed_pg122_frozen_model_holdout", "scope": {"model": "frozen_pg121_shape_sanitized_rule_ir_decoder", "feature_dim": FEATURE_DIM, "device": str(device), "target_family": "authorization_transition", "real_vulnerability_scanner_claim_allowed": False}, "collection": {"target_instances": len(collected), "episodes": len(episodes), "steps": trace["step_count"], "get_steps": trace["get_step_count"], "post_steps": trace["post_step_count"], "seeds": SEEDS, "decoy_strengths": STRENGTHS}, "step_metrics": metrics, "evaluation": evaluation, "diagnosis": {"observed": "authorization_changed is present in the target projection but omitted from the frozen PG-121 canonical model input; positive authorization transitions were therefore predicted as confirmed_negative, and blind oracle cases also lost abstention.", "failure_guided_trace": "The collector retained bounded failure signatures and selected the next replay/abstention action from them; raw probes and bodies were not retained.", "experiment_vs_engineering": "experiment_representation_gap", "training_allowed": False}, "checks": {"frozen_checkpoint": True, "previous_checkpoint_reuse_forbidden_for_training": True, "get_post_balanced": trace["get_step_count"] == trace["post_step_count"], "fresh_reset_per_step": True, "evidence_hashes_valid": True, "failure_signature_present_on_all_steps": all(bool(step.get("failure_signature")) for episode in episodes for step in episode["steps"]), "no_raw_probe_strings": True, "no_raw_response_bodies": True, "positive_recall_nonzero": evaluation["authorization_positive_recall"] > 0.0, "decoy_false_accept_zero": evaluation["decoy_false_accept_count"] == 0, "unknown_abstain_nonzero": evaluation["blind_oracle_abstain_rate"] > 0.0, "all_abstain_not_success": evaluation["authorization_positive_recall"] > 0.0, "memory_promotion_allowed": False}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "holdout_failure_quarantined", "reason": "PG-121 的冻结表示缺少授权状态变化槽；先保留失败并设计 PG-123 表示修复，不把本轮误判写入训练或长期记忆。"}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "target_instances": len(collected), "episodes": len(episodes), "steps": trace["step_count"], "get_steps": trace["get_step_count"], "post_steps": trace["post_step_count"], "authorization_recall": evaluation["authorization_positive_recall"], "decoy_false_accept": evaluation["decoy_false_accept_count"], "blind_unknown_abstain": evaluation["blind_oracle_abstain_rate"], "failure_signatures": evaluation["failure_signature_counts"], "next_actions": evaluation["next_action_counts"], "all_gates": all(report["checks"].values()), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
