"""PG-117: frozen PG-116 decoder on an independent double-encoding target.

This is a strict evaluation-only run.  The gamma target is a separate Python
implementation and its ``[url_percent, html_entity]`` encoding chain was not
present in PG-116 training (which used ``[identity]``).  The runner therefore
does not mutate weights, does not create training rows, and cannot promote
anything to long-term memory.  Its purpose is to expose the representation
gap before adding a new double-encoding training source.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg115_small_rule_ir_decoder import (
    PG115_DECISIONS,
    SmallRuleIRDecisionDecoder,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)
from app.pg117_double_holdout_replay import ENCODING_CHAIN, collect_target


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
TRACE_PATH = RESEARCH / "pg117_double_holdout_trace_v1.json"
VISIBLE_PATH = RESEARCH / "pg117_double_holdout_visible_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg117_double_holdout_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg117_double_holdout_report_v1.md"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg116-multisource-rule-ir-decoder-v1" / "model.pt"

TARGET_SEEDS = [11701, 11703, 11705]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
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
        per_class[name] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "count": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_class": per_class,
    }


def _load_frozen_decoder() -> tuple[nn.Module, dict[str, Any]]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"missing frozen PG-116 checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = SmallRuleIRDecisionDecoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _predict_episode(model: nn.Module, episode: dict[str, Any]) -> dict[str, Any]:
    prior_inputs: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for step in episode["steps"]:
        model_input = canonical_model_input(step["model_input"])
        vector = torch.tensor(
            [model_input_feature_vector(model_input, prior_inputs=prior_inputs)],
            dtype=torch.float32,
        )
        with torch.inference_mode():
            probabilities = torch.softmax(model(vector), dim=-1)[0]
        confidence, index = probabilities.max(dim=-1)
        decision = PG115_DECISIONS[int(index)]
        predictions.append(
            {
                "step_id": step["step_id"],
                "expected_decision": step["decision"],
                "predicted_decision": decision,
                "confidence": round(float(confidence), 6),
            }
        )
        prior_inputs.append(model_input)
    return {
        "episode_id": episode["episode_id"],
        "surface_kind": episode["surface_kind"],
        "expected_final_decision": episode["final_decision"],
        "predicted_final_decision": predictions[-1]["predicted_decision"],
        "steps": predictions,
    }


async def _collect() -> list[dict[str, Any]]:
    return [await collect_target(seed) for seed in TARGET_SEEDS]


def main() -> None:
    if list(ENCODING_CHAIN) != ["url_percent", "html_entity"]:
        raise RuntimeError("PG-117 encoding chain changed; update the preregistered holdout")
    collected = asyncio.run(_collect())
    episodes = [episode for target in collected for episode in target["episodes"]]
    steps = [step for episode in episodes for step in episode["steps"]]

    trace = {
        "schema_version": "pg117-double-holdout-trace-v1",
        "protocol_id": "pg-pk-117-double-implementation-encoding-holdout-v1",
        "status": "evaluation_only_trace_collected",
        "evaluation_only": True,
        "training_eligible": False,
        "memory_promotion_allowed": False,
        "execution_mode": "in_process_loopback_asgi_get_post",
        "target_implementation": "app/pg117_gamma_target.py:create_app",
        "target_seeds": TARGET_SEEDS,
        "encoding_chain": list(ENCODING_CHAIN),
        "implementation_holdout_from_pg116": True,
        "encoding_holdout_from_pg116": True,
        "target_instance_count": len(collected),
        "episode_count": len(episodes),
        "step_count": len(steps),
        "get_step_count": sum(step["action_manifest"]["method"] == "GET" for step in steps),
        "post_step_count": sum(step["action_manifest"]["method"] == "POST" for step in steps),
        "fresh_reset_per_step": all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for step in steps),
        "evidence_hash_valid": all(
            record["evidence_hash"] == _sha256_json({key: value for key, value in record.items() if key != "evidence_hash"})
            for episode in episodes
            for record in episode["evidence_records"]
        ),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
        "sources": collected,
    }
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    visible_rows = [
        {
            "row_id": step["step_id"],
            "episode_id": episode["episode_id"],
            "target_seed": episode["target_seed"],
            "surface_kind": episode["surface_kind"],
            "model_input": canonical_model_input(step["model_input"]),
            "evaluation_label": step["decision"],
            "memory_promotion_allowed": False,
        }
        for episode in episodes
        for step in episode["steps"]
    ]
    visible = {
        "schema_version": "pg117-double-holdout-visible-dataset-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "model_input_family_free": True,
        "typed_oracle_labels_outside_model_input": True,
        "encoding_chain": list(ENCODING_CHAIN),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "rows": visible_rows,
    }
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model, checkpoint = _load_frozen_decoder()
    episode_predictions = [_predict_episode(model, episode) for episode in episodes]
    predicted = [decision_index(row["predicted_decision"]) for row in [step for episode in episode_predictions for step in episode["steps"]]]
    labels = [decision_index(step["decision"]) for step in steps]
    final_route = [row for row in episode_predictions if row["surface_kind"] == "route"]
    final_decoy = [row for row in episode_predictions if row["surface_kind"] == "decoy"]
    final_steady = [row for row in episode_predictions if row["surface_kind"] == "steady"]
    final_blind = [row for row in episode_predictions if row["surface_kind"] == "blind"]
    final_confidences = [row["steps"][-1]["confidence"] for row in episode_predictions]
    report = {
        "protocol_id": "pg-pk-117-double-implementation-encoding-holdout-v1",
        "schema_version": "pg117-double-holdout-report-v1",
        "status": "completed_pg117_double_holdout_evaluation",
        "scope": {
            "model": "frozen_pg116_small_rule_ir_decision_mlp",
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_schema": checkpoint.get("schema_version"),
            "device": "cpu_evaluation",
            "real_vulnerability_scanner_claim_allowed": False,
        },
        "collection": {
            "target_implementation": "pg117_gamma_independent_target",
            "target_instance_count": trace["target_instance_count"],
            "target_seeds": TARGET_SEEDS,
            "episode_count": trace["episode_count"],
            "step_count": trace["step_count"],
            "get_step_count": trace["get_step_count"],
            "post_step_count": trace["post_step_count"],
            "encoding_chain": list(ENCODING_CHAIN),
            "fresh_reset_per_step": trace["fresh_reset_per_step"],
            "evidence_hash_valid": trace["evidence_hash_valid"],
        },
        "evaluation": {
            "step_metrics": _metrics(predicted, labels),
            "final_episode_predictions": episode_predictions,
            "route_positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in final_route) / len(final_route), 6),
            "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in final_decoy),
            "steady_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in final_steady) / len(final_steady), 6),
            "blind_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in final_blind) / len(final_blind), 6),
            "mean_final_confidence": round(sum(final_confidences) / len(final_confidences), 6),
        },
        "checks": {
            "independent_implementation": trace["implementation_holdout_from_pg116"],
            "double_encoding_holdout": trace["encoding_holdout_from_pg116"],
            "get_post_balanced": trace["get_step_count"] == trace["post_step_count"],
            "fresh_reset_per_step": trace["fresh_reset_per_step"],
            "evidence_hashes_valid": trace["evidence_hash_valid"],
            "model_input_family_free": visible["model_input_family_free"],
            "model_input_oracle_blind": visible["typed_oracle_labels_outside_model_input"],
            "no_raw_probe_strings": visible["raw_probe_strings_stored"] is False,
            "no_raw_response_bodies": visible["raw_response_bodies_stored"] is False,
            "decoy_false_accept_zero": sum(row["predicted_final_decision"] == "confirmed_positive" for row in final_decoy) == 0,
            "blind_strict_abstain": all(row["predicted_final_decision"] == "abstain" for row in final_blind),
            "positive_recall_nonzero": any(row["predicted_final_decision"] == "confirmed_positive" for row in final_route),
            "all_abstain_not_success": any(row["predicted_final_decision"] == "confirmed_positive" for row in final_route),
        },
        "diagnosis": {
            "representation_gap": "location_changed is present in target evidence but is not encoded in PG-115/116 canonical model input",
            "training_data_added": False,
            "recommended_repair": "add a family-free transition-delta slot, collect double-encoding train/dev pairs, then replay this frozen gamma holdout",
        },
        "promotion": {
            "training_artifact_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "status": "blocked_pg117_holdout_capability_gate" if not any(row["predicted_final_decision"] == "confirmed_positive" for row in final_route) else "holdout_diagnostic_only",
            "reason": "PG-117 is a double implementation/encoding holdout; frozen failure must be repaired before any new memory promotion",
        },
        "source": {
            "target": _sha256_file(ROOT / "app/pg117_gamma_target.py"),
            "bridge": _sha256_file(ROOT / "app/pg117_double_holdout_replay.py"),
            "runner": _sha256_file(Path(__file__)),
            "pg116_checkpoint": _sha256_file(CHECKPOINT_PATH),
            "pg116_report": _sha256_file(RESEARCH / "pg116_multisource_trace_training_report_v1.json"),
        },
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    route_recall = report["evaluation"]["route_positive_recall"]
    MARKDOWN_PATH.write_text(
        "\n".join(
            [
                "# PG-117 双重实现/编码保持出",
                "",
                "PG-117 用独立 gamma 实现和 `[url_percent, html_entity]` 双重编码链，冻结 PG-116 checkpoint 进行族外评估。",
                "",
                f"- 目标实例/episode/step：`{trace['target_instance_count']}/{trace['episode_count']}/{trace['step_count']}`；GET/POST：`{trace['get_step_count']}/{trace['post_step_count']}`。",
                f"- 族外 route 正例最终召回：`{route_recall}`；decoy 误接受：`{report['evaluation']['decoy_false_accept_count']}`；blind 弃权：`{report['evaluation']['blind_oracle_abstain_rate']}`。",
                f"- 逐步准确率：`{report['evaluation']['step_metrics']['accuracy']}`；宏 F1：`{report['evaluation']['step_metrics']['macro_f1']}`。",
                "- 结论：这是评估失败/能力诊断，不生成训练样本，不提升长期记忆。缺口指向未抽象的通用 transition-delta 特征，而不是继续记忆目标表面。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "target_instance_count": trace["target_instance_count"],
        "episode_count": trace["episode_count"],
        "step_count": trace["step_count"],
        "get_step_count": trace["get_step_count"],
        "post_step_count": trace["post_step_count"],
        "route_positive_recall": route_recall,
        "decoy_false_accept_count": report["evaluation"]["decoy_false_accept_count"],
        "blind_oracle_abstain_rate": report["evaluation"]["blind_oracle_abstain_rate"],
        "step_accuracy": report["evaluation"]["step_metrics"]["accuracy"],
        "promotion": report["promotion"]["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
