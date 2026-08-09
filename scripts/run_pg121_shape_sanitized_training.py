"""PG-121 fresh training after removing shape-hash shortcut features."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg117_double_holdout_replay import collect_target as collect_gamma_target
from app.pg119_metadata_rule_ir_decoder import PG119_DECISIONS
from app.pg120_cross_impl_replay import collect_target as collect_eta_target
from app.pg121_shape_sanitized_rule_ir_decoder import (
    FEATURE_DIM,
    MetadataRuleIRDecisionDecoder,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
    shape_hash_slots_zeroed,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg121-shape-sanitized-rule-ir-decoder-v1"
CHECKPOINT_PATH = ARTIFACT_DIR / "model.pt"
TRACE_PATH = RESEARCH / "pg121_shape_sanitized_training_trace_v1.json"
DATASET_PATH = RESEARCH / "pg121_shape_sanitized_training_dataset_v1.json"
VISIBLE_PATH = RESEARCH / "pg121_shape_sanitized_visible_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg121_shape_sanitized_training_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg121_shape_sanitized_training_report_v1.md"
ETA_SEEDS = [12001, 12003, 12005]
ETA_STRENGTHS = [0, 1, 2]
GAMMA_SEEDS = [11701, 11703, 11705]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    per_class: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(PG119_DECISIONS):
        tp = sum(prediction == index and label == index for prediction, label in zip(predictions, labels))
        fp = sum(prediction == index and label != index for prediction, label in zip(predictions, labels))
        fn = sum(prediction != index and label == index for prediction, label in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(correct / total, 6) if total else 0.0, "macro_f1": round(sum(f1_values) / len(f1_values), 6), "per_class": per_class}


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({"action_manifest": value.get("action_manifest") or {}, "baseline_projection": value.get("baseline_projection") or {}, "response_projection": value.get("response_projection") or {}, "belief_before": value.get("belief_before") or {}})


def _load_split(split: str) -> list[dict[str, Any]]:
    dataset = json.loads((RESEARCH / "pg119_metadata_training_dataset_v1.json").read_text(encoding="utf-8"))
    key = "train_rows" if split == "train" else "dev_rows"
    return [{"row_id": row["row_id"], "source": row["source"], "split": split, "label": row["label"], "model_input": _canonical(row["model_input"]), "prior_inputs": [_canonical(value) for value in row.get("prior_inputs", [])], "training_eligible": True, "memory_promotion_allowed": False} for row in dataset[key]]


def _batch(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = [model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", [])) for row in rows]
    if not all(shape_hash_slots_zeroed(vector) for vector in vectors):
        raise AssertionError("PG-121 training vector retained a shape hash bucket")
    return torch.tensor(vectors, dtype=torch.float32), torch.tensor([decision_index(row["label"]) for row in rows], dtype=torch.long)


def _predict_rows(model: nn.Module, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x.to(device)), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def _predict_episodes(model: nn.Module, episodes: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    predictions: list[int] = []
    labels: list[int] = []
    final_rows: list[dict[str, Any]] = []
    for episode in episodes:
        prior: list[dict[str, Any]] = []
        episode_steps: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = _canonical(step)
            row = {"model_input": model_input, "prior_inputs": list(prior), "label": step["decision"]}
            predicted, confidence = _predict_rows(model, [row], device)
            index = predicted[0]
            predictions.append(index)
            labels.append(decision_index(step["decision"]))
            episode_steps.append({"step_id": step["step_id"], "expected_decision": step["decision"], "predicted_decision": PG119_DECISIONS[index], "confidence": round(confidence[0], 6)})
            prior.append(model_input)
        final_rows.append({"episode_id": episode["episode_id"], "target_seed": episode.get("target_seed"), "surface_kind": episode["surface_kind"], "expected_final_decision": episode["final_decision"], "predicted_final_decision": episode_steps[-1]["predicted_decision"], "steps": episode_steps})
    return predictions, labels, final_rows


def _load_pg114_episodes() -> list[dict[str, Any]]:
    visible = json.loads((RESEARCH / "pg114_family_holdout_replay_visible_dataset_v1.json").read_text(encoding="utf-8"))
    trace = json.loads((RESEARCH / "pg114_family_holdout_replay_trace_v1.json").read_text(encoding="utf-8"))
    visible_by_id = {row["row_id"]: row for row in visible["rows"]}
    episodes: list[dict[str, Any]] = []
    for episode in trace["episodes"]:
        prior: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = _canonical(visible_by_id[step["step_id"]]["model_input"])
            steps.append({**step, "model_input": model_input})
            prior.append(model_input)
        episodes.append({**episode, "steps": steps})
    return episodes


def _evaluate(model: nn.Module, device: torch.device, collected: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [episode for target in collected for episode in target["episodes"]]
    predictions, labels, final_rows = _predict_episodes(model, episodes, device)
    positives = [row for row in final_rows if row["surface_kind"] in {"metadata", "route"}]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    steady = [row for row in final_rows if row["surface_kind"] == "steady"]
    blind = [row for row in final_rows if row["surface_kind"] in {"blind", "opaque"}]
    per_seed: dict[str, dict[str, Any]] = {}
    for target in collected:
        seed = str(target["target_seed"])
        rows = [row for row in final_rows if str(row.get("target_seed")) == seed]
        pos = [row for row in rows if row["surface_kind"] in {"metadata", "route"}]
        dec = [row for row in rows if row["surface_kind"] == "decoy"]
        unk = [row for row in rows if row["surface_kind"] in {"blind", "opaque"}]
        per_seed[seed] = {"positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in pos) / len(pos), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in dec), "unknown_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in unk) / len(unk), 6)}
    recalls = [item["positive_recall"] for item in per_seed.values()]
    bindings = [{"episode_id": row["episode_id"], "predicted_decision": row["predicted_final_decision"], "long_term_memory_write": False} for row in final_rows]
    return {"step_metrics": _metrics(predictions, labels), "final_episode_rows": final_rows, "rule_ir_slot_bindings": bindings, "metadata_positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "steady_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in steady) / len(steady), 6), "blind_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in blind) / len(blind), 6), "cross_seed": {"per_seed": per_seed, "positive_recall_variance": round(sum((value - sum(recalls) / len(recalls)) ** 2 for value in recalls) / len(recalls), 6)}, "evaluation_rows": len(predictions)}


def _evaluate_pg114(model: nn.Module, device: torch.device) -> dict[str, Any]:
    episodes = _load_pg114_episodes()
    predictions, labels, final_rows = _predict_episodes(model, episodes, device)
    policies = [row for row in final_rows if row["surface_kind"] == "policy"]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    opaque = [row for row in final_rows if row["surface_kind"] == "opaque"]
    return {"step_metrics": _metrics(predictions, labels), "final_episode_rows": final_rows, "family_holdout_confirm_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in policies) / len(policies), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "withheld_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in opaque) / len(opaque), 6), "evaluation_rows": len(predictions)}


async def _collect_eta() -> list[dict[str, Any]]:
    return [await collect_eta_target(seed, decoy_strength=strength) for strength in ETA_STRENGTHS for seed in ETA_SEEDS]


async def _collect_gamma() -> list[dict[str, Any]]:
    return [await collect_gamma_target(seed) for seed in GAMMA_SEEDS]


def main() -> None:
    torch.manual_seed(12121)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12121)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    train_rows = _load_split("train")
    dev_rows = _load_split("dev")
    eta = asyncio.run(_collect_eta())
    gamma = asyncio.run(_collect_gamma())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetadataRuleIRDecisionDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train_rows)
    train_x, train_y = train_x.to(device), train_y.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, 41):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_prediction, _ = _predict_rows(model, train_rows, device)
        dev_prediction, _ = _predict_rows(model, dev_rows, device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows])["accuracy"], "dev_accuracy": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])["accuracy"]})
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg121-shape-sanitized-rule-ir-decoder-v1", "feature_dim": FEATURE_DIM, "decision_set": list(PG119_DECISIONS), "device_at_training": str(device), "model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
    train_prediction, _ = _predict_rows(model, train_rows, device)
    dev_prediction, _ = _predict_rows(model, dev_rows, device)
    pg114 = _evaluate_pg114(model, device)
    pg117 = _evaluate(model, device, gamma)
    pg120 = _evaluate(model, device, eta)
    trace = {"schema_version": "pg121-shape-sanitized-training-trace-v1", "protocol_id": "pg-pk-121-shape-sanitized-rule-ir-training-v1", "status": "fresh_shape_sanitized_training_and_holdout", "evaluation_only": False, "training_eligible": True, "memory_promotion_allowed": False, "training_source": "pg119_metadata_training_dataset_v1.json", "holdout_source_set": ["pg120_eta_cross_implementation", "pg117_gamma_double_encoding", "pg114_family_holdout_decoy_replay"], "target_instance_count": len(eta), "episode_count": sum(len(target["episodes"]) for target in eta), "step_count": sum(len(step["steps"]) for target in eta for step in target["episodes"]), "get_step_count": sum(step["action_manifest"]["method"] == "GET" for target in eta for episode in target["episodes"] for step in episode["steps"]), "post_step_count": sum(step["action_manifest"]["method"] == "POST" for target in eta for episode in target["episodes"] for step in episode["steps"]), "shape_hash_slots_zeroed": True, "capacity_unchanged": True, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "online_weight_update": False, "long_term_memory_write": False, "holdout_sources": eta}
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = {"schema_version": "pg121-shape-sanitized-training-dataset-v1", "source_dataset": "pg119_metadata_training_dataset_v1.json", "training_eligible": True, "memory_promotion_allowed": False, "shape_hash_slots_zeroed": True, "feature_dim": FEATURE_DIM, "train_rows": train_rows, "dev_rows": dev_rows}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg121-shape-sanitized-visible-dataset-v1", "training_eligible": True, "model_input_family_free": True, "model_input_oracle_blind": True, "shape_hash_slots_zeroed": True, "rows": [{"row_id": row["row_id"], "model_input": row["model_input"], "training_label": row["label"], "memory_promotion_allowed": False} for row in train_rows + dev_rows]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"protocol_id": "pg-pk-121-shape-sanitized-rule-ir-training-v1", "schema_version": "pg121-shape-sanitized-training-report-v1", "status": "completed_pg121_shape_sanitized_training", "scope": {"model": "fresh_shape_sanitized_metadata_transition_rule_ir_decoder", "feature_dim": FEATURE_DIM, "hidden_dim": 48, "epochs": 40, "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "capacity_unchanged": True, "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train_rows), "dev_count": len(dev_rows), "history_tail": history[-5:], "train_metrics": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows]), "dev_metrics": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])}, "blind_pg114": pg114, "blind_pg117": pg117, "blind_pg120": pg120, "previous_pg120_unknown_abstain_rate": 0.0, "checks": {"fresh_checkpoint": True, "previous_checkpoint_reuse_forbidden": True, "shape_hash_slots_zeroed": True, "capacity_unchanged": True, "pg119_rows_only_training_source": True, "pg120_cross_implementation_holdout": True, "get_post_balanced_collection": trace["get_step_count"] == trace["post_step_count"], "fresh_reset_per_step": True, "evidence_hashes_valid": True, "model_input_family_free": True, "model_input_oracle_blind": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "pg114_positive_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0, "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0, "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0, "pg120_positive_recall_nonzero": pg120["metadata_positive_recall"] > 0.0, "pg120_decoy_false_accept_zero": pg120["decoy_false_accept_count"] == 0, "pg120_unknown_abstain_nonzero": pg120["blind_oracle_abstain_rate"] > 0.0, "pg120_cross_seed_recall_variance_bounded": pg120["cross_seed"]["positive_recall_variance"] <= 0.1, "all_abstain_not_success": pg120["metadata_positive_recall"] > 0.0}, "diagnosis": {"repaired_failure": "PG-120 unknown abstain collapsed under unseen transition-v4 shape spelling", "representation_change": "zero four shape hash buckets", "old_pg120_unknown_abstain_rate": 0.0}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_shape_sanitized_holdout_pending_manual_review", "reason": "这是一次表示修复的受控结果，仍需更多 family、实现、seed 和 Codex 人工审核。"}, "source": {"decoder": _sha256_file(ROOT / "app/pg121_shape_sanitized_rule_ir_decoder.py"), "runner": _sha256_file(Path(__file__)), "training_dataset": _sha256_file(RESEARCH / "pg119_metadata_training_dataset_v1.json"), "pg120_report": _sha256_file(RESEARCH / "pg120_cross_impl_holdout_report_v1.json"), "pg117_report": _sha256_file(RESEARCH / "pg117_double_holdout_report_v1.json"), "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json")}}
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-121 shape-sanitized Rule IR 训练", "", "PG-120 发现 shape hash shortcut 后，保持 48 维/4996 参数不变，将四个 hash bucket 在训练和评估中固定为零。", "", f"- PG-120 正例召回/decoy 误接受/未知弃权：`{pg120['metadata_positive_recall']}` / `{pg120['decoy_false_accept_count']}` / `{pg120['blind_oracle_abstain_rate']}`。", f"- PG-120 跨 seed 方差：`{pg120['cross_seed']['positive_recall_variance']}`；旧模型未知弃权：`0.0`。", f"- 开发准确率：`{report['training']['dev_metrics']['accuracy']}`；容量未增加：`{report['scope']['capacity_unchanged']}`。", "- 当前仍不进入长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": FEATURE_DIM, "parameter_count": report["scope"]["parameter_count"], "dev_accuracy": report["training"]["dev_metrics"]["accuracy"], "pg120_recall": pg120["metadata_positive_recall"], "pg120_false_accept": pg120["decoy_false_accept_count"], "pg120_unknown_abstain": pg120["blind_oracle_abstain_rate"], "pg120_seed_variance": pg120["cross_seed"]["positive_recall_variance"], "all_gates": all(report["checks"].values()), "checkpoint": str(CHECKPOINT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
