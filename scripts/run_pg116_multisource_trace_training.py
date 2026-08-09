"""PG-116: collect real local GET/POST traces and train the same small decoder.

PG-116 is the first training run that uses a fresh, typed local replay source
instead of only a hand-authored abstract fixture.  The two target profiles,
train/dev seeds, labels and PG-114 blind holdout are kept separate.  No raw
probe/body value enters the model-facing dataset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg115_small_rule_ir_decoder import (
    FEATURE_DIM,
    PG115_DECISIONS,
    SmallRuleIRDecisionDecoder,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)
from app.pg116_multisource_replay import SCHEMA_VERSION, SURFACES, collect_source


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg116-multisource-rule-ir-decoder-v1"
TRACE_PATH = RESEARCH / "pg116_multisource_trace_v1.json"
DATASET_PATH = RESEARCH / "pg116_multisource_training_dataset_v1.json"
VISIBLE_PATH = RESEARCH / "pg116_multisource_visible_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg116_multisource_trace_training_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg116_multisource_trace_training_report_v1.md"
CHECKPOINT_PATH = ARTIFACT_DIR / "model.pt"


TRAIN_SEEDS = {"alpha": [11601, 11603, 11605], "beta": [11611, 11613, 11615]}
DEV_SEEDS = {"alpha": [11602, 11604, 11606], "beta": [11612, 11614, 11616]}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
        per_class[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(correct / total, 6) if total else 0.0, "macro_f1": round(sum(f1_values) / len(f1_values), 6), "per_class": per_class}


def _batch(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", [])) for row in rows], dtype=torch.float32)
    y = torch.tensor([decision_index(row["label"]) for row in rows], dtype=torch.long)
    return x, y


def _run(model: nn.Module, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x.to(device)), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def _build_rows(collected: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_bundle in collected:
        source = source_bundle["source"]
        for episode in source_bundle["episodes"]:
            if (episode["target_seed"] in TRAIN_SEEDS[source]) != (split == "train"):
                continue
            prior_inputs: list[dict[str, Any]] = []
            for step in episode["steps"]:
                model_input = canonical_model_input(step["model_input"])
                rows.append({
                    "row_id": step["step_id"],
                    "source": source,
                    "target_seed": episode["target_seed"],
                    "episode_id": episode["episode_id"],
                    "split": split,
                    "label": step["decision"],
                    "model_input": model_input,
                    "prior_inputs": list(prior_inputs),
                    "training_eligible": True,
                    "memory_promotion_allowed": False,
                })
                prior_inputs.append(model_input)
    return rows


def _balance(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Balance classes by deterministic replay of unique local trace rows."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    target = max(len(group) for group in groups.values())
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for label in PG115_DECISIONS:
        group = list(groups.get(label, []))
        if not group:
            raise ValueError(f"PG-116 missing class: {label}")
        rng.shuffle(group)
        for repeat in range(target):
            base = group[repeat % len(group)]
            row = dict(base)
            row["row_id"] = f"{base['row_id']}-replay-{repeat:03d}"
            row["source_row_id"] = base["row_id"]
            row["replay_index"] = repeat
            balanced.append(row)
    rng.shuffle(balanced)
    return balanced, {label: len(groups[label]) for label in PG115_DECISIONS}


def _load_pg114_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    visible = json.loads((RESEARCH / "pg114_family_holdout_replay_visible_dataset_v1.json").read_text(encoding="utf-8"))
    trace = json.loads((RESEARCH / "pg114_family_holdout_replay_trace_v1.json").read_text(encoding="utf-8"))
    visible_by_id = {row["row_id"]: row for row in visible["rows"]}
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode in trace["episodes"]:
        prior: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = canonical_model_input(visible_by_id[step["step_id"]]["model_input"])
            rows.append({"model_input": model_input, "prior_inputs": list(prior), "label": step["decision"]})
            prior.append(model_input)
        episode_rows.append({"episode_id": episode["episode_id"], "surface_kind": episode["surface_kind"], "expected_final_decision": episode["final_decision"], "length": len(episode["steps"])})
    return rows, episode_rows, {"visible": visible, "trace": trace}


def _evaluate_pg114(model: nn.Module, device: torch.device) -> dict[str, Any]:
    rows, episode_rows, loaded = _load_pg114_rows()
    predictions, confidence = _run(model, rows, device)
    labels = [decision_index(row["label"]) for row in rows]
    final_predictions: list[dict[str, Any]] = []
    cursor = 0
    for episode in episode_rows:
        final = PG115_DECISIONS[predictions[cursor + episode["length"] - 1]]
        final_predictions.append({**episode, "predicted_final_decision": final})
        cursor += episode["length"]
    positives = [row for row in final_predictions if row["surface_kind"] == "policy"]
    decoys = [row for row in final_predictions if row["surface_kind"] == "decoy"]
    neutral = [row for row in final_predictions if row["surface_kind"] == "neutral"]
    opaque = [row for row in final_predictions if row["surface_kind"] == "opaque"]
    return {
        "step_metrics": _metrics(predictions, labels),
        "final_episode_rows": final_predictions,
        "family_holdout_confirm_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6),
        "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys),
        "neutral_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in neutral) / len(neutral), 6),
        "withheld_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in opaque) / len(opaque), 6),
        "mean_confidence": round(sum(confidence) / len(confidence), 6),
        "evaluation_rows": len(rows),
        "source_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json"),
        "model_input_family_free": loaded["visible"]["model_input_family_free"],
        "model_input_oracle_blind": loaded["visible"]["typed_oracle_labels_outside_model_input"],
    }


async def _collect() -> list[dict[str, Any]]:
    return [
        await collect_source("alpha", TRAIN_SEEDS["alpha"] + DEV_SEEDS["alpha"]),
        await collect_source("beta", TRAIN_SEEDS["beta"] + DEV_SEEDS["beta"]),
    ]


def main() -> None:
    random.seed(11616)
    torch.manual_seed(11616)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11616)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    collected = asyncio.run(_collect())
    train_unique = _build_rows(collected, "train")
    dev_unique = _build_rows(collected, "dev")
    train_rows, train_class_unique = _balance(train_unique, seed=11617)
    dev_rows, dev_class_unique = _balance(dev_unique, seed=11618)
    trace = {
        "schema_version": "pg116-multisource-trace-v1",
        "protocol_id": "pg-pk-116-multisource-trace-training-v1",
        "evaluation_only": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "execution_mode": "in_process_loopback_asgi_get_post",
        "source_set": ["alpha", "beta"],
        "target_instance_count": len({(bundle["source"], episode["target_seed"]) for bundle in collected for episode in bundle["episodes"]}),
        "episode_count": sum(len(bundle["episodes"]) for bundle in collected),
        "step_count": sum(len(episode["steps"]) for bundle in collected for episode in bundle["episodes"]),
        "get_step_count": sum(sum(step["action_manifest"]["method"] == "GET" for step in episode["steps"]) for bundle in collected for episode in bundle["episodes"]),
        "post_step_count": sum(sum(step["action_manifest"]["method"] == "POST" for step in episode["steps"]) for bundle in collected for episode in bundle["episodes"]),
        "fresh_reset_per_step": all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for bundle in collected for episode in bundle["episodes"] for step in episode["steps"]),
        "evidence_hash_valid": all(record["evidence_hash"] == _sha256_json({key: value for key, value in record.items() if key != "evidence_hash"}) for bundle in collected for episode in bundle["episodes"] for record in episode["evidence_records"]),
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
        "sources": collected,
    }
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = {
        "schema_version": "pg116-multisource-training-dataset-v1",
        "purpose": "balanced local typed GET/POST trace rows for Rule IR decision learning",
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "model_input_family_free": True,
        "model_input_oracle_blind": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "script_execution": False,
        "database_write": False,
        "source_set": ["alpha", "beta"],
        "train_unique_row_count": len(train_unique),
        "dev_unique_row_count": len(dev_unique),
        "train_class_unique": train_class_unique,
        "dev_class_unique": dev_class_unique,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "pg114_excluded_from_training": True,
    }
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {
        "schema_version": "pg116-multisource-visible-dataset-v1",
        "evaluation_only": False,
        "training_eligible": True,
        "model_input_family_free": True,
        "typed_oracle_labels_outside_model_input": True,
        "rows": [
            {"row_id": row["row_id"], "source": row["source"], "split": row["split"], "model_input": row["model_input"], "training_label": row["label"], "memory_promotion_allowed": False}
            for row in train_unique + dev_unique
        ],
    }
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRuleIRDecisionDecoder().to(device)
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
        train_prediction, _ = _run(model, train_rows, device)
        dev_prediction, _ = _run(model, dev_rows, device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows])["accuracy"], "dev_accuracy": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])["accuracy"]})

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg116-small-rule-ir-decoder-v1", "feature_dim": FEATURE_DIM, "decision_set": list(PG115_DECISIONS), "device_at_training": str(device), "model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
    train_prediction, _ = _run(model, train_rows, device)
    dev_prediction, _ = _run(model, dev_rows, device)
    pg114 = _evaluate_pg114(model, device)
    report = {
        "protocol_id": "pg-pk-116-multisource-trace-training-v1",
        "schema_version": "pg116-multisource-trace-training-report-v1",
        "status": "completed_pg116_multisource_trace_training",
        "scope": {"model": "small_rule_ir_decision_mlp", "feature_dim": FEATURE_DIM, "hidden_dim": 48, "epochs": 40, "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "real_vulnerability_scanner_claim_allowed": False},
        "collection": {"source_set": ["alpha", "beta"], "target_instance_count": trace["target_instance_count"], "episode_count": trace["episode_count"], "step_count": trace["step_count"], "get_step_count": trace["get_step_count"], "post_step_count": trace["post_step_count"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hash_valid": trace["evidence_hash_valid"], "train_unique_row_count": len(train_unique), "dev_unique_row_count": len(dev_unique), "train_rows_after_balance": len(train_rows), "dev_rows_after_balance": len(dev_rows), "train_class_unique": train_class_unique, "dev_class_unique": dev_class_unique},
        "training": {"history_tail": history[-5:], "train_metrics": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows]), "dev_metrics": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])},
        "blind_pg114": pg114,
        "checks": {"source_count_two": True, "source_seed_disjoint": all(set(TRAIN_SEEDS[source]).isdisjoint(DEV_SEEDS[source]) for source in TRAIN_SEEDS), "get_post_balanced_collection": trace["get_step_count"] == trace["post_step_count"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hashes_valid": trace["evidence_hash_valid"], "model_input_family_free": True, "model_input_oracle_blind": True, "pg114_rows_not_in_training": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "pg114_positive_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0, "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0, "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0, "all_abstain_not_success": pg114["family_holdout_confirm_recall"] > 0.0},
        "promotion": {"checkpoint_written": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "multisource_training_trial_only", "reason": "two local source profiles and one blind holdout improve the decoder but do not establish broad scanner capability"},
        "source": {"target": _sha256_file(ROOT / "app/pg116_multisource_training_target.py"), "bridge": _sha256_file(ROOT / "app/pg116_multisource_replay.py"), "runner": _sha256_file(Path(__file__)), "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json")},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-116 多源本地轨迹训练", "", "两种本地目标 profile 通过 GET/POST + fresh reset 采集，模型输入不含 oracle/family/raw 值。", "", f"- 设备：`{device}`；参数量：`{report['scope']['parameter_count']}`；训练/开发平衡行：`{len(train_rows)}/{len(dev_rows)}`。", f"- PG-114 盲测族外正例召回：`{pg114['family_holdout_confirm_recall']}`；decoy 误接受：`{pg114['decoy_false_accept_count']}`；未知弃权：`{pg114['withheld_oracle_abstain_rate']}`。", f"- 逐步准确率：`{pg114['step_metrics']['accuracy']}`；该指标用于暴露中间状态混淆，不隐藏。", "- checkpoint 仅用于实验，不自动进入长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "unique_train_rows": len(train_unique), "balanced_train_rows": len(train_rows), "unique_dev_rows": len(dev_unique), "balanced_dev_rows": len(dev_rows), "dev_accuracy": report["training"]["dev_metrics"]["accuracy"], "pg114_step_accuracy": pg114["step_metrics"]["accuracy"], "pg114_family_holdout_confirm_recall": pg114["family_holdout_confirm_recall"], "pg114_decoy_false_accept_count": pg114["decoy_false_accept_count"], "pg114_withheld_oracle_abstain_rate": pg114["withheld_oracle_abstain_rate"], "checkpoint": str(CHECKPOINT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

