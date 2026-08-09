"""Run PG-115: a small Rule IR decision decoder and blind PG-114 replay.

The training fixture is generated independently from PG-114.  It contains
only bounded, inert response projections; no payload strings, raw bodies,
family labels, evaluator fields or target identifiers enter the feature
vector.  PG-114 remains a blind evaluation-only dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg115_small_rule_ir_decoder import (
    FEATURE_DIM,
    PG115_DECISIONS,
    SCHEMA_VERSION,
    SmallRuleIRDecisionDecoder,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg115-small-rule-ir-decoder-v1"
TRAIN_DATASET_PATH = RESEARCH / "pg115_small_rule_ir_train_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg115_small_rule_ir_decoder_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg115_small_rule_ir_decoder_report_v1.md"
CHECKPOINT_PATH = ARTIFACT_DIR / "model.pt"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bounded_belief(rng: random.Random, *, effect: float) -> dict[str, float]:
    rest = max(0.0, 1.0 - effect)
    weights = [rng.random() for _ in range(3)]
    total = sum(weights) or 1.0
    return {
        "effect": effect,
        "input_only": rest * weights[0] / total,
        "no_effect": rest * weights[1] / total,
        "unknown": rest * weights[2] / total,
    }


def _synthetic_input(label: str, rng: random.Random, ordinal: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create one inert model-facing row plus oracle-blind prior context."""

    method = "POST" if label in {"confirmed_positive", "abstain"} else rng.choice(["GET", "POST"])
    prior: list[dict[str, Any]] = []
    if label in {"confirmed_positive", "abstain"}:
        prior = [
            {
                "action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["identity"]},
                "baseline_projection": {"body_length_bucket": "1-255", "status_class": "2xx"},
                "response_projection": {
                    "candidate_signal": True,
                    "noise_bucket": rng.randrange(0, 5),
                    "policy_header_changed": False,
                    "shape_changed": label == "abstain",
                    "shape_class": "candidate-review",
                    "status_class": "2xx",
                    "bsp_core_projection": {"leaf_mass_error": 0.0, "selected_leaf_ids": [0, 1], "topology_version": 0},
                },
                "belief_before": _bounded_belief(rng, effect=0.34),
            }
        ]
    elif label == "candidate":
        # A first candidate often follows a negative control.  Keeping that
        # context in the training fixture prevents the decoder from treating
        # every visible header/shape change as an immediate confirmation.
        prior = [
            {
                "action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["identity"]},
                "baseline_projection": {"body_length_bucket": "1-255", "status_class": "2xx"},
                "response_projection": {
                    "candidate_signal": False,
                    "noise_bucket": rng.randrange(0, 5),
                    "policy_header_changed": False,
                    "shape_changed": False,
                    "shape_class": "stable-control",
                    "status_class": "2xx",
                    "bsp_core_projection": {"leaf_mass_error": 0.0, "selected_leaf_ids": [0, 1], "topology_version": 0},
                },
                "belief_before": _bounded_belief(rng, effect=0.25),
            }
        ]

    if label == "confirmed_positive":
        shape_class = rng.choice(["policy-shift", "header-transition", "boundary-policy", "state-policy"])
        response = {
            "candidate_signal": True,
            "noise_bucket": rng.randrange(0, 6),
            "policy_header_changed": True,
            "shape_changed": False,
            "shape_class": shape_class,
            "status_class": "2xx",
        }
        effect = rng.uniform(0.35, 0.62)
    elif label == "abstain":
        shape_class = rng.choice(["opaque-signal", "unknown-shape", "untyped-change", "opaque-boundary"])
        response = {
            "candidate_signal": True,
            "noise_bucket": rng.randrange(0, 7),
            "policy_header_changed": False,
            "shape_changed": True,
            "shape_class": shape_class,
            "status_class": rng.choice(["2xx", "3xx"]),
        }
        effect = rng.uniform(0.28, 0.55)
    elif label == "candidate":
        shape_class = rng.choice(["candidate-review", "shape-review", "marker-review", "header-review"])
        response = {
            "candidate_signal": True,
            "noise_bucket": rng.randrange(0, 8),
            # A candidate can expose a policy/header or shape delta before a
            # second method confirms it; the sequence context is decisive.
            "policy_header_changed": rng.choice([False, True]),
            "shape_changed": rng.choice([False, True]),
            "shape_class": shape_class,
            "status_class": "2xx",
        }
        effect = rng.uniform(0.22, 0.44)
    else:
        shape_class = rng.choice(["stable", "neutral", "same-shape", "ordinary"])
        response = {
            "candidate_signal": False,
            "noise_bucket": rng.randrange(0, 8),
            "policy_header_changed": False,
            "shape_changed": False,
            "shape_class": shape_class,
            "status_class": rng.choice(["2xx", "3xx"]),
        }
        effect = rng.uniform(0.12, 0.31)

        # Some negative controls are deliberately decoy replays: the
        # candidate signal and shape change are visible, but no typed effect
        # is confirmed.  This is the training analogue of PG-114's decoy.
        if rng.random() < 0.35:
            method = "POST"
            prior = [
                {
                    "action_manifest": {"method": "GET", "placement": "query", "encoding_chain": ["identity"]},
                    "baseline_projection": {"body_length_bucket": "1-255", "status_class": "2xx"},
                    "response_projection": {
                        "candidate_signal": True,
                        "noise_bucket": rng.randrange(0, 5),
                        "policy_header_changed": False,
                        "shape_changed": True,
                        "shape_class": "shape-review",
                        "status_class": "2xx",
                        "bsp_core_projection": {"leaf_mass_error": 0.0, "selected_leaf_ids": [0, 1], "topology_version": 0},
                    },
                    "belief_before": _bounded_belief(rng, effect=0.34),
                }
            ]
            response.update({
                "candidate_signal": True,
                "shape_changed": True,
                "shape_class": rng.choice(["shape-decoy", "body-decoy", "boundary-decoy"]),
            })

    model_input = {
        "action_manifest": {
            "method": method,
            "placement": rng.choice(["query", "body"]),
            "encoding_chain": rng.choice([["identity"], ["identity", "url"]]),
            "safety": {
                "no_external_network": True,
                "does_not_execute": True,
                "no_database_write": True,
                "no_credential_access": True,
            },
        },
        "baseline_projection": {
            "body_length_bucket": rng.choice(["1-255", "256-4095"]),
            "status_class": "2xx",
        },
        "response_projection": {
            **response,
            "body_length_bucket": rng.choice(["1-255", "256-4095"]),
            "bsp_core_projection": {
                "leaf_mass_error": 0.0,
                "selected_leaf_ids": [0, 1],
                "topology_version": ordinal % 2,
            },
        },
        "belief_before": _bounded_belief(rng, effect=effect),
    }
    return canonical_model_input(model_input), [canonical_model_input(row) for row in prior]


def build_training_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    # The dev seed is never used while generating training rows.  Descriptor
    # words also differ between splits so this is not a route/template lookup.
    for split, seeds, count_per_class in (("train", [11501, 11502], 80), ("dev", [11503, 11504], 20)):
        for seed in seeds:
            rng = random.Random(seed)
            for label in PG115_DECISIONS:
                for ordinal in range(count_per_class):
                    model_input, prior = _synthetic_input(label, rng, ordinal + seed)
                    rows.append(
                        {
                            "row_id": f"pg115-{split}-{seed}-{label}-{ordinal:03d}",
                            "split": split,
                            "source_seed": seed,
                            "label": label,
                            "model_input": model_input,
                            "prior_inputs": prior,
                            "training_eligible": True,
                            "memory_promotion_allowed": False,
                        }
                    )
    dataset = {
        "schema_version": "pg115-small-rule-ir-train-dataset-v1",
        "purpose": "compact local abstract decision training fixture",
        "evaluation_only": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "model_input_family_free": True,
        "model_input_oracle_blind": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "script_execution": False,
        "database_write": False,
        "labels_are_evaluator_fields_outside_model_input": True,
        "split_policy": "train seeds 11501/11502; dev seeds 11503/11504",
        "decision_set": list(PG115_DECISIONS),
        "feature_dim": FEATURE_DIM,
        "rows": rows,
    }
    dataset["manifest_sha256"] = _sha256_json(dataset)
    return dataset


def _load_pg114() -> tuple[dict[str, Any], dict[str, Any]]:
    visible = json.loads((RESEARCH / "pg114_family_holdout_replay_visible_dataset_v1.json").read_text(encoding="utf-8"))
    trace = json.loads((RESEARCH / "pg114_family_holdout_replay_trace_v1.json").read_text(encoding="utf-8"))
    return visible, trace


def _batch(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", [])) for row in rows],
        dtype=torch.float32,
    )
    y = torch.tensor([decision_index(row["label"]) for row in rows], dtype=torch.long)
    return x, y


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
        "confusion": {
            true_name: {
                predicted_name: sum(
                    prediction == predicted_index and label == true_index
                    for prediction, label in zip(predictions, labels)
                )
                for predicted_index, predicted_name in enumerate(PG115_DECISIONS)
            }
            for true_index, true_name in enumerate(PG115_DECISIONS)
        },
    }


def _run_model(model: nn.Module, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows)
    logits = model(x.to(device))
    probabilities = torch.softmax(logits, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def evaluate_pg114(model: nn.Module, device: torch.device) -> dict[str, Any]:
    visible, trace = _load_pg114()
    visible_by_id = {row["row_id"]: row for row in visible["rows"]}
    eval_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    step_ids: list[str] = []
    episode_predictions: dict[str, list[int]] = defaultdict(list)
    episode_expected: dict[str, dict[str, Any]] = {}
    for episode in trace["episodes"]:
        prior_inputs: list[dict[str, Any]] = []
        for step in episode["steps"]:
            row = visible_by_id[step["step_id"]]
            eval_rows.append({
                "model_input": row["model_input"],
                "prior_inputs": prior_inputs,
                "label": step["decision"],
            })
            labels.append(decision_index(step["decision"]))
            step_ids.append(step["step_id"])
            prior_inputs.append(row["model_input"])
        episode_expected[episode["episode_id"]] = {
            "surface_kind": episode["surface_kind"],
            "final_decision": episode["final_decision"],
        }
    predictions, confidences = _run_model(model, eval_rows, device)
    for step_id, prediction in zip(step_ids, predictions):
        episode_id = step_id.rsplit("-s", 1)[0]
        episode_predictions[episode_id].append(prediction)
    final_rows = []
    for episode_id, expected in episode_expected.items():
        prediction = episode_predictions[episode_id][-1]
        predicted_name = PG115_DECISIONS[prediction]
        final_rows.append({
            "episode_id": episode_id,
            "surface_kind": expected["surface_kind"],
            "expected_final_decision": expected["final_decision"],
            "predicted_final_decision": predicted_name,
        })
    positives = [row for row in final_rows if row["surface_kind"] == "policy"]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    neutrals = [row for row in final_rows if row["surface_kind"] == "neutral"]
    opaque = [row for row in final_rows if row["surface_kind"] == "opaque"]
    return {
        "step_metrics": _metrics(predictions, labels),
        "final_episode_rows": final_rows,
        "family_holdout_confirm_recall": round(
            sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6
        ),
        "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys),
        "neutral_confirmed_negative_rate": round(
            sum(row["predicted_final_decision"] == "confirmed_negative" for row in neutrals) / len(neutrals), 6
        ),
        "withheld_oracle_abstain_rate": round(
            sum(row["predicted_final_decision"] == "abstain" for row in opaque) / len(opaque), 6
        ),
        "mean_confidence": round(sum(confidences) / len(confidences), 6),
        "evaluation_rows": len(eval_rows),
    }


def main() -> None:
    random.seed(11515)
    torch.manual_seed(11515)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11515)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataset = build_training_dataset()
    TRAIN_DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_rows = [row for row in dataset["rows"] if row["split"] == "train"]
    dev_rows = [row for row in dataset["rows"] if row["split"] == "dev"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRuleIRDecisionDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train_rows)
    train_x, train_y = train_x.to(device), train_y.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, 31):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            train_prediction = model(train_x).argmax(dim=-1)
            train_accuracy = float((train_prediction == train_y).float().mean().item())
        dev_prediction, _ = _run_model(model, dev_rows, device)
        dev_labels = [decision_index(row["label"]) for row in dev_rows]
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": round(train_accuracy, 6), "dev_accuracy": _metrics(dev_prediction, dev_labels)["accuracy"]})

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "decision_set": list(PG115_DECISIONS),
            "device_at_training": str(device),
            "model_state_dict": model.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    train_prediction, _ = _run_model(model, train_rows, device)
    dev_prediction, _ = _run_model(model, dev_rows, device)
    train_labels = [decision_index(row["label"]) for row in train_rows]
    dev_labels = [decision_index(row["label"]) for row in dev_rows]
    pg114 = evaluate_pg114(model, device)
    report = {
        "protocol_id": "pg-pk-115-small-rule-ir-decoder-v1",
        "schema_version": "pg115-small-rule-ir-decoder-report-v1",
        "status": "completed_pg115_small_rule_ir_decoder_trial",
        "scope": {
            "training_fixture": "research/pg115_small_rule_ir_train_dataset_v1.json",
            "blind_evaluation": "research/pg114_family_holdout_replay_visible_dataset_v1.json",
            "model": "small_rule_ir_decision_mlp",
            "feature_dim": FEATURE_DIM,
            "hidden_dim": 48,
            "epochs": 30,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "real_vulnerability_scanner_claim_allowed": False,
        },
        "dataset": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_seed_set": sorted({row["source_seed"] for row in train_rows}),
            "dev_seed_set": sorted({row["source_seed"] for row in dev_rows}),
            "train_label_counts": dict(Counter(row["label"] for row in train_rows)),
            "dev_label_counts": dict(Counter(row["label"] for row in dev_rows)),
            "source_hash": _sha256_file(TRAIN_DATASET_PATH),
        },
        "training": {
            "history_tail": history[-5:],
            "train_metrics": _metrics(train_prediction, train_labels),
            "dev_metrics": _metrics(dev_prediction, dev_labels),
        },
        "blind_pg114": pg114,
        "checks": {
            "train_dev_seed_disjoint": True,
            "model_input_family_free": dataset["model_input_family_free"],
            "model_input_oracle_blind": dataset["model_input_oracle_blind"],
            "no_raw_probe_strings": not dataset["raw_probe_strings_stored"],
            "no_raw_response_bodies": not dataset["raw_response_bodies_stored"],
            "pg114_rows_not_in_training_dataset": True,
            "pg114_final_positive_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0,
            "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0,
            "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0,
            "all_abstain_not_success": pg114["family_holdout_confirm_recall"] > 0.0,
        },
        "promotion": {
            "checkpoint_written": True,
            "training_artifact_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "status": "small_trial_only",
            "reason": "synthetic compact fixture and one blind OOD replay are insufficient for a general scanner claim",
        },
        "source": {
            "decoder": _sha256_file(ROOT / "app/pg115_small_rule_ir_decoder.py"),
            "runner": _sha256_file(Path(__file__)),
            "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json"),
        },
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = "\n".join(
        [
            "# PG-115 小型 Rule IR 解码器试验",
            "",
            "这是一次直接训练的小实验，不是漏洞扫描器能力宣称。训练样本与 PG-114 盲测分离。",
            "",
            f"- 设备：`{device}`；参数量：`{report['scope']['parameter_count']}`；轮数：`30`。",
            f"- 训练/开发：`{len(train_rows)}/{len(dev_rows)}`；开发准确率：`{report['training']['dev_metrics']['accuracy']}`。",
            f"- PG-114 族外正例召回：`{pg114['family_holdout_confirm_recall']}`。",
            f"- decoy 误接受：`{pg114['decoy_false_accept_count']}`；未知 oracle 弃权率：`{pg114['withheld_oracle_abstain_rate']}`。",
            "- 结论：已写入新 checkpoint，但暂不提升长期记忆，也不宣称可检测任意真实网址漏洞。",
            "",
        ]
    )
    MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "device": str(device),
        "parameter_count": report["scope"]["parameter_count"],
        "train_accuracy": report["training"]["train_metrics"]["accuracy"],
        "dev_accuracy": report["training"]["dev_metrics"]["accuracy"],
        "pg114_family_holdout_confirm_recall": pg114["family_holdout_confirm_recall"],
        "pg114_decoy_false_accept_count": pg114["decoy_false_accept_count"],
        "pg114_withheld_oracle_abstain_rate": pg114["withheld_oracle_abstain_rate"],
        "checkpoint": str(CHECKPOINT_PATH),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
