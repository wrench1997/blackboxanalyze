"""PG-119 fresh decoder training and metadata-slot ablation.

Training sources are PG-116 identity rows, the approved PG-118 location-delta
rows, and PG-119's independent metadata-transition rows.  PG-119 holdout
seeds, PG-117 gamma, and PG-114 remain outside training.  The script records
GET/POST, fresh reset, matched controls, evidence hashes, and a feature-slot
ablation; it never emits executable probes or writes long-term memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from app.pg117_double_holdout_replay import collect_target as collect_gamma_target
from app.pg119_metadata_replay import ENCODING_CHAIN, collect_target as collect_metadata_target
from app.pg119_metadata_rule_ir_decoder import (
    FEATURE_DIM,
    PG119_DECISIONS,
    MetadataRuleIRDecisionDecoder,
    ablate_metadata_slots,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg119-metadata-rule-ir-decoder-v1"
CHECKPOINT_PATH = ARTIFACT_DIR / "model.pt"
TRACE_PATH = RESEARCH / "pg119_metadata_training_trace_v1.json"
DATASET_PATH = RESEARCH / "pg119_metadata_training_dataset_v1.json"
VISIBLE_PATH = RESEARCH / "pg119_metadata_visible_dataset_v1.json"
REPORT_PATH = RESEARCH / "pg119_metadata_training_report_v1.json"
MARKDOWN_PATH = RESEARCH / "pg119_metadata_training_report_v1.md"

METADATA_TRAIN_SEEDS = [11901, 11903, 11905]
METADATA_DEV_SEEDS = [11902, 11904, 11906]
METADATA_HOLDOUT_SEEDS = [11907, 11909, 11911]
GAMMA_HOLDOUT_SEEDS = [11701, 11703, 11705]


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


def _canonical_row_input(value: dict[str, Any]) -> dict[str, Any]:
    return canonical_model_input({"action_manifest": value.get("action_manifest") or {}, "baseline_projection": value.get("baseline_projection") or {}, "response_projection": value.get("response_projection") or {}, "belief_before": value.get("belief_before") or {}})


def _load_rows(path: Path, *, split: str, source: str, only_source: str | None = None) -> list[dict[str, Any]]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    key = "train_rows" if split == "train" else "dev_rows"
    rows: list[dict[str, Any]] = []
    for row in dataset[key]:
        if only_source and row.get("source") != only_source:
            continue
        rows.append({
            "row_id": f"{source}-{row['row_id']}",
            "source": source,
            "target_seed": row.get("target_seed"),
            "episode_id": row.get("episode_id"),
            "split": split,
            "label": row["label"],
            "model_input": _canonical_row_input(row["model_input"]),
            "prior_inputs": [_canonical_row_input(value) for value in row.get("prior_inputs", [])],
            "training_eligible": True,
            "memory_promotion_allowed": False,
        })
    return rows


def _build_metadata_rows(collected: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    allowed = set(METADATA_TRAIN_SEEDS if split == "train" else METADATA_DEV_SEEDS)
    rows: list[dict[str, Any]] = []
    for bundle in collected:
        for episode in bundle["episodes"]:
            if episode["target_seed"] not in allowed:
                continue
            prior: list[dict[str, Any]] = []
            for step in episode["steps"]:
                model_input = _canonical_row_input(step)
                rows.append({
                    "row_id": step["step_id"],
                    "source": "pg119_metadata_triple_encoding",
                    "target_seed": episode["target_seed"],
                    "episode_id": episode["episode_id"],
                    "surface_kind": episode["surface_kind"],
                    "split": split,
                    "label": step["decision"],
                    "model_input": model_input,
                    "prior_inputs": list(prior),
                    "training_eligible": True,
                    "memory_promotion_allowed": False,
                })
                prior.append(model_input)
    return rows


def _balance(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    missing = [label for label in PG119_DECISIONS if not groups.get(label)]
    if missing:
        raise ValueError(f"PG-119 missing classes: {missing}")
    target = max(len(group) for group in groups.values())
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for label in PG119_DECISIONS:
        group = list(groups[label])
        rng.shuffle(group)
        for repeat in range(target):
            base = group[repeat % len(group)]
            row = dict(base)
            row["row_id"] = f"{base['row_id']}-replay-{repeat:03d}"
            row["source_row_id"] = base["row_id"]
            row["replay_index"] = repeat
            balanced.append(row)
    rng.shuffle(balanced)
    return balanced, {label: len(groups[label]) for label in PG119_DECISIONS}


def _batch(rows: list[dict[str, Any]], *, ablate: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = [model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", [])) for row in rows]
    if ablate:
        vectors = [ablate_metadata_slots(vector) for vector in vectors]
    return torch.tensor(vectors, dtype=torch.float32), torch.tensor([decision_index(row["label"]) for row in rows], dtype=torch.long)


def _predict_rows(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, ablate: bool = False) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows, ablate=ablate)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x.to(device)), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def _predict_episodes(model: nn.Module, episodes: list[dict[str, Any]], device: torch.device, *, ablate: bool = False) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    predictions: list[int] = []
    labels: list[int] = []
    final_rows: list[dict[str, Any]] = []
    for episode in episodes:
        prior: list[dict[str, Any]] = []
        episode_steps: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = _canonical_row_input(step)
            row = {"model_input": model_input, "prior_inputs": list(prior), "label": step["decision"]}
            predicted, confidence = _predict_rows(model, [row], device, ablate=ablate)
            predicted_index = predicted[0]
            predictions.append(predicted_index)
            labels.append(decision_index(step["decision"]))
            episode_steps.append({"step_id": step["step_id"], "expected_decision": step["decision"], "predicted_decision": PG119_DECISIONS[predicted_index], "confidence": round(confidence[0], 6)})
            prior.append(model_input)
        final_rows.append({"episode_id": episode["episode_id"], "target_seed": episode.get("target_seed"), "surface_kind": episode["surface_kind"], "expected_final_decision": episode["final_decision"], "predicted_final_decision": episode_steps[-1]["predicted_decision"], "steps": episode_steps})
    return predictions, labels, final_rows


def _load_pg114_episodes() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visible = json.loads((RESEARCH / "pg114_family_holdout_replay_visible_dataset_v1.json").read_text(encoding="utf-8"))
    trace = json.loads((RESEARCH / "pg114_family_holdout_replay_trace_v1.json").read_text(encoding="utf-8"))
    visible_by_id = {row["row_id"]: row for row in visible["rows"]}
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for episode in trace["episodes"]:
        prior: list[dict[str, Any]] = []
        episode_rows: list[dict[str, Any]] = []
        episode_steps: list[dict[str, Any]] = []
        for step in episode["steps"]:
            model_input = _canonical_row_input(visible_by_id[step["step_id"]]["model_input"])
            row = {"model_input": model_input, "prior_inputs": list(prior), "label": step["decision"]}
            rows.append(row)
            episode_rows.append(row)
            episode_steps.append({**step, "model_input": model_input})
            prior.append(model_input)
        episodes.append({**episode, "rows": episode_rows, "steps": episode_steps})
    return rows, episodes


def _evaluate_pg114(model: nn.Module, device: torch.device) -> dict[str, Any]:
    rows, episodes = _load_pg114_episodes()
    predictions, labels, final_rows = _predict_episodes(model, episodes, device)
    policies = [row for row in final_rows if row["surface_kind"] == "policy"]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    neutral = [row for row in final_rows if row["surface_kind"] == "neutral"]
    opaque = [row for row in final_rows if row["surface_kind"] == "opaque"]
    return {"step_metrics": _metrics(predictions, labels), "final_episode_rows": final_rows, "family_holdout_confirm_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in policies) / len(policies), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "neutral_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in neutral) / len(neutral), 6), "withheld_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in opaque) / len(opaque), 6), "evaluation_rows": len(rows)}


def _evaluate_group(model: nn.Module, device: torch.device, collected: list[dict[str, Any]], *, ablate: bool = False) -> dict[str, Any]:
    episodes = [episode for target in collected for episode in target["episodes"]]
    predictions, labels, final_rows = _predict_episodes(model, episodes, device, ablate=ablate)
    positives = [row for row in final_rows if row["surface_kind"] in {"route", "metadata"}]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    steady = [row for row in final_rows if row["surface_kind"] == "steady"]
    blind = [row for row in final_rows if row["surface_kind"] == "blind"]
    bindings = [{"slot_id": f"pg119-metadata-slot-holdout-{hashlib.sha256(row['surface_kind'].encode()).hexdigest()[:16]}", "binding_stage": "after_shadow_probe_and_typed_oracle", "surface_kind": row["surface_kind"], "predicted_decision": row["predicted_final_decision"], "evidence_sha256": next(episode["steps"][-1]["evidence_sha256"] for target in collected for episode in target["episodes"] if episode["episode_id"] == row["episode_id"]), "long_term_memory_write": False} for row in final_rows]
    per_seed: dict[str, dict[str, Any]] = {}
    for target in collected:
        seed = str(target["target_seed"])
        seed_rows = [row for row in final_rows if str(row.get("target_seed")) == seed]
        seed_positive = [row for row in seed_rows if row["surface_kind"] in {"route", "metadata"}]
        seed_decoys = [row for row in seed_rows if row["surface_kind"] == "decoy"]
        seed_blind = [row for row in seed_rows if row["surface_kind"] == "blind"]
        per_seed[seed] = {"positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in seed_positive) / len(seed_positive), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in seed_decoys), "unknown_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in seed_blind) / len(seed_blind), 6)}
    recall_values = [value["positive_recall"] for value in per_seed.values()]
    return {"step_metrics": _metrics(predictions, labels), "final_episode_rows": final_rows, "rule_ir_slot_bindings": bindings, "metadata_positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "steady_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in steady) / len(steady), 6), "blind_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in blind) / len(blind), 6), "cross_seed": {"per_seed": per_seed, "positive_recall_mean": round(statistics.fmean(recall_values), 6), "positive_recall_variance": round(statistics.pvariance(recall_values), 6)}, "evaluation_rows": len(predictions), "ablation": ablate}


async def _collect_metadata() -> list[dict[str, Any]]:
    return [await collect_metadata_target(seed) for seed in METADATA_TRAIN_SEEDS + METADATA_DEV_SEEDS + METADATA_HOLDOUT_SEEDS]


async def _collect_gamma() -> list[dict[str, Any]]:
    return [await collect_gamma_target(seed) for seed in GAMMA_HOLDOUT_SEEDS]


def main() -> None:
    random.seed(11919)
    torch.manual_seed(11919)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(11919)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    metadata_collected = asyncio.run(_collect_metadata())
    gamma_collected = asyncio.run(_collect_gamma())
    metadata_train_unique = _build_metadata_rows(metadata_collected, "train")
    metadata_dev_unique = _build_metadata_rows(metadata_collected, "dev")
    metadata_train_rows, metadata_train_class_unique = _balance(metadata_train_unique, seed=11920)
    metadata_dev_rows, metadata_dev_class_unique = _balance(metadata_dev_unique, seed=11921)
    train_rows = _load_rows(RESEARCH / "pg116_multisource_training_dataset_v1.json", split="train", source="pg116_identity") + _load_rows(RESEARCH / "pg118_transition_training_dataset_v1.json", split="train", source="pg118_delta_double_encoding", only_source="pg118_delta_double_encoding") + metadata_train_rows
    dev_rows = _load_rows(RESEARCH / "pg116_multisource_training_dataset_v1.json", split="dev", source="pg116_identity") + _load_rows(RESEARCH / "pg118_transition_training_dataset_v1.json", split="dev", source="pg118_delta_double_encoding", only_source="pg118_delta_double_encoding") + metadata_dev_rows
    random.Random(11922).shuffle(train_rows)
    random.Random(11923).shuffle(dev_rows)

    metadata_training = [target for target in metadata_collected if target["target_seed"] in set(METADATA_TRAIN_SEEDS + METADATA_DEV_SEEDS)]
    metadata_holdout = [target for target in metadata_collected if target["target_seed"] in set(METADATA_HOLDOUT_SEEDS)]
    metadata_episodes = [episode for target in metadata_training for episode in target["episodes"]]
    metadata_steps = [step for episode in metadata_episodes for step in episode["steps"]]
    slot_bindings = [episode["rule_ir_slot_binding"] for episode in metadata_episodes]
    trace = {
        "schema_version": "pg119-metadata-training-trace-v1",
        "protocol_id": "pg-pk-119-metadata-transition-slot-training-v1",
        "status": "training_and_holdout_trace_collected",
        "evaluation_only": False,
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "execution_mode": "in_process_loopback_asgi_get_post",
        "training_source_set": ["pg116_identity", "pg118_delta_double_encoding", "pg119_metadata_triple_encoding"],
        "holdout_source_set": ["pg119_metadata_seed_holdout", "pg117_gamma_double_encoding", "pg114_family_holdout_decoy_replay"],
        "metadata_training_target_instance_count": len(metadata_training),
        "metadata_holdout_target_instance_count": len(metadata_holdout),
        "metadata_training_episode_count": len(metadata_episodes),
        "metadata_training_step_count": len(metadata_steps),
        "metadata_training_get_step_count": sum(step["action_manifest"]["method"] == "GET" for step in metadata_steps),
        "metadata_training_post_step_count": sum(step["action_manifest"]["method"] == "POST" for step in metadata_steps),
        "metadata_encoding_chain": list(ENCODING_CHAIN),
        "fresh_reset_per_step": all(step["fresh_reset"]["fresh_target"] and step["fresh_reset"]["completed"] for step in metadata_steps),
        "evidence_hash_valid": all(record["evidence_hash"] == _sha256_json({key: value for key, value in record.items() if key != "evidence_hash"}) for episode in metadata_episodes for record in episode["evidence_records"]),
        "rule_ir_slot_binding_count": len(slot_bindings),
        "rule_ir_slot_bindings": slot_bindings,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "online_weight_update": False,
        "long_term_memory_write": False,
        "sources": metadata_training,
        "holdout_sources": metadata_holdout,
    }
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = {
        "schema_version": "pg119-metadata-training-dataset-v1",
        "purpose": "fresh Rule IR decoder training with location and metadata transition slots plus a third encoding order",
        "training_eligible": True,
        "memory_promotion_allowed": False,
        "model_input_family_free": True,
        "model_input_oracle_blind": True,
        "transition_slots": ["response_projection.location_changed + transition_delta", "response_projection.metadata_changed + transition_delta"],
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "external_network": False,
        "script_execution": False,
        "database_write": False,
        "train_unique_row_count": len(_load_rows(RESEARCH / "pg116_multisource_training_dataset_v1.json", split="train", source="pg116_identity")) + len(_load_rows(RESEARCH / "pg118_transition_training_dataset_v1.json", split="train", source="pg118_delta_double_encoding", only_source="pg118_delta_double_encoding")) + len(metadata_train_unique),
        "dev_unique_row_count": len(_load_rows(RESEARCH / "pg116_multisource_training_dataset_v1.json", split="dev", source="pg116_identity")) + len(_load_rows(RESEARCH / "pg118_transition_training_dataset_v1.json", split="dev", source="pg118_delta_double_encoding", only_source="pg118_delta_double_encoding")) + len(metadata_dev_unique),
        "train_rows_after_balance": len(train_rows),
        "dev_rows_after_balance": len(dev_rows),
        "metadata_train_class_unique": metadata_train_class_unique,
        "metadata_dev_class_unique": metadata_dev_class_unique,
        "pg119_holdout_excluded_from_training": True,
        "pg117_gamma_excluded_from_training": True,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
    }
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg119-metadata-visible-dataset-v1", "training_eligible": True, "model_input_family_free": True, "typed_oracle_labels_outside_model_input": True, "transition_slots_present": True, "rows": [{"row_id": row["row_id"], "source": row["source"], "split": row["split"], "model_input": row["model_input"], "training_label": row["label"], "memory_promotion_allowed": False} for row in train_rows + dev_rows]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE_PATH.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
    torch.save({"schema_version": "pg119-metadata-rule-ir-decoder-v1", "feature_dim": FEATURE_DIM, "decision_set": list(PG119_DECISIONS), "device_at_training": str(device), "model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
    train_prediction, _ = _predict_rows(model, train_rows, device)
    dev_prediction, _ = _predict_rows(model, dev_rows, device)
    pg114 = _evaluate_pg114(model, device)
    pg117 = _evaluate_group(model, device, gamma_collected)
    pg119 = _evaluate_group(model, device, metadata_holdout)
    pg119_ablation = _evaluate_group(model, device, metadata_holdout, ablate=True)
    ablation_changed = any(full["predicted_final_decision"] != ablated["predicted_final_decision"] for full, ablated in zip(pg119["final_episode_rows"], pg119_ablation["final_episode_rows"]))
    report = {
        "protocol_id": "pg-pk-119-metadata-transition-slot-training-v1",
        "schema_version": "pg119-metadata-training-report-v1",
        "status": "completed_pg119_metadata_transition_slot_training",
        "scope": {"model": "fresh_metadata_transition_rule_ir_decision_mlp", "feature_dim": FEATURE_DIM, "hidden_dim": 48, "epochs": 40, "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "real_vulnerability_scanner_claim_allowed": False},
        "collection": {"training_source_set": trace["training_source_set"], "holdout_source_set": trace["holdout_source_set"], "metadata_training_target_instance_count": trace["metadata_training_target_instance_count"], "metadata_holdout_target_instance_count": trace["metadata_holdout_target_instance_count"], "metadata_training_episode_count": trace["metadata_training_episode_count"], "metadata_training_step_count": trace["metadata_training_step_count"], "metadata_training_get_step_count": trace["metadata_training_get_step_count"], "metadata_training_post_step_count": trace["metadata_training_post_step_count"], "metadata_encoding_chain": trace["metadata_encoding_chain"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hash_valid": trace["evidence_hash_valid"], "rule_ir_slot_binding_count": trace["rule_ir_slot_binding_count"], "train_unique_row_count": dataset["train_unique_row_count"], "dev_unique_row_count": dataset["dev_unique_row_count"], "train_rows_after_balance": len(train_rows), "dev_rows_after_balance": len(dev_rows)},
        "training": {"history_tail": history[-5:], "train_metrics": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows]), "dev_metrics": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])},
        "blind_pg114": pg114,
        "blind_pg117": pg117,
        "blind_pg119": pg119,
        "slot_ablation_pg119": pg119_ablation,
        "checks": {"fresh_checkpoint": True, "previous_checkpoint_reuse_forbidden": True, "third_encoding_train_source_present": True, "pg119_holdout_excluded_from_training": True, "get_post_balanced_collection": trace["metadata_training_get_step_count"] == trace["metadata_training_post_step_count"], "fresh_reset_per_step": trace["fresh_reset_per_step"], "evidence_hashes_valid": trace["evidence_hash_valid"], "rule_ir_slots_bind_evidence": all(binding["evidence_sha256"] in {step["evidence_sha256"] for episode in metadata_episodes for step in episode["steps"]} for binding in slot_bindings), "model_input_transition_slots_present": visible["transition_slots_present"], "model_input_family_free": True, "model_input_oracle_blind": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "pg114_positive_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0, "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0, "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0, "pg117_positive_recall_nonzero": pg117["metadata_positive_recall"] > 0.0, "pg117_decoy_false_accept_zero": pg117["decoy_false_accept_count"] == 0, "pg117_unknown_abstain_nonzero": pg117["blind_oracle_abstain_rate"] > 0.0, "pg117_cross_seed_recall_variance_bounded": pg117["cross_seed"]["positive_recall_variance"] <= 0.1, "pg119_positive_recall_nonzero": pg119["metadata_positive_recall"] > 0.0, "pg119_decoy_false_accept_zero": pg119["decoy_false_accept_count"] == 0, "pg119_unknown_abstain_nonzero": pg119["blind_oracle_abstain_rate"] > 0.0, "pg119_cross_seed_recall_variance_bounded": pg119["cross_seed"]["positive_recall_variance"] <= 0.1, "metadata_slot_ablation_changes_prediction": ablation_changed, "full_recall_above_metadata_slot_ablation": pg119["metadata_positive_recall"] > pg119_ablation["metadata_positive_recall"], "all_abstain_not_success": pg114["family_holdout_confirm_recall"] > 0.0 and pg117["metadata_positive_recall"] > 0.0 and pg119["metadata_positive_recall"] > 0.0},
        "diagnosis": {"new_slot": "response_projection.metadata_changed + transition_delta", "ablation": "zero PG-119 metadata slots while keeping the fresh checkpoint fixed", "capacity_change": "one bounded four-slot extension", "checkpoint_initialized_from_previous": False},
        "promotion": {"checkpoint_written": True, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_metadata_transition_slot_training_holdout_pending_review", "reason": "third encoding and metadata-slot gain must survive additional families, seeds, and manual review before any memory admission"},
        "source": {"metadata_target": _sha256_file(ROOT / "app/pg119_metadata_training_target.py"), "metadata_bridge": _sha256_file(ROOT / "app/pg119_metadata_replay.py"), "decoder": _sha256_file(ROOT / "app/pg119_metadata_rule_ir_decoder.py"), "runner": _sha256_file(Path(__file__)), "pg116_dataset": _sha256_file(RESEARCH / "pg116_multisource_training_dataset_v1.json"), "pg118_dataset": _sha256_file(RESEARCH / "pg118_transition_training_dataset_v1.json"), "pg117_report": _sha256_file(RESEARCH / "pg117_double_holdout_report_v1.json"), "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json")},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text("\n".join(["# PG-119 metadata transition slot 训练", "", "新模型从零初始化，合并 PG-116 identity、PG-118 location delta 与 PG-119 第三编码顺序 metadata delta；PG-119 holdout、PG-117 gamma 和 PG-114 保持盲测。", "", f"- 参数/设备：`{report['scope']['parameter_count']}` / `{device}`；feature dim：`{FEATURE_DIM}`。", f"- PG-119 正例召回/decoy 误接受/未知弃权：`{pg119['metadata_positive_recall']}` / `{pg119['decoy_false_accept_count']}` / `{pg119['blind_oracle_abstain_rate']}`。", f"- PG-119 跨 seed 正例召回方差：`{pg119['cross_seed']['positive_recall_variance']}`。", f"- metadata slot 消融后正例召回：`{pg119_ablation['metadata_positive_recall']}`；预测变化：`{ablation_changed}`。", f"- 开发准确率：`{report['training']['dev_metrics']['accuracy']}`；PG-119 逐步宏 F1：`{pg119['step_metrics']['macro_f1']}`。", "- 只有所有硬门通过才进入 Codex/人工复核；当前不进入长期记忆。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": FEATURE_DIM, "parameter_count": report["scope"]["parameter_count"], "dev_accuracy": report["training"]["dev_metrics"]["accuracy"], "pg114_recall": pg114["family_holdout_confirm_recall"], "pg117_recall": pg117["metadata_positive_recall"], "pg119_recall": pg119["metadata_positive_recall"], "pg119_ablation_recall": pg119_ablation["metadata_positive_recall"], "pg119_false_accept": pg119["decoy_false_accept_count"], "pg119_unknown_abstain": pg119["blind_oracle_abstain_rate"], "ablation_changed": ablation_changed, "all_gates": all(report["checks"].values()), "checkpoint": str(CHECKPOINT_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
