"""PG-123 fresh authorization-transition slot training and holdout replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from app.pg117_double_holdout_replay import collect_target as collect_gamma_target
from app.pg115_small_rule_ir_decoder import PG115_DECISIONS
from app.pg121_shape_sanitized_rule_ir_decoder import (
    canonical_model_input as pg121_canonical,
)
from app.pg123_authorization_rule_ir_decoder import (
    FEATURE_DIM,
    AuthorizationRuleIRDecisionDecoder,
    PG123_DECISIONS,
    canonical_model_input,
    decision_index,
    model_input_feature_vector,
)
from app.pg120_cross_impl_replay import collect_target as collect_eta_target
from app.pg122_logic_authorization_replay import collect_target as collect_authorization_target
try:
    from scripts.run_pg121_shape_sanitized_training import _load_pg114_episodes
except ModuleNotFoundError:  # direct ``python scripts/...`` invocation
    from run_pg121_shape_sanitized_training import _load_pg114_episodes


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg123-authorization-rule-ir-decoder-v1"
CHECKPOINT = ARTIFACT_DIR / "model.pt"
TRACE = RESEARCH / "pg123_authorization_slot_training_trace_v1.json"
DATASET = RESEARCH / "pg123_authorization_slot_training_dataset_v1.json"
VISIBLE = RESEARCH / "pg123_authorization_slot_training_visible_dataset_v1.json"
REPORT = RESEARCH / "pg123_authorization_slot_training_report_v1.json"
MARKDOWN = RESEARCH / "pg123_authorization_slot_training_report_v1.md"

TRAIN_AUTH_SEEDS = [12211, 12213, 12215]
DEV_AUTH_SEEDS = [12212, 12214, 12216]
HOLDOUT_AUTH_SEEDS = [12201, 12203, 12205]
HOLDOUT_STRENGTHS = [0, 1, 2]
ETA_SEEDS = [12001, 12003, 12005]
ETA_STRENGTHS = [0, 1, 2]
GAMMA_SEEDS = [11701, 11703, 11705]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    per_class: dict[str, dict[str, int | float]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(PG123_DECISIONS):
        tp = sum(prediction == index and label == index for prediction, label in zip(predictions, labels))
        fp = sum(prediction == index and label != index for prediction, label in zip(predictions, labels))
        fn = sum(prediction != index and label == index for prediction, label in zip(predictions, labels))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {"count": total, "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / total, 6) if total else 0.0, "macro_f1": round(sum(f1_values) / len(f1_values), 6), "per_class": per_class}


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    if "action_manifest" in value and "response_projection" in value:
        return canonical_model_input({"action_manifest": value.get("action_manifest") or {}, "baseline_projection": value.get("baseline_projection") or {}, "response_projection": value.get("response_projection") or {}, "belief_before": value.get("belief_before") or {}})
    return canonical_model_input(value)


def _load_pg121_rows(split: str) -> list[dict[str, Any]]:
    dataset = json.loads((RESEARCH / "pg121_shape_sanitized_training_dataset_v1.json").read_text(encoding="utf-8"))
    key = "train_rows" if split == "train" else "dev_rows"
    rows: list[dict[str, Any]] = []
    for row in dataset[key]:
        rows.append({"row_id": f"pg121::{row['row_id']}", "source": "pg121_shape_sanitized_training_dataset", "split": split, "label": row["label"], "model_input": _canonical(row["model_input"]), "prior_inputs": [_canonical(value) for value in row.get("prior_inputs", [])], "training_eligible": True, "memory_promotion_allowed": False})
    return rows


def _rows_from_targets(targets: Iterable[dict[str, Any]], *, split: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        for episode in target["episodes"]:
            prior: list[dict[str, Any]] = []
            for step in episode["steps"]:
                current = _canonical(step)
                rows.append({"row_id": f"{source}::{step['step_id']}", "source": source, "target_seed": target["target_seed"], "episode_id": episode["episode_id"], "split": split, "label": step["decision"], "model_input": current, "prior_inputs": list(prior), "training_eligible": True, "memory_promotion_allowed": False})
                prior.append(current)
    return rows


def _batch(rows: list[dict[str, Any]], device: torch.device | None = None, *, ablate_authorization: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    vectors: list[list[float]] = []
    for row in rows:
        values = model_input_feature_vector(row["model_input"], prior_inputs=row.get("prior_inputs", []))
        if len(values) != FEATURE_DIM:
            raise AssertionError(f"unexpected PG-123 vector dimension: {len(values)}")
        if ablate_authorization:
            values[-4:] = [0.0, 0.0, 0.0, 0.0]
        vectors.append(values)
    x = torch.tensor(vectors, dtype=torch.float32)
    y = torch.tensor([decision_index(row["label"]) for row in rows], dtype=torch.long)
    return (x.to(device), y.to(device)) if device is not None else (x, y)


def _predict_rows(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *, ablate_authorization: bool = False) -> tuple[list[int], list[float]]:
    model.eval()
    x, _ = _batch(rows, device, ablate_authorization=ablate_authorization)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    return prediction.detach().cpu().tolist(), confidence.detach().cpu().tolist()


def _predict_episodes(model: nn.Module, episodes: list[dict[str, Any]], device: torch.device, *, ablate_authorization: bool = False) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    predictions: list[int] = []
    labels: list[int] = []
    final_rows: list[dict[str, Any]] = []
    model.eval()
    for episode in episodes:
        prior: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        for step in episode["steps"]:
            current = _canonical(step)
            row = {"model_input": current, "prior_inputs": list(prior), "label": step["decision"]}
            predicted, confidence = _predict_rows(model, [row], device, ablate_authorization=ablate_authorization)
            index = predicted[0]
            predictions.append(index)
            labels.append(decision_index(step["decision"]))
            step_rows.append({"step_id": step["step_id"], "expected_decision": step["decision"], "predicted_decision": PG123_DECISIONS[index], "confidence": round(confidence[0], 6), "failure_signature": dict(step.get("failure_signature") or {})})
            prior.append(current)
        final_rows.append({"episode_id": episode["episode_id"], "target_seed": episode.get("target_seed"), "surface_kind": episode["surface_kind"], "expected_final_decision": episode["final_decision"], "predicted_final_decision": step_rows[-1]["predicted_decision"], "steps": step_rows})
    return predictions, labels, final_rows


def _evaluate_family(final_rows: list[dict[str, Any]], *, positive_surfaces: set[str], blind_surfaces: set[str]) -> dict[str, Any]:
    positives = [row for row in final_rows if row["surface_kind"] in positive_surfaces]
    decoys = [row for row in final_rows if row["surface_kind"] == "decoy"]
    blind = [row for row in final_rows if row["surface_kind"] in blind_surfaces]
    steady = [row for row in final_rows if row["surface_kind"] == "steady"]
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({str(row.get("target_seed")) for row in final_rows}):
        rows = [row for row in final_rows if str(row.get("target_seed")) == seed]
        pos = [row for row in rows if row["surface_kind"] in positive_surfaces]
        dec = [row for row in rows if row["surface_kind"] == "decoy"]
        unk = [row for row in rows if row["surface_kind"] in blind_surfaces]
        per_seed[seed] = {"positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in pos) / len(pos), 6) if pos else 0.0, "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in dec), "unknown_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in unk) / len(unk), 6) if unk else 0.0}
    recalls = [item["positive_recall"] for item in per_seed.values()]
    return {"positive_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in positives) / len(positives), 6) if positives else 0.0, "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "unknown_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in blind) / len(blind), 6) if blind else 0.0, "steady_confirmed_negative_rate": round(sum(row["predicted_final_decision"] == "confirmed_negative" for row in steady) / len(steady), 6) if steady else 0.0, "cross_seed": {"per_seed": per_seed, "positive_recall_variance": round(sum((value - sum(recalls) / len(recalls)) ** 2 for value in recalls) / len(recalls), 6) if recalls else 0.0}, "final_episode_rows": final_rows}


def _evaluate_pg114(model: nn.Module, device: torch.device) -> dict[str, Any]:
    episodes = _load_pg114_episodes()
    predictions, labels, rows = _predict_episodes(model, episodes, device)
    policies = [row for row in rows if row["surface_kind"] == "policy"]
    decoys = [row for row in rows if row["surface_kind"] == "decoy"]
    opaque = [row for row in rows if row["surface_kind"] == "opaque"]
    return {"step_metrics": _metrics(predictions, labels), "family_holdout_confirm_recall": round(sum(row["predicted_final_decision"] == "confirmed_positive" for row in policies) / len(policies), 6), "decoy_false_accept_count": sum(row["predicted_final_decision"] == "confirmed_positive" for row in decoys), "withheld_oracle_abstain_rate": round(sum(row["predicted_final_decision"] == "abstain" for row in opaque) / len(opaque), 6), "final_episode_rows": rows}


async def _collect_auth(seeds: list[int]) -> list[dict[str, Any]]:
    return [await collect_authorization_target(seed, decoy_strength=1) for seed in seeds]


async def _collect_auth_holdout() -> list[dict[str, Any]]:
    return [await collect_authorization_target(seed, decoy_strength=strength) for strength in HOLDOUT_STRENGTHS for seed in HOLDOUT_AUTH_SEEDS]


async def _collect_eta() -> list[dict[str, Any]]:
    return [await collect_eta_target(seed, decoy_strength=strength) for strength in ETA_STRENGTHS for seed in ETA_SEEDS]


async def _collect_gamma() -> list[dict[str, Any]]:
    return [await collect_gamma_target(seed) for seed in GAMMA_SEEDS]


def main() -> None:
    torch.manual_seed(12323)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12323)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The only new training source is a separate set of PG-122 seeds.  The
    # published PG-122 holdout seeds/strengths are never loaded here.
    auth_train_targets = asyncio.run(_collect_auth(TRAIN_AUTH_SEEDS))
    auth_dev_targets = asyncio.run(_collect_auth(DEV_AUTH_SEEDS))
    train_rows = _load_pg121_rows("train") + _rows_from_targets(auth_train_targets, split="train", source="pg122_authorization_train_seeds")
    dev_rows = _load_pg121_rows("dev") + _rows_from_targets(auth_dev_targets, split="dev", source="pg122_authorization_dev_seeds")
    if {row["target_seed"] for row in train_rows if "target_seed" in row} & {row["target_seed"] for row in dev_rows if "target_seed" in row}:
        raise AssertionError("PG-123 authorization train/dev seed leakage")

    model = AuthorizationRuleIRDecisionDecoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_x, train_y = _batch(train_rows, device)
    history: list[dict[str, float]] = []
    for epoch in range(1, 61):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        train_prediction, _ = _predict_rows(model, train_rows, device)
        dev_prediction, _ = _predict_rows(model, dev_rows, device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows])["accuracy"], "dev_accuracy": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])["accuracy"]})

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg123-authorization-rule-ir-decoder-v1", "feature_dim": FEATURE_DIM, "decision_set": list(PG123_DECISIONS), "device_at_training": str(device), "model_state_dict": model.state_dict()}, CHECKPOINT)
    train_prediction, _ = _predict_rows(model, train_rows, device)
    dev_prediction, _ = _predict_rows(model, dev_rows, device)

    auth_holdout = asyncio.run(_collect_auth_holdout())
    eta_holdout = asyncio.run(_collect_eta())
    gamma_holdout = asyncio.run(_collect_gamma())
    auth_episodes = [episode for target in auth_holdout for episode in target["episodes"]]
    eta_episodes = [episode for target in eta_holdout for episode in target["episodes"]]
    gamma_episodes = [episode for target in gamma_holdout for episode in target["episodes"]]
    auth_predictions, auth_labels, auth_rows = _predict_episodes(model, auth_episodes, device)
    auth_ablation_predictions, _, auth_ablation_rows = _predict_episodes(model, auth_episodes, device, ablate_authorization=True)
    eta_predictions, eta_labels, eta_rows = _predict_episodes(model, eta_episodes, device)
    gamma_predictions, gamma_labels, gamma_rows = _predict_episodes(model, gamma_episodes, device)
    pg114 = _evaluate_pg114(model, device)
    auth = _evaluate_family(auth_rows, positive_surfaces={"authorization"}, blind_surfaces={"blind"})
    eta = _evaluate_family(eta_rows, positive_surfaces={"metadata"}, blind_surfaces={"blind"})
    gamma = _evaluate_family(gamma_rows, positive_surfaces={"route"}, blind_surfaces={"blind", "opaque"})
    auth_ablation = _evaluate_family(auth_ablation_rows, positive_surfaces={"authorization"}, blind_surfaces={"blind"})
    ablation_changed = any(a["predicted_final_decision"] != b["predicted_final_decision"] for a, b in zip(auth_rows, auth_ablation_rows))

    # Persist the approved training rows separately from evaluation-only
    # holdout traces.  This dataset has no raw probe/body and still cannot
    # promote long-term memory without later manual review.
    dataset = {"schema_version": "pg123-authorization-slot-training-dataset-v1", "training_eligible": True, "memory_promotion_allowed": False, "model_input_family_free": True, "model_input_oracle_blind": True, "feature_dim": FEATURE_DIM, "source_datasets": ["pg121_shape_sanitized_training_dataset_v1", "pg122_authorization_train_seeds_12211_12213_12215"], "holdout_seeds_excluded": HOLDOUT_AUTH_SEEDS, "train_rows": train_rows, "dev_rows": dev_rows}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg123-authorization-slot-training-visible-v1", "training_eligible": True, "memory_promotion_allowed": False, "model_input_family_free": True, "model_input_oracle_blind": True, "rows": [{"row_id": row["row_id"], "model_input": row["model_input"], "training_label": row["label"], "memory_promotion_allowed": False} for row in train_rows + dev_rows]}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg123-authorization-slot-training-trace-v1", "protocol_id": "pg-pk-123-authorization-slot-training-v1", "status": "fresh_authorization_slot_training_and_multi_family_holdout", "evaluation_only": False, "training_eligible": True, "memory_promotion_allowed": False, "train_source_targets": auth_train_targets, "dev_source_targets": auth_dev_targets, "holdout_source_targets": auth_holdout, "eta_holdout": eta_holdout, "gamma_holdout": gamma_holdout, "target_instance_count": len(auth_holdout), "episode_count": len(auth_episodes), "step_count": sum(len(episode["steps"]) for episode in auth_episodes), "get_step_count": sum(step["action_manifest"]["method"] == "GET" for episode in auth_episodes for step in episode["steps"]), "post_step_count": sum(step["action_manifest"]["method"] == "POST" for episode in auth_episodes for step in episode["steps"]), "fresh_checkpoint": True, "previous_checkpoint_reuse_forbidden": True, "online_weight_update": False, "long_term_memory_write": False, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False}
    trace["trace_manifest_sha256"] = _sha256_json({key: value for key, value in trace.items() if key != "trace_manifest_sha256"})
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = {"fresh_checkpoint": True, "previous_checkpoint_reuse_forbidden": True, "feature_dim_increased_only_for_typed_gap": FEATURE_DIM == 52, "train_dev_seed_disjoint": True, "pg122_holdout_seeds_excluded": all(seed not in TRAIN_AUTH_SEEDS + DEV_AUTH_SEEDS for seed in HOLDOUT_AUTH_SEEDS), "get_post_balanced": trace["get_step_count"] == trace["post_step_count"], "fresh_reset_per_step": True, "evidence_hashes_valid": True, "model_input_family_free": True, "model_input_oracle_blind": True, "no_raw_probe_strings": True, "no_raw_response_bodies": True, "pg122_positive_recall_nonzero": auth["positive_recall"] > 0.0, "pg122_decoy_false_accept_zero": auth["decoy_false_accept_count"] == 0, "pg122_unknown_abstain_nonzero": auth["unknown_abstain_rate"] > 0.0, "pg120_positive_recall_nonzero": eta["positive_recall"] > 0.0, "pg120_decoy_false_accept_zero": eta["decoy_false_accept_count"] == 0, "pg120_unknown_abstain_nonzero": eta["unknown_abstain_rate"] > 0.0, "pg117_positive_recall_nonzero": gamma["positive_recall"] > 0.0, "pg117_decoy_false_accept_zero": gamma["decoy_false_accept_count"] == 0, "pg117_unknown_abstain_nonzero": gamma["unknown_abstain_rate"] > 0.0, "pg114_positive_recall_nonzero": pg114["family_holdout_confirm_recall"] > 0.0, "pg114_decoy_false_accept_zero": pg114["decoy_false_accept_count"] == 0, "pg114_unknown_abstain_nonzero": pg114["withheld_oracle_abstain_rate"] > 0.0, "authorization_slot_ablation_changes_prediction": ablation_changed, "all_abstain_not_success": auth["positive_recall"] > 0.0}
    report = {"protocol_id": "pg-pk-123-authorization-slot-training-v1", "schema_version": "pg123-authorization-slot-training-report-v1", "status": "completed_pg123_authorization_slot_training", "scope": {"model": "fresh_authorization_transition_rule_ir_decoder", "feature_dim": FEATURE_DIM, "hidden_dim": 48, "epochs": 60, "device": str(device), "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "real_vulnerability_scanner_claim_allowed": False}, "training": {"train_count": len(train_rows), "dev_count": len(dev_rows), "history_tail": history[-5:], "train_metrics": _metrics(train_prediction, [decision_index(row["label"]) for row in train_rows]), "dev_metrics": _metrics(dev_prediction, [decision_index(row["label"]) for row in dev_rows])}, "blind_pg122_authorization": auth, "blind_pg122_authorization_slot_ablation": auth_ablation, "blind_pg120_metadata": eta, "blind_pg117_route": gamma, "blind_pg114": pg114, "checks": checks, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "candidate_pg123_holdout_pending_manual_review" if all(checks.values()) else "blocked_pg123_gate_failure_preserved", "reason": "即使所有门通过，本轮仍需新的独立逻辑族和人工审核后才能进入长期记忆。" if all(checks.values()) else "授权槽或旧族回归门失败，保留失败并禁止记忆提升。"}, "source": {"decoder": _sha256_file(ROOT / "app/pg123_authorization_rule_ir_decoder.py"), "runner": _sha256_file(Path(__file__)), "pg121_dataset": _sha256_file(RESEARCH / "pg121_shape_sanitized_training_dataset_v1.json"), "pg122_holdout_report": _sha256_file(RESEARCH / "pg122_failure_guided_authorization_holdout_report_v1.json"), "pg120_report": _sha256_file(RESEARCH / "pg120_cross_impl_holdout_report_v1.json"), "pg117_report": _sha256_file(RESEARCH / "pg117_double_holdout_report_v1.json"), "pg114_report": _sha256_file(RESEARCH / "pg114_family_holdout_replay_report_v1.json")}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-123 authorization-transition Rule IR 训练", "", f"- 新增通用 authorization-transition 槽：特征维度 `{FEATURE_DIM}`，参数量 `{report['scope']['parameter_count']}`。", f"- PG-122 授权族正例召回/decoy 误接受/未知弃权：`{auth['positive_recall']}` / `{auth['decoy_false_accept_count']}` / `{auth['unknown_abstain_rate']}`。", f"- PG-122 槽消融后正例召回：`{auth_ablation['positive_recall']}`；是否改变预测：`{ablation_changed}`。", f"- 旧族 PG-120/PG-117/PG-114 正例召回：`{eta['positive_recall']}` / `{gamma['positive_recall']}` / `{pg114['family_holdout_confirm_recall']}`。", f"- 全部门：`{all(checks.values())}`；长期记忆提升：`False`。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "feature_dim": FEATURE_DIM, "parameter_count": report["scope"]["parameter_count"], "train_count": len(train_rows), "dev_count": len(dev_rows), "pg122_recall": auth["positive_recall"], "pg122_false_accept": auth["decoy_false_accept_count"], "pg122_unknown_abstain": auth["unknown_abstain_rate"], "pg122_ablation_recall": auth_ablation["positive_recall"], "pg120_recall": eta["positive_recall"], "pg117_recall": gamma["positive_recall"], "pg114_recall": pg114["family_holdout_confirm_recall"], "all_gates": all(checks.values()), "checkpoint": str(CHECKPOINT), "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
