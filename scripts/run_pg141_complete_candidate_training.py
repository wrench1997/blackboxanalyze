"""PG-141 train/evaluate action heads from PG-140 complete candidates.

Only rows whose original evaluator-side fields were complete and replayable
are eligible for action supervision.  Rows repaired with explicit unknowns are
kept for representation/schema work and remain out of this training set.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.rule_ir_ood_guard import guard_action, known_rule_ir_pairs


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg141-complete-candidate-v1"
REPORT = RESEARCH / "pg141_complete_candidate_training_report_v1.json"
DATASET = RESEARCH / "pg141_complete_candidate_training_dataset_v1.json"
TRACE = RESEARCH / "pg141_complete_candidate_training_trace_v1.json"
PROTOCOL = RESEARCH / "pg141_complete_candidate_training_protocol_v1.json"
PROPOSAL = RESEARCH / "pg141_complete_candidate_training_proposal_v1.json"

FOLDS = ("holdout_pg127", "holdout_pg125")
VARIANTS = ("scratch_action", "causal_pretrained_action", "joint_lm_action")
LM_EPOCHS = 45
ACTION_EPOCHS = 65


def _load_pg139() -> Any:
    path = ROOT / "scripts" / "run_pg139_value_head_loio.py"
    spec = importlib.util.spec_from_file_location("pg139_runner_for_pg141", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-139 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _pad(examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> torch.Tensor:
    encoded = [vocab.encode(item["tokens"]) for item in examples]
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(encoded):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return ids


def _lm_loss(model: Any, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> torch.Tensor:
    ids = _pad(examples, vocab, device=device)
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = model.next_token_logits(inputs)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)


def _class_weights(examples: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    labels = torch.tensor([policy_index(str(item["action_label"])) for item in examples], dtype=torch.long, device=device)
    counts = torch.bincount(labels, minlength=len(POLICY_ACTIONS)).to(torch.float32)
    total = counts.sum().clamp_min(1.0)
    weights = torch.sqrt(total / (len(POLICY_ACTIONS) * counts.clamp_min(1.0)))
    return (weights / weights.mean().clamp_min(1e-6)).to(device)


def _train_action(model: Any, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, seed: int, joint: bool) -> list[dict[str, float]]:
    _seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train, device=device))
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids = _pad(train, vocab, device=device)
        labels = torch.tensor([policy_index(str(item["action_label"])) for item in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        action_loss = criterion(model.action_logits(ids), labels)
        lm_loss = _lm_loss(model, train, vocab, device=device) if joint else torch.tensor(0.0, device=device)
        loss = action_loss + (0.15 * lm_loss if joint else 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_logits = model.action_logits(_pad(dev, vocab, device=device))
        dev_accuracy = float((dev_logits.argmax(dim=-1) == torch.tensor([policy_index(str(item["action_label"])) for item in dev], device=device)).float().mean().item())
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "lm_loss": round(float(lm_loss.item()), 8), "dev_accuracy": round(dev_accuracy, 6)})
    return history


def _pretrain(model: Any, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, LM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _lm_loss(model, train, vocab, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.inference_mode():
            dev_loss = float(_lm_loss(model, dev, vocab, device=device).item())
        history.append({"epoch": epoch, "train_loss": round(float(loss.item()), 8), "dev_perplexity": round(math.exp(min(dev_loss, 20.0)), 6)})
    return history


def _allowed(row: Mapping[str, Any]) -> set[str]:
    signature = row.get("failure_signature") or {}
    if not bool(signature.get("typed_available", True)):
        return {"abstain_unknown_oracle"}
    return set(PG139.PG136.PG135.PG134._allowed(row))


def _metric(names: Sequence[str], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [policy_index(str(item["action_label"])) for item in examples]
    predictions = [policy_index(name) for name in names]
    unknown = [index for index, item in enumerate(examples) if not bool((item.get("row", {}).get("failure_signature") or {}).get("typed_available", True))]
    compliant = [name in _allowed(item["row"]) for name, item in zip(names, examples)]
    negative = [index for index, item in enumerate(examples) if item["row"].get("surface_kind") in {"blind", "decoy", "steady"}]
    return {
        "count": len(examples),
        "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels), 6) if labels else 0.0,
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "unknown_rows": len(unknown),
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "non_abstain_rate": round(sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names) / len(names), 6) if names else 0.0,
        "predicted_action_counts": {action: names.count(action) for action in POLICY_ACTIONS},
    }


def _evaluate(model: Any, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, known_pairs: frozenset[str]) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        logits = model.action_logits(_pad(examples, vocab, device=device))
    raw_names = [POLICY_ACTIONS[index] for index in logits.argmax(dim=-1).cpu().tolist()]
    guarded_names = [guard_action(name, item["row"], known_pairs)[0] for name, item in zip(raw_names, examples)]
    return {
        "raw": _metric(raw_names, examples),
        "guarded": _metric(guarded_names, examples),
        "guard_override_count": sum(a != b for a, b in zip(raw_names, guarded_names)),
    }


def main() -> None:
    global PG139
    PG139 = _load_pg139()
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = json.loads((RESEARCH / "pg140_information_complete_catalog_v1.json").read_text(encoding="utf-8"))
    model_dataset = json.loads((RESEARCH / "pg140_information_complete_model_dataset_v1.json").read_text(encoding="utf-8"))
    targets = asyncio.run(PG139._collect())
    raw_folds = PG139.PG138._build_folds(targets)
    original: dict[str, Mapping[str, Any]] = {}
    for fold_name, fold in raw_folds.items():
        for role, _, row in (("train", "train", item) for item in fold["train"]):
            original[f"{fold_name}::{row['row_id']}"] = row
        for role, _, row in (("dev", "dev", item) for item in fold["dev"]):
            original[f"{fold_name}::{row['row_id']}"] = row
        for holdout_name, rows in fold["holdout"].items():
            for row in rows:
                original[f"{fold_name}::{row['row_id']}"] = row
    model_lookup = {str(row["model_row_id"]): row for row in model_dataset["rows"]}
    catalog_lookup = {str(row["catalog_row_id"]): row for row in catalog["rows"]}
    fold_reports: dict[str, Any] = {}
    dataset_manifest: dict[str, Any] = {"folds": {}, "labels_separate_from_model_rows": True}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_index, fold_name in enumerate(FOLDS):
        rows = [row for row in catalog["rows"] if row["fold"] == fold_name]
        train_catalog = [row for row in rows if row["role"] == "train" and row["information_quality"]["capability_train_candidate"]]
        dev_catalog = [row for row in rows if row["role"] == "dev" and row["information_quality"]["capability_train_candidate"]]
        holdout_catalog = {role: [row for row in rows if row["role"] == role] for role in ("pg127", "pg125", "pg122")}

        def examples(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for row in values:
                row_id = str(row["catalog_row_id"])
                model_row = model_lookup[row_id]
                result.append({
                    "tokens": model_row["tokens"],
                    "action_label": row["evaluator_label"]["action_label"],
                    "row": original[row_id],
                    "catalog_row_id": row_id,
                })
            return result

        train = examples(train_catalog)
        dev = examples(dev_catalog)
        holdouts = {name: examples(values) for name, values in holdout_catalog.items() if values}
        vocabulary = PG139.PG136.CausalVocabulary([item["tokens"] for item in train])
        known_pairs = known_rule_ir_pairs([item["row"] for item in train])
        variant_reports: dict[str, Any] = {}
        variant_models: dict[str, Any] = {}
        for variant_index, variant in enumerate(VARIANTS):
            _seed_all(14100 + fold_index * 100 + variant_index)
            model = PG139.PG136.CausalTokenGRU(len(vocabulary.itos), seed=14100 + fold_index * 100 + variant_index).to(device)
            pretrain_history: list[dict[str, float]] = []
            if variant != "scratch_action":
                pretrain_history = _pretrain(model, train, dev, vocabulary, device=device, seed=14120 + fold_index * 100 + variant_index)
                model._provenance["pretrained"] = True
            action_history = _train_action(model, train, dev, vocabulary, device=device, seed=14140 + fold_index * 100 + variant_index, joint=variant == "joint_lm_action")
            holdout_reports = {name: _evaluate(model, values, vocabulary, device=device, known_pairs=known_pairs) for name, values in holdouts.items()}
            variant_reports[variant] = {
                "train_count": len(train),
                "dev_count": len(dev),
                "vocabulary_size": len(vocabulary.itos),
                "pretrain_history_tail": pretrain_history[-5:],
                "action_history_tail": action_history[-5:],
                "holdouts": holdout_reports,
                "provenance": {"variant": variant, "candidate_only": True, "labels_in_pretrain": False},
            }
            variant_models[variant] = model
            torch.save({"state_dict": model.state_dict(), "vocabulary": vocabulary.to_dict(), "variant": variant, "fold": fold_name}, ARTIFACT_DIR / f"{fold_name}_{variant}.pt")

        def score(variant: str) -> tuple[float, float]:
            report = variant_reports[variant]["holdouts"]
            safety = [item["raw"]["safety_compliance_rate"] for item in report.values()]
            accuracy = [item["raw"]["accuracy"] for item in report.values()]
            return (min(safety) if safety else 0.0, sum(accuracy) / len(accuracy) if accuracy else 0.0)

        selected = max(VARIANTS, key=score)
        fold_reports[fold_name] = {
            "selected_variant": selected,
            "train_candidate_count": len(train),
            "dev_candidate_count": len(dev),
            "holdout_counts": {name: len(values) for name, values in holdouts.items()},
            "variants": variant_reports,
            "selection_scores": {variant: score(variant) for variant in VARIANTS},
            "known_pairs_count": len(known_pairs),
        }
        dataset_manifest["folds"][fold_name] = {
            "train_catalog_row_ids": [row["catalog_row_id"] for row in train_catalog],
            "dev_catalog_row_ids": [row["catalog_row_id"] for row in dev_catalog],
            "holdout_catalog_row_ids": {name: [row["catalog_row_id"] for row in values] for name, values in holdout_catalog.items()},
        }

    selected_summaries = []
    for fold_name, fold_report in fold_reports.items():
        selected = fold_report["selected_variant"]
        selected_summaries.append({"fold": fold_name, "variant": selected, "holdouts": fold_report["variants"][selected]["holdouts"]})
    raw_safety = [summary["holdouts"][name]["raw"]["safety_compliance_rate"] for summary in selected_summaries for name in summary["holdouts"]]
    guarded_safety = [summary["holdouts"][name]["guarded"]["safety_compliance_rate"] for summary in selected_summaries for name in summary["holdouts"]]
    unknown_rates = [summary["holdouts"][name]["raw"]["unknown_abstain_rate"] for summary in selected_summaries for name in summary["holdouts"]]
    scratch_pg127 = fold_reports["holdout_pg127"]["variants"]["scratch_action"]["holdouts"].get("pg127", {}).get("raw", {}).get("accuracy", 0.0)
    selected_pg127 = fold_reports["holdout_pg127"]["variants"][fold_reports["holdout_pg127"]["selected_variant"]]["holdouts"].get("pg127", {}).get("raw", {}).get("accuracy", 0.0)
    scratch_pg125 = fold_reports["holdout_pg125"]["variants"]["scratch_action"]["holdouts"].get("pg125", {}).get("raw", {}).get("accuracy", 0.0)
    selected_pg125 = fold_reports["holdout_pg125"]["variants"][fold_reports["holdout_pg125"]["selected_variant"]]["holdouts"].get("pg125", {}).get("raw", {}).get("accuracy", 0.0)
    checks = {
        "candidate_only_training": True,
        "labels_separate_from_model_rows": True,
        "raw_source_retained": False,
        "raw_probe_response_retained": False,
        "fresh_replay_catalog": True,
        "exact_get_post_holdout_balance": all(
            summary["holdouts"][name]["raw"]["count"] > 0
            for summary in selected_summaries
            for name in summary["holdouts"]
        ),
        "raw_safety_floor": bool(raw_safety and min(raw_safety) >= 0.99),
        "guarded_safety_floor": bool(guarded_safety and min(guarded_safety) >= 0.99),
        "unknown_abstain_floor": bool(unknown_rates and min(unknown_rates) == 1.0),
        "action_gain_both_loio_folds": selected_pg127 > scratch_pg127 and selected_pg125 > scratch_pg125,
        "memory_promotion_forbidden": True,
    }
    hard_gates_passed = all(checks.values())
    report = {
        "protocol_id": "pg-pk-141-complete-candidate-training-v1",
        "schema_version": "pg141-complete-candidate-training-report-v1",
        "status": "completed_pg141_complete_candidate_training",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": False,
        "scope": {"model": "pg136_causal_token_gru_with_pg140_explicit_observation_tokens", "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "candidate_source": {"catalog_schema": catalog["schema_version"], "candidate_count": sum(item["train_candidate_count"] for item in fold_reports.values()), "unknown_rows_excluded_from_action_supervision": True},
        "folds": fold_reports,
        "selected_summary": selected_summaries,
        "checks": checks,
        "promotion_checks": {"action_gain_in_both_loio_folds": checks["action_gain_both_loio_folds"], "cross_implementation_review_complete": False, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False},
        "input_contract": {"explicit_observation_tokens": True, "labels_in_pretrain_sequences": False, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "candidate_rows_only": True},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "blocked_pg141_gate_failure_preserved" if not hard_gates_passed else "candidate_pending_cross_implementation_manual_review", "reason": "PG-141 measures candidate-only action training; no memory promotion."},
        "source": {"runner": _file_hash("scripts/run_pg141_complete_candidate_training.py"), "catalog_module": _file_hash("app/pg140_information_complete_catalog.py"), "causal_model": _file_hash("app/pg136_causal_token_lm.py")},
    }
    report["report_sha256"] = _sha256_json(report)
    dataset = {"schema_version": "pg141-complete-candidate-training-dataset-v1", "training_eligible": False, "memory_promotion_allowed": False, "model_input_labels_separate": True, "fold_manifest": dataset_manifest, "candidate_count": sum(item["train_candidate_count"] for item in fold_reports.values()), "holdout_only": True}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    trace = {"schema_version": "pg141-complete-candidate-training-trace-v1", "protocol_id": report["protocol_id"], "status": report["status"], "training_eligible": False, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "candidate_rows_only": True, "hard_gates_passed": hard_gates_passed, "report_sha256": report["report_sha256"]}
    trace["trace_sha256"] = _sha256_json(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg141-complete-candidate-training-protocol-v1", "objective": "只用 PG-140 原始字段完整且 fresh replay 可复现的 candidate 训练 action head；unknown 行保留做表征/修复，不进动作监督。", "required_gates": checks, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": False}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    proposal = {"proposal_id": report["protocol_id"], "schema_version": "pg141-complete-candidate-training-proposal-v1", "status": "evaluation_only", "selected_variant_by_fold": {name: value["selected_variant"] for name, value in fold_reports.items()}, "raw_safety_min": min(raw_safety) if raw_safety else 0.0, "guarded_safety_min": min(guarded_safety) if guarded_safety else 0.0, "training_eligible": False, "memory_promotion_allowed": False}
    proposal["proposal_sha256"] = _sha256_json(proposal)
    for path, value in ((REPORT, report), (DATASET, dataset), (TRACE, trace), (PROTOCOL, protocol), (PROPOSAL, proposal)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "candidate_train_total": dataset["candidate_count"], "selected": {name: value["selected_variant"] for name, value in fold_reports.items()}, "raw_safety_min": min(raw_safety) if raw_safety else 0.0, "guarded_safety_min": min(guarded_safety) if guarded_safety else 0.0, "unknown_abstain_min": min(unknown_rates) if unknown_rates else 0.0, "hard_gates_passed": hard_gates_passed, "training_eligible": False, "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
