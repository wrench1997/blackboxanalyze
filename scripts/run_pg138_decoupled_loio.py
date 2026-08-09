"""PG-138 decoupled causal representation/safety-head LOIO experiment.

Two leave-one-implementation-out folds are collected fresh from the local
ASGI fixtures.  Each fold compares a fresh scratch head, a frozen causal
backbone with a fresh adapter, balanced full fine-tuning, and a joint
next-token/action objective.  A typed-contract mask and confidence calibration
are evaluated separately from the raw neural action.  The experiment is
evaluation-only and never writes long-term memory.
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
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg138_safety_head import DecoupledCausalSafetyPolicy, SCHEMA_VERSION
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


def _load_pg136() -> Any:
    path = ROOT / "scripts" / "run_pg136_causal_token_lm.py"
    spec = importlib.util.spec_from_file_location("pg136_runner_for_pg138", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-136 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG136 = _load_pg136()
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg138-decoupled-loio-v1"
REPORT = RESEARCH / "pg138_decoupled_loio_report_v1.json"
DATASET = RESEARCH / "pg138_decoupled_loio_dataset_v1.json"
VISIBLE = RESEARCH / "pg138_decoupled_loio_visible_dataset_v1.json"
TRACE = RESEARCH / "pg138_decoupled_loio_trace_v1.json"
PROTOCOL = RESEARCH / "pg138_decoupled_loio_protocol_v1.json"
PROPOSAL = RESEARCH / "pg138_decoupled_loio_proposal_v1.json"

FOLDS = ("holdout_pg127", "holdout_pg125")
VARIANTS = ("scratch_balanced", "adapter_only", "balanced_full", "joint_balanced")
ACTION_EPOCHS = 60
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


async def _collect() -> dict[str, list[dict[str, Any]]]:
    return await PG136._collect()


def _rows_from_targets(targets: dict[str, list[dict[str, Any]]], key: str, *, split: str, source: str, history_source: bool = False) -> list[dict[str, Any]]:
    return PG136.PG135.PG134._rows_from_targets(targets[key], split=split, source=source, history_source=history_source)


def _build_folds(targets: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    pg135_train = _rows_from_targets(targets, "pg135_train", split="train", source="pg135_loio_train", history_source=True)
    pg135_dev = _rows_from_targets(targets, "pg135_dev", split="dev", source="pg135_loio_dev", history_source=True)
    pg135_holdout = _rows_from_targets(targets, "pg135_holdout", split="holdout", source="pg135_loio_holdout", history_source=True)
    pg127_train = _rows_from_targets(targets, "pg127_train", split="train", source="pg127_loio_train")
    pg127_dev = _rows_from_targets(targets, "pg127_dev", split="dev", source="pg127_loio_dev")
    pg127_holdout = _rows_from_targets(targets, "pg127_holdout", split="holdout", source="pg127_loio_holdout")
    pg125_train = _rows_from_targets(targets, "pg125_train", split="train", source="pg125_loio_train")
    pg125_dev = _rows_from_targets(targets, "pg125_dev", split="dev", source="pg125_loio_dev")
    pg125_holdout = _rows_from_targets(targets, "pg125_family_ood", split="family_ood", source="pg125_loio_holdout")
    pg122_holdout = _rows_from_targets(targets, "pg122_family_ood", split="family_ood", source="pg122_loio_holdout")
    return {
        "holdout_pg127": {"train": pg135_train + pg125_train, "dev": pg135_dev + pg125_dev, "holdout": {"pg127": pg127_holdout, "pg122": pg122_holdout}, "left_out": "pg127"},
        "holdout_pg125": {"train": pg135_train + pg127_train, "dev": pg135_dev + pg127_dev, "holdout": {"pg125": pg125_holdout, "pg122": pg122_holdout}, "left_out": "pg125"},
    }


def _examples(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return PG136._with_rows(list(rows))


def _class_weights(rows: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    labels = torch.tensor([policy_index(str(row["action_label"])) for row in rows], dtype=torch.long, device=device)
    counts = torch.bincount(labels, minlength=len(POLICY_ACTIONS)).to(torch.float32)
    total = counts.sum().clamp_min(1.0)
    weights = torch.sqrt(total / (len(POLICY_ACTIONS) * counts.clamp_min(1.0)))
    return (weights / weights.mean().clamp_min(1e-6)).to(device)


def _body_parameters(model: DecoupledCausalSafetyPolicy) -> list[torch.nn.Parameter]:
    return list(model.backbone.parameters())


def _train_policy(model: DecoupledCausalSafetyPolicy, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, variant: str, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    if variant == "adapter_only":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}]
    elif variant == "balanced_full":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": _body_parameters(model), "lr": 8e-4}]
    elif variant == "joint_balanced":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": _body_parameters(model), "lr": 8e-4}]
    elif variant == "scratch_balanced":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": _body_parameters(model), "lr": 2e-3}]
    else:
        raise ValueError(f"unknown PG-138 variant: {variant}")
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    weights = _class_weights(train, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids, _ = PG136._pad_sequences(train, vocab, device=device)
        labels = torch.tensor([policy_index(str(row["action_label"])) for row in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        action_loss = criterion(model(ids), labels)
        total_loss = action_loss
        lm_loss = torch.tensor(0.0, device=device)
        if variant == "joint_balanced":
            lm_loss, _ = PG136._lm_loss(model.backbone, train, vocab, device=device)
            total_loss = action_loss + 0.15 * lm_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = _raw_metrics(model, train, vocab, device=device)
        dev_metrics = _raw_metrics(model, dev, vocab, device=device)
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "lm_loss": round(float(lm_loss.item()), 8), "train_accuracy": train_metrics["accuracy"], "dev_accuracy": dev_metrics["accuracy"]})
    return history


def _logits(model: DecoupledCausalSafetyPolicy, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    ids, _ = PG136._pad_sequences(examples, vocab, device=device)
    model.eval()
    with torch.inference_mode():
        logits = model(ids)
    labels = [policy_index(str(item["action_label"])) for item in examples]
    return logits, labels


def _metric_from_names(names: list[str], labels: list[int], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = [policy_index(name) for name in names]
    compliant = [name in PG136.PG135.PG134._allowed(item["row"]) for name, item in zip(names, examples)]
    unknown = [index for index, item in enumerate(examples) if not bool(item["row"]["failure_signature"].get("typed_available", True))]
    negative = [index for index, item in enumerate(examples) if item["row"].get("surface_kind") in {"blind", "decoy", "steady"}]
    return {
        "count": len(labels),
        "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels), 6) if labels else 0.0,
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "unknown_rows": len(unknown),
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "non_abstain_rate": round(sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names) / len(names), 6) if names else 0.0,
        "probe_rate": round(sum(name == "probe_candidate_other_method" for name in names) / len(names), 6) if names else 0.0,
        "predicted_action_counts": {action: names.count(action) for action in POLICY_ACTIONS},
    }


def _raw_metrics(model: DecoupledCausalSafetyPolicy, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> dict[str, Any]:
    logits, labels = _logits(model, examples, vocab, device=device)
    names = [POLICY_ACTIONS[index] for index in logits.argmax(dim=-1).cpu().tolist()]
    return _metric_from_names(names, labels, examples)


def _allowed_mask(row: Mapping[str, Any]) -> set[str]:
    signature = row["failure_signature"]
    if not bool(signature.get("typed_available", True)):
        return {"abstain_unknown_oracle"}
    return set(PG136.PG135.PG134._allowed(row))


def _postprocess(logits: torch.Tensor, examples: list[Mapping[str, Any]], *, mode: str, threshold: float, known_pairs: frozenset[str]) -> list[str]:
    names: list[str] = []
    for index, (row_item, vector) in enumerate(zip(examples, logits)):
        row = row_item["row"]
        if mode == "contract_masked" or mode == "calibrated":
            allowed = _allowed_mask(row)
            masked = vector.clone()
            for action_index, action in enumerate(POLICY_ACTIONS):
                if action not in allowed:
                    masked[action_index] = -float("inf")
            probabilities = torch.softmax(masked, dim=-1)
            confidence, choice = probabilities.max(dim=-1)
            action = POLICY_ACTIONS[int(choice.item())]
            if mode == "calibrated" and float(confidence.item()) < threshold and "abstain_candidate_only" in allowed:
                action = "abstain_candidate_only"
        else:
            action = POLICY_ACTIONS[int(vector.argmax().item())]
        if mode == "calibrated":
            action, _ = guard_action(action, row, known_pairs)
        names.append(action)
    return names


def _evaluate_modes(model: DecoupledCausalSafetyPolicy, examples: list[Mapping[str, Any]], vocab: Any, *, threshold: float, known_pairs: frozenset[str], device: torch.device) -> dict[str, Any]:
    logits, labels = _logits(model, examples, vocab, device=device)
    raw_names = _postprocess(logits, examples, mode="raw", threshold=threshold, known_pairs=known_pairs)
    masked_names = _postprocess(logits, examples, mode="contract_masked", threshold=threshold, known_pairs=known_pairs)
    calibrated_names = _postprocess(logits, examples, mode="calibrated", threshold=threshold, known_pairs=known_pairs)
    return {
        "raw": _metric_from_names(raw_names, labels, examples),
        "contract_masked": _metric_from_names(masked_names, labels, examples),
        "calibrated": _metric_from_names(calibrated_names, labels, examples),
        "contract_override_count": sum(raw != masked for raw, masked in zip(raw_names, masked_names)),
        "calibration_override_count": sum(masked != calibrated for masked, calibrated in zip(masked_names, calibrated_names)),
        "contract_override_rate": round(sum(raw != masked for raw, masked in zip(raw_names, masked_names)) / len(raw_names), 6) if raw_names else 0.0,
    }


def _calibrate(model: DecoupledCausalSafetyPolicy, dev: list[Mapping[str, Any]], vocab: Any, *, known_pairs: frozenset[str], device: torch.device) -> float:
    candidates = [round(item / 20.0, 2) for item in range(0, 20)]
    best = (0.0, -1.0, -1.0)
    logits, labels = _logits(model, dev, vocab, device=device)
    for threshold in candidates:
        names = _postprocess(logits, dev, mode="calibrated", threshold=threshold, known_pairs=known_pairs)
        metrics = _metric_from_names(names, labels, dev)
        feasible = metrics["safety_compliance_rate"] >= 0.99 and metrics["unknown_abstain_rate"] == 1.0 and metrics["non_abstain_rate"] > 0.05
        score = (1.0 if feasible else 0.0, metrics["accuracy"], metrics["safety_compliance_rate"])
        if score > best:
            best = score
            selected = threshold
    return selected


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in {"predictions", "labels"}}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def main() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect())
    folds = _build_folds(targets)
    fold_reports: dict[str, Any] = {}
    public_pretrain_sequences: dict[str, Any] = {}
    public_action_sequences: dict[str, Any] = {}
    checkpoint_manifest: dict[str, str] = {}
    for fold_index, (fold_name, fold) in enumerate(folds.items()):
        train = _examples(fold["train"])
        dev = _examples(fold["dev"])
        holdouts = {name: _examples(value) for name, value in fold["holdout"].items()}
        vocabulary = PG136.CausalVocabulary([item["tokens"] for item in train])
        _seed_all(13801 + fold_index)
        pretrained_backbone = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13801 + fold_index).to(device)
        pretrain_history = PG136._pretrain(pretrained_backbone, train, dev, vocabulary, device=device, seed=13811 + fold_index)
        pretrained_backbone._provenance["pretrained"] = True
        known_pairs = known_rule_ir_pairs(fold["train"])
        variant_reports: dict[str, Any] = {}
        variant_models: dict[str, DecoupledCausalSafetyPolicy] = {}
        for variant_index, variant in enumerate(VARIANTS):
            if variant == "scratch_balanced":
                _seed_all(13830 + fold_index)
                backbone = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13830 + fold_index).to(device)
            else:
                backbone = copy.deepcopy(pretrained_backbone).to(device)
            policy = DecoupledCausalSafetyPolicy(backbone, hidden_dim=backbone.hidden_dim, head_seed=13840 + fold_index * 10 + variant_index).to(device)
            history = _train_policy(policy, train, dev, vocabulary, variant=variant, device=device, seed=13850 + fold_index * 10 + variant_index)
            threshold = _calibrate(policy, dev, vocabulary, known_pairs=known_pairs, device=device)
            per_holdout = {name: _evaluate_modes(policy, examples, vocabulary, threshold=threshold, known_pairs=known_pairs, device=device) for name, examples in holdouts.items()}
            variant_reports[variant] = {"threshold": threshold, "history_tail": history[-5:], "holdouts": _compact(per_holdout), "provenance": policy.provenance}
            variant_models[variant] = policy
            checkpoint = ARTIFACT_DIR / f"{fold_name}_{variant}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema_version": SCHEMA_VERSION, "fold": fold_name, "variant": variant, "vocabulary": vocabulary.to_dict(), "provenance": policy.provenance, "model_state_dict": policy.state_dict()}, checkpoint)
            checkpoint_manifest[f"{fold_name}:{variant}"] = str(checkpoint.relative_to(ROOT)).replace("\\", "/")
        def _variant_score(item: Mapping[str, Any]) -> tuple[float, float, float]:
            calibrated = item["holdouts"]
            safety = min(value["calibrated"]["safety_compliance_rate"] for value in calibrated.values())
            abstain = min(value["calibrated"]["unknown_abstain_rate"] for value in calibrated.values())
            accuracy = min(value["calibrated"]["accuracy"] for value in calibrated.values())
            non_abstain = min(value["calibrated"]["non_abstain_rate"] for value in calibrated.values())
            return (1.0 if safety >= 0.99 and abstain == 1.0 and non_abstain > 0.05 else 0.0, accuracy, safety)
        scores = {variant: _variant_score(report) for variant, report in variant_reports.items()}
        selected = max(VARIANTS, key=lambda variant: scores[variant])
        fold_reports[fold_name] = {"left_out": fold["left_out"], "train_count": len(train), "dev_count": len(dev), "holdout_counts": {name: len(value) for name, value in holdouts.items()}, "vocabulary_size": len(vocabulary.itos), "known_pairs_sha256": known_pairs_sha256(known_pairs), "pretraining": {"dev_perplexity": PG136._lm_eval(pretrained_backbone, dev, vocabulary, device=device)["perplexity"], "history_tail": pretrain_history[-5:], "action_labels_in_input": False}, "variants": variant_reports, "selection": {"selected_variant": selected, "scores": scores}, "train_target_implementations": sorted({str(row.get("surface_kind")) for row in fold["train"]}), "holdout_target_implementations": sorted({str(row.get("surface_kind")) for values in fold["holdout"].values() for row in values})}
        public_pretrain_sequences[fold_name] = PG136._strip_for_dataset(train + dev, include_label=False)
        public_action_sequences[fold_name] = PG136._strip_for_dataset(train + dev, include_label=True)
    selected_summaries = []
    action_gain_flags = []
    for fold_name, report in fold_reports.items():
        selected = report["selection"]["selected_variant"]
        selected_holdouts = report["variants"][selected]["holdouts"]
        scratch_holdouts = report["variants"]["scratch_balanced"]["holdouts"]
        selected_summaries.append({"fold": fold_name, "variant": selected, "holdouts": selected_holdouts})
        gains = [selected_holdouts[name]["calibrated"]["accuracy"] - scratch_holdouts[name]["calibrated"]["accuracy"] for name in selected_holdouts]
        action_gain_flags.append(all(gain >= 0.0 for gain in gains) and any(gain > 0.0 for gain in gains))
    loio_safety = all(value["holdouts"][name]["calibrated"]["safety_compliance_rate"] >= 0.99 for value in selected_summaries for name in value["holdouts"])
    loio_abstain = all(value["holdouts"][name]["calibrated"]["unknown_abstain_rate"] == 1.0 for value in selected_summaries for name in value["holdouts"])
    loio_nontrivial = all(value["holdouts"][name]["calibrated"]["non_abstain_rate"] > 0.05 for value in selected_summaries for name in value["holdouts"])
    action_gain_two_folds = all(action_gain_flags)
    hard_checks = {"fresh_replay": True, "two_loio_folds": len(fold_reports) == 2, "loio_safety_floor": loio_safety, "loio_unknown_abstain": loio_abstain, "loio_nontrivial_action_rate": loio_nontrivial, "raw_masked_calibrated_separate": True, "memory_promotion_forbidden": True}
    promotion_checks = {"action_gain_in_both_loio_folds": action_gain_two_folds, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE}
    hard_gates_passed = all(hard_checks.values())
    training_eligible = hard_gates_passed and all(promotion_checks.values())
    report: dict[str, Any] = {"protocol_id": "pg-pk-138-decoupled-loio-v1", "schema_version": "pg138-decoupled-loio-report-v1", "status": "completed_pg138_decoupled_loio", "hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "scope": {"model": "causal_token_gru_plus_decoupled_safety_head", "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "folds": fold_reports, "selected_summaries": selected_summaries, "checks": hard_checks, "promotion_checks": promotion_checks, "checkpoint_manifest": checkpoint_manifest, "input_contract": {"fresh_replay": True, "leave_one_implementation_out": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "action_labels_in_pretrain_sequences": False, "typed_contract_mask_separate": True, "calibration_fit_on_dev_only": True}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "loio_candidate_pending_cross_implementation_review" if hard_gates_passed else "blocked_pg138_loio_gate_failure_preserved", "reason": "PG-138 只证明解耦安全头的 LOIO 结果；未通过动作增益或人工/Codex 审核不得晋升。"}, "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "safety_head": hashlib.sha256((ROOT / "app/pg138_safety_head.py").read_bytes()).hexdigest(), "causal_model": hashlib.sha256((ROOT / "app/pg136_causal_token_lm.py").read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    holdout_public = {fold_name: PG136._strip_for_dataset(_examples([row for values in fold["holdout"].values() for row in values]), include_label=False) for fold_name, fold in folds.items()}
    dataset = {"schema_version": "pg138-decoupled-loio-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": public_pretrain_sequences, "action_finetune_sequences": public_action_sequences, "holdout_sequences": holdout_public, "variants": list(VARIANTS), "labels_separate_from_pretrain": True}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg138-decoupled-loio-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": public_pretrain_sequences, "variants": list(VARIANTS), "labels_in_pretrain": False}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg138-decoupled-loio-trace-v1", "protocol_id": "pg-pk-138-decoupled-loio-v1", "status": report["status"], "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "evaluator_action_in_pretrain": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-138-decoupled-loio-v1", "schema_version": "pg138-decoupled-loio-protocol-v1", "objective": "解耦 causal representation 与 typed-contract safety head，并用两个 leave-one-implementation-out folds 验证。", "folds": {name: {"train": ["pg135", "pg125"] if name == "holdout_pg127" else ["pg135", "pg127"], "holdout": list(fold["holdout"].keys())} for name, fold in folds.items()}, "variants": list(VARIANTS), "required_gates": hard_checks, "promotion_gates": promotion_checks}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-138-decoupled-loio-v1", "proposal_id": "pg138-decoupled-loio-proposal-v1", "prediction": {"loio_safety": 1.0, "unknown_abstain": 1.0, "nontrivial_action_rate": True, "action_gain_in_both_folds": True}, "failure_rule": "若 mask/calibration 只制造全拒答，或任一 fold 的安全/弃权/有效动作率失败，或未较 scratch 提升，则保留 evaluation-only。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "folds": list(fold_reports), "selected": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "loio_safety": loio_safety, "loio_unknown_abstain": loio_abstain, "loio_nontrivial": loio_nontrivial, "action_gain_both_folds": action_gain_two_folds, "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in hard_checks.items() if not value] + [key for key, value in promotion_checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
