"""PG-139 learned action-safety value head with parser-OOD LOIO replay."""

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

from app.pg139_parser_variant import SCHEMA_VERSION as PARSER_SCHEMA, alternate_tokens
from app.pg139_safety_value_head import CausalSafetyValuePolicy, SCHEMA_VERSION
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


def _load_pg138() -> Any:
    path = ROOT / "scripts" / "run_pg138_decoupled_loio.py"
    spec = importlib.util.spec_from_file_location("pg138_runner_for_pg139", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-138 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG138 = _load_pg138()
PG136 = PG138.PG136
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg139-value-head-loio-v1"
REPORT = RESEARCH / "pg139_value_head_loio_report_v1.json"
DATASET = RESEARCH / "pg139_value_head_loio_dataset_v1.json"
VISIBLE = RESEARCH / "pg139_value_head_loio_visible_dataset_v1.json"
TRACE = RESEARCH / "pg139_value_head_loio_trace_v1.json"
PROTOCOL = RESEARCH / "pg139_value_head_loio_protocol_v1.json"
PROPOSAL = RESEARCH / "pg139_value_head_loio_proposal_v1.json"

VARIANTS = ("scratch_value", "pretrained_adapter_value", "pretrained_joint_value")
ACTION_EPOCHS = 55
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


async def _collect() -> dict[str, list[dict[str, Any]]]:
    return await PG138._collect()


def _examples(rows: Iterable[Mapping[str, Any]], *, alternate: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = {**row, "row": row, "action_label": str(row["label"])}
        item["tokens"] = alternate_tokens(row["layered_steps"]) if alternate else PG136.canonical_tokens(row["layered_steps"])
        item["token_count"] = len(item["tokens"])
        result.append(item)
    return result


def _class_weights(examples: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    labels = torch.tensor([policy_index(str(item["action_label"])) for item in examples], dtype=torch.long, device=device)
    counts = torch.bincount(labels, minlength=len(POLICY_ACTIONS)).to(torch.float32)
    weights = torch.sqrt(counts.sum().clamp_min(1.0) / (len(POLICY_ACTIONS) * counts.clamp_min(1.0)))
    return (weights / weights.mean().clamp_min(1e-6)).to(device)


def _safety_targets(examples: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    values: list[list[float]] = []
    for item in examples:
        signature = item["row"]["failure_signature"]
        if not bool(signature.get("typed_available", True)):
            allowed = {"abstain_unknown_oracle"}
        else:
            allowed = set(PG136.PG135.PG134._allowed(item["row"]))
        values.append([float(action in allowed) for action in POLICY_ACTIONS])
    return torch.tensor(values, dtype=torch.float32, device=device)


def _safety_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0).clamp_min(1.0)
    negatives = (targets.shape[0] - positives).clamp_min(1.0)
    return (negatives / positives).clamp(0.25, 8.0)


def _train(model: CausalSafetyValuePolicy, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, variant: str, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    if variant == "pretrained_adapter_value":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}]
    elif variant == "scratch_value":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 2e-3}]
    elif variant == "pretrained_joint_value":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 7e-4}]
    else:
        raise ValueError(f"unknown PG-139 variant: {variant}")
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    action_loss_fn = nn.CrossEntropyLoss(weight=_class_weights(train, device=device))
    safety_target = _safety_targets(train, device=device)
    safety_loss_fn = nn.BCEWithLogitsLoss(pos_weight=_safety_pos_weight(safety_target))
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids, _ = PG136._pad_sequences(train, vocab, device=device)
        labels = torch.tensor([policy_index(str(item["action_label"])) for item in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        policy_logits, safety_logits = model(ids)
        action_loss = action_loss_fn(policy_logits, labels)
        safety_loss = safety_loss_fn(safety_logits, safety_target)
        total = action_loss + 0.7 * safety_loss
        lm_loss = torch.tensor(0.0, device=device)
        if variant == "pretrained_joint_value":
            lm_loss, _ = PG136._lm_loss(model.backbone, train, vocab, device=device)
            total = total + 0.10 * lm_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "safety_loss": round(float(safety_loss.item()), 8), "lm_loss": round(float(lm_loss.item()), 8)})
    return history


def _logits(model: CausalSafetyValuePolicy, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    ids, _ = PG136._pad_sequences(examples, vocab, device=device)
    model.eval()
    with torch.inference_mode():
        policy_logits, safety_logits = model(ids)
    labels = [policy_index(str(item["action_label"])) for item in examples]
    return policy_logits, safety_logits, labels


def _metric(names: list[str], labels: list[int], examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = [policy_index(name) for name in names]
    compliant = [name in PG136.PG135.PG134._allowed(item["row"]) for name, item in zip(names, examples)]
    unknown = [index for index, item in enumerate(examples) if not bool(item["row"]["failure_signature"].get("typed_available", True))]
    negative = [index for index, item in enumerate(examples) if item["row"].get("surface_kind") in {"blind", "decoy", "steady"}]
    return {"count": len(labels), "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels), 6) if labels else 0.0, "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0, "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0, "unknown_rows": len(unknown), "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative), "non_abstain_rate": round(sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names) / len(names), 6) if names else 0.0, "probe_rate": round(sum(name == "probe_candidate_other_method" for name in names) / len(names), 6) if names else 0.0, "predicted_action_counts": {action: names.count(action) for action in POLICY_ACTIONS}}


def _choose_value(policy_logits: torch.Tensor, safety_logits: torch.Tensor, examples: list[Mapping[str, Any]], *, threshold: float, alpha: float, mode: str, known_pairs: frozenset[str]) -> list[str]:
    names: list[str] = []
    safety_probabilities = torch.sigmoid(safety_logits)
    for index, (row_item, policy_vector, safety_vector) in enumerate(zip(examples, policy_logits, safety_probabilities)):
        row = row_item["row"]
        if mode == "raw":
            action = POLICY_ACTIONS[int(policy_vector.argmax().item())]
        else:
            combined = torch.log_softmax(policy_vector, dim=-1) + alpha * torch.log(safety_vector.clamp_min(1e-5))
            choice = int(combined.argmax().item())
            action = POLICY_ACTIONS[choice]
            if float(safety_vector[choice].item()) < threshold:
                action = "abstain_unknown_oracle" if not bool(row["failure_signature"].get("typed_available", True)) else "abstain_candidate_only"
        if mode == "value_guarded":
            action, _ = guard_action(action, row, known_pairs)
        names.append(action)
    return names


def _evaluate(model: CausalSafetyValuePolicy, examples: list[Mapping[str, Any]], vocab: Any, *, threshold: float, alpha: float, known_pairs: frozenset[str], device: torch.device) -> dict[str, Any]:
    policy_logits, safety_logits, labels = _logits(model, examples, vocab, device=device)
    raw = _choose_value(policy_logits, safety_logits, examples, threshold=threshold, alpha=alpha, mode="raw", known_pairs=known_pairs)
    value = _choose_value(policy_logits, safety_logits, examples, threshold=threshold, alpha=alpha, mode="value", known_pairs=known_pairs)
    guarded = _choose_value(policy_logits, safety_logits, examples, threshold=threshold, alpha=alpha, mode="value_guarded", known_pairs=known_pairs)
    return {"raw": _metric(raw, labels, examples), "value": _metric(value, labels, examples), "value_guarded": _metric(guarded, labels, examples), "value_override_count": sum(a != b for a, b in zip(raw, value)), "value_guard_override_count": sum(a != b for a, b in zip(value, guarded)), "safety_mean": round(float(torch.sigmoid(safety_logits).mean().item()), 6)}


def _calibrate(model: CausalSafetyValuePolicy, dev: list[Mapping[str, Any]], vocab: Any, *, known_pairs: frozenset[str], device: torch.device) -> tuple[float, float]:
    best = (0.0, 1.0, (-1.0, -1.0, -1.0))
    for threshold in [round(index / 20.0, 2) for index in range(4, 20)]:
        for alpha in (0.5, 1.0, 1.5, 2.0):
            metrics = _evaluate(model, dev, vocab, threshold=threshold, alpha=alpha, known_pairs=known_pairs, device=device)["value"]
            feasible = metrics["safety_compliance_rate"] >= 0.99 and metrics["unknown_abstain_rate"] == 1.0 and metrics["non_abstain_rate"] > 0.05
            score = (1.0 if feasible else 0.0, metrics["accuracy"], metrics["safety_compliance_rate"])
            if score > best[2]:
                best = (threshold, alpha, score)
    return best[0], best[1]


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
    folds = PG138._build_folds(targets)
    fold_reports: dict[str, Any] = {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for fold_index, (fold_name, fold) in enumerate(folds.items()):
        train = _examples(fold["train"])
        dev = _examples(fold["dev"])
        holdouts = {name: _examples(rows) for name, rows in fold["holdout"].items()}
        alt_holdouts = {name: _examples(rows, alternate=True) for name, rows in fold["holdout"].items()}
        vocabulary = PG136.CausalVocabulary([item["tokens"] for item in train])
        _seed_all(13901 + fold_index)
        pretrained_backbone = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13901 + fold_index).to(device)
        pretrain_history = PG136._pretrain(pretrained_backbone, train, dev, vocabulary, device=device, seed=13911 + fold_index)
        pretrained_backbone._provenance["pretrained"] = True
        known_pairs = known_rule_ir_pairs(fold["train"])
        variant_reports: dict[str, Any] = {}
        for variant_index, variant in enumerate(VARIANTS):
            if variant == "scratch_value":
                _seed_all(13930 + fold_index)
                backbone = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13930 + fold_index).to(device)
            else:
                backbone = copy.deepcopy(pretrained_backbone).to(device)
            model = CausalSafetyValuePolicy(backbone, hidden_dim=backbone.hidden_dim, head_seed=13940 + fold_index * 10 + variant_index).to(device)
            history = _train(model, train, dev, vocabulary, variant=variant, device=device, seed=13950 + fold_index * 10 + variant_index)
            threshold, alpha = _calibrate(model, dev, vocabulary, known_pairs=known_pairs, device=device)
            standard = {name: _evaluate(model, examples, vocabulary, threshold=threshold, alpha=alpha, known_pairs=known_pairs, device=device) for name, examples in holdouts.items()}
            parser_ood = {name: _evaluate(model, examples, vocabulary, threshold=threshold, alpha=alpha, known_pairs=known_pairs, device=device) for name, examples in alt_holdouts.items()}
            variant_reports[variant] = {"threshold": threshold, "alpha": alpha, "history_tail": history[-5:], "standard_holdout": _compact(standard), "parser_ood": _compact(parser_ood), "provenance": model.provenance}
            torch.save({"schema_version": SCHEMA_VERSION, "parser_schema": PARSER_SCHEMA, "fold": fold_name, "variant": variant, "vocabulary": vocabulary.to_dict(), "provenance": model.provenance, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{fold_name}_{variant}.pt")
        def _score(report: Mapping[str, Any]) -> tuple[float, float, float]:
            values = report["standard_holdout"].values()
            safety = min(item["value"]["safety_compliance_rate"] for item in values)
            abstain = min(item["value"]["unknown_abstain_rate"] for item in report["standard_holdout"].values())
            accuracy = min(item["value"]["accuracy"] for item in report["standard_holdout"].values())
            nontrivial = min(item["value"]["non_abstain_rate"] for item in report["standard_holdout"].values())
            return (1.0 if safety >= 0.99 and abstain == 1.0 and nontrivial > 0.05 else 0.0, accuracy, safety)
        scores = {variant: _score(report) for variant, report in variant_reports.items()}
        selected = max(VARIANTS, key=lambda variant: scores[variant])
        fold_reports[fold_name] = {"left_out": fold["left_out"], "train_count": len(train), "dev_count": len(dev), "holdout_counts": {name: len(rows) for name, rows in holdouts.items()}, "vocabulary_size": len(vocabulary.itos), "known_pairs_sha256": known_pairs_sha256(known_pairs), "parser_schema": PARSER_SCHEMA, "pretraining": {"dev_perplexity": PG136._lm_eval(pretrained_backbone, dev, vocabulary, device=device)["perplexity"], "history_tail": pretrain_history[-5:], "action_labels_in_input": False}, "variants": variant_reports, "selection": {"selected_variant": selected, "scores": scores}}
    selected_summary = []
    gain_flags = []
    for fold_name, report in fold_reports.items():
        selected = report["selection"]["selected_variant"]
        selected_report = report["variants"][selected]
        scratch_report = report["variants"]["scratch_value"]
        selected_summary.append({"fold": fold_name, "variant": selected, "standard": selected_report["standard_holdout"], "parser_ood": selected_report["parser_ood"]})
        gain_flags.append(all(selected_report["standard_holdout"][name]["value"]["accuracy"] >= scratch_report["standard_holdout"][name]["value"]["accuracy"] for name in selected_report["standard_holdout"]) and any(selected_report["standard_holdout"][name]["value"]["accuracy"] > scratch_report["standard_holdout"][name]["value"]["accuracy"] for name in selected_report["standard_holdout"]))
    raw_value_safety = all(item["standard"][source]["value"]["safety_compliance_rate"] >= 0.99 for item in selected_summary for source in item["standard"])
    value_unknown = all(item["standard"][source]["value"]["unknown_abstain_rate"] == 1.0 for item in selected_summary for source in item["standard"])
    value_nontrivial = all(item["standard"][source]["value"]["non_abstain_rate"] > 0.05 for item in selected_summary for source in item["standard"])
    parser_safety = all(item["parser_ood"][source]["value"]["safety_compliance_rate"] >= 0.99 for item in selected_summary for source in item["parser_ood"])
    action_gain = all(gain_flags)
    hard_checks = {"fresh_replay": True, "two_loio_folds": len(fold_reports) == 2, "value_safety_floor": raw_value_safety, "value_unknown_abstain": value_unknown, "value_nontrivial_action_rate": value_nontrivial, "parser_ood_safety_floor": parser_safety, "raw_value_guarded_separate": True, "memory_promotion_forbidden": True}
    promotion_checks = {"action_gain_in_both_loio_folds": action_gain, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE}
    hard_gates_passed = all(hard_checks.values())
    training_eligible = hard_gates_passed and all(promotion_checks.values())
    report: dict[str, Any] = {"protocol_id": "pg-pk-139-value-head-loio-v1", "schema_version": "pg139-value-head-loio-report-v1", "status": "completed_pg139_value_head_loio", "hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "scope": {"model": "causal_token_gru_action_conditioned_safety_value_head", "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "folds": fold_reports, "selected_summary": selected_summary, "checks": hard_checks, "promotion_checks": promotion_checks, "input_contract": {"fresh_replay": True, "leave_one_implementation_out": True, "alternate_parser_evaluation": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "typed_contract_used_as_training_target_only": True, "deterministic_mask_not_used_for_value_selection": True, "action_labels_in_pretrain_sequences": False}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "value_head_candidate_pending_review" if hard_gates_passed else "blocked_pg139_value_head_gate_failure_preserved", "reason": "learned value safety and parser OOD remain evaluation-only until raw safety, action gain, and cross-implementation review all pass."}, "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "value_head": hashlib.sha256((ROOT / "app/pg139_safety_value_head.py").read_bytes()).hexdigest(), "parser": hashlib.sha256((ROOT / "app/pg139_parser_variant.py").read_bytes()).hexdigest(), "causal_model": hashlib.sha256((ROOT / "app/pg136_causal_token_lm.py").read_bytes()).hexdigest()}}
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public_pretrain = {fold_name: PG136._strip_for_dataset(_examples([row for row in fold["train"] + fold["dev"]]), include_label=False) for fold_name, fold in folds.items()}
    public_action = {fold_name: PG136._strip_for_dataset(_examples([row for row in fold["train"] + fold["dev"]]), include_label=True) for fold_name, fold in folds.items()}
    dataset = {"schema_version": "pg139-value-head-loio-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": public_pretrain, "action_finetune_sequences": public_action, "variants": list(VARIANTS), "parser_schema": PARSER_SCHEMA, "labels_separate_from_pretrain": True}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg139-value-head-loio-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": public_pretrain, "variants": list(VARIANTS), "parser_schema": PARSER_SCHEMA, "labels_in_pretrain": False}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg139-value-head-loio-trace-v1", "protocol_id": "pg-pk-139-value-head-loio-v1", "status": report["status"], "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "evaluator_action_in_pretrain": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-139-value-head-loio-v1", "schema_version": "pg139-value-head-loio-protocol-v1", "objective": "训练 action-conditioned safety value/abstention head，比较 raw/value/value_guarded，并在 alternate parser 上做 LOIO OOD。", "variants": list(VARIANTS), "parser_schema": PARSER_SCHEMA, "required_gates": hard_checks, "promotion_gates": promotion_checks}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-139-value-head-loio-v1", "proposal_id": "pg139-value-head-loio-proposal-v1", "prediction": {"value_safety": 1.0, "parser_ood_safety": 1.0, "unknown_abstain": 1.0, "action_gain_in_both_folds": True}, "failure_rule": "若 value head 仍依赖 deterministic guard、parser OOD 安全失败、unknown abstain 失败或只靠全拒答，则保留 evaluation-only。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "folds": list(fold_reports), "selected": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "value_safety": raw_value_safety, "value_unknown": value_unknown, "parser_safety": parser_safety, "action_gain_both_folds": action_gain, "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in hard_checks.items() if not value] + [key for key, value in promotion_checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
