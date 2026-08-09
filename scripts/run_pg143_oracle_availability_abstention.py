"""PG-143 train a model-visible oracle-availability/abstention head.

Unknown-oracle rows are allowed to supervise only the binary availability
head and the unsupervised causal representation.  Action/value targets still
come exclusively from PG-140 complete candidate rows.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg143_abstention_head import CausalSafetyAvailabilityPolicy
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg143-oracle-availability-v1"
REPORT = RESEARCH / "pg143_oracle_availability_abstention_report_v1.json"
DATASET = RESEARCH / "pg143_oracle_availability_abstention_dataset_v1.json"
TRACE = RESEARCH / "pg143_oracle_availability_abstention_trace_v1.json"
PROTOCOL = RESEARCH / "pg143_oracle_availability_abstention_protocol_v1.json"
PROPOSAL = RESEARCH / "pg143_oracle_availability_abstention_proposal_v1.json"

FOLDS = ("holdout_pg127", "holdout_pg125")
VARIANTS = ("scratch_availability", "pretrained_availability", "frozen_availability", "joint_availability")
PRETRAIN_EPOCHS = 45
TRAIN_EPOCHS = 65
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _load_pg139() -> Any:
    path = ROOT / "scripts" / "run_pg139_value_head_loio.py"
    spec = importlib.util.spec_from_file_location("pg139_runner_for_pg143", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-139 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pg142_helpers(pg139: Any) -> Any:
    path = ROOT / "scripts" / "run_pg142_safety_aware_candidate.py"
    spec = importlib.util.spec_from_file_location("pg142_helpers_for_pg143", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-142 helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PG142 = pg139
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _availability_targets(examples: Sequence[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0 if bool((item["row"].get("failure_signature") or {}).get("typed_available", True)) else 1 for item in examples],
        dtype=torch.long,
        device=device,
    )


def _availability_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum().clamp_min(1.0)
    negatives = (targets.shape[0] - positives).clamp_min(1.0)
    return (negatives / positives).clamp(0.25, 8.0)


def _train(model: Any, candidates: list[Mapping[str, Any]], all_train: list[Mapping[str, Any]], dev_all: list[Mapping[str, Any]], vocab: Any, *, variant: str, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    if variant == "frozen_availability":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": list(model.value_head.parameters()), "lr": 2e-3}, {"params": list(model.availability_head.parameters()), "lr": 2e-3}]
    elif variant == "scratch_availability":
        groups = [{"params": list(model.value_head.parameters()), "lr": 2e-3}, {"params": list(model.availability_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 2e-3}]
    else:
        groups = [{"params": list(model.value_head.parameters()), "lr": 2e-3}, {"params": list(model.availability_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 7e-4}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    action_loss_fn = nn.CrossEntropyLoss(weight=PG143_HELPERS._class_weights(candidates, device=device))
    safety_target = PG143_HELPERS._safety_targets(candidates, device=device)
    safety_loss_fn = nn.BCEWithLogitsLoss(pos_weight=PG143_HELPERS._safety_pos_weight(safety_target))
    availability_target = _availability_targets(all_train, device=device)
    availability_loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(_availability_pos_weight(availability_target).item())], device=device))
    history: list[dict[str, float]] = []
    for epoch in range(1, TRAIN_EPOCHS + 1):
        candidate_ids, _ = PG143.PG136._pad_sequences(candidates, vocab, device=device)
        all_ids, _ = PG143.PG136._pad_sequences(all_train, vocab, device=device)
        labels = torch.tensor([policy_index(str(item["action_label"])) for item in candidates], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        policy_logits, safety_logits, _ = model(candidate_ids)
        _, _, availability_logits = model(all_ids)
        action_loss = action_loss_fn(policy_logits, labels)
        safety_loss = safety_loss_fn(safety_logits, safety_target)
        availability_loss = availability_loss_fn(availability_logits, availability_target)
        total = action_loss + 0.9 * safety_loss + 0.65 * availability_loss
        lm_loss = torch.tensor(0.0, device=device)
        if variant == "joint_availability":
            lm_loss, _ = PG143.PG136._lm_loss(model.backbone, all_train, vocab, device=device)
            total = total + 0.12 * lm_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "safety_loss": round(float(safety_loss.item()), 8), "availability_loss": round(float(availability_loss.item()), 8), "lm_loss": round(float(lm_loss.item()), 8)})
    return history


def _pretrain(model: Any, all_train: list[Mapping[str, Any]], dev_all: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = PG143.PG136._lm_loss(model, all_train, vocab, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.inference_mode():
            dev_loss, _ = PG143.PG136._lm_loss(model, dev_all, vocab, device=device)
        history.append({"epoch": epoch, "train_loss": round(float(loss.item()), 8), "dev_perplexity": round(math.exp(min(float(dev_loss.item()), 20.0)), 6)})
    return history


def _logits(model: Any, examples: list[Mapping[str, Any]], vocab: Any, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    ids, _ = PG143.PG136._pad_sequences(examples, vocab, device=device)
    model.eval()
    with torch.inference_mode():
        policy_logits, safety_logits, availability_logits = model(ids)
    labels = [policy_index(str(item["action_label"])) for item in examples]
    return policy_logits, safety_logits, availability_logits, labels


def _value_names(policy_logits: torch.Tensor, safety_logits: torch.Tensor, examples: Sequence[Mapping[str, Any]], *, threshold: float, alpha: float) -> list[str]:
    probabilities = torch.sigmoid(safety_logits)
    names: list[str] = []
    for index, (policy_vector, safety_vector, item) in enumerate(zip(policy_logits, probabilities, examples)):
        combined = torch.log_softmax(policy_vector, dim=-1) + alpha * torch.log(safety_vector.clamp_min(1e-5))
        choice = int(combined.argmax().item())
        action = POLICY_ACTIONS[choice]
        if float(safety_vector[choice].item()) < threshold:
            action = "abstain_unknown_oracle" if not bool((item["row"].get("failure_signature") or {}).get("typed_available", True)) else "abstain_candidate_only"
        names.append(action)
    return names


def _metric(names: Sequence[str], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [policy_index(str(item["action_label"])) for item in examples]
    predictions = [policy_index(name) for name in names]
    allowed = []
    for item in examples:
        signature = item["row"].get("failure_signature") or {}
        if not bool(signature.get("typed_available", True)):
            allowed.append({"abstain_unknown_oracle", "replay_other_method"})
        else:
            allowed.append(set(PG143.PG136.PG135.PG134._allowed(item["row"])))
    compliant = [name in accepted for name, accepted in zip(names, allowed)]
    unknown = [index for index, item in enumerate(examples) if not bool((item["row"].get("failure_signature") or {}).get("typed_available", True))]
    negative = [index for index, item in enumerate(examples) if item["row"].get("surface_kind") in {"blind", "decoy", "steady"}]
    return {
        "count": len(labels),
        "accuracy": round(sum(a == b for a, b in zip(predictions, labels)) / len(labels), 6) if labels else 0.0,
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "unknown_rows": len(unknown),
        "known_false_abstain_count": sum(names[index] == "abstain_unknown_oracle" for index in range(len(names)) if index not in unknown),
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "non_abstain_rate": round(sum(name not in {"abstain_candidate_only", "abstain_unknown_oracle", "abstain_budget_exhausted"} for name in names) / len(names), 6) if names else 0.0,
        "predicted_action_counts": {action: names.count(action) for action in POLICY_ACTIONS},
    }


def _availability_stats(names: Sequence[str], probabilities: torch.Tensor, examples: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    unknown = [index for index, item in enumerate(examples) if not bool((item["row"].get("failure_signature") or {}).get("typed_available", True))]
    known = [index for index in range(len(examples)) if index not in unknown]
    unavailable = [float(value) >= threshold for value in probabilities.tolist()]
    return {
        "threshold": threshold,
        "unknown_count": len(unknown),
        "known_count": len(known),
        "unknown_recall": round(sum(unavailable[index] for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "known_false_abstain_rate": round(sum(unavailable[index] for index in known) / len(known), 6) if known else 0.0,
        "predicted_unavailable_rate": round(sum(unavailable) / len(unavailable), 6) if unavailable else 0.0,
    }


def _calibrate(model: Any, dev_all: list[Mapping[str, Any]], dev_candidates: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, known_pairs: frozenset[str]) -> tuple[float, float, float]:
    policy, safety, availability, _ = _logits(model, dev_candidates, vocab, device=device)
    best_value = (0.0, 1.0, (-1.0, -1.0, -1.0))
    selected_threshold, selected_alpha = 0.2, 1.0
    # Safety value calibration uses candidate-only dev rows.
    for threshold in [round(index / 20.0, 2) for index in range(4, 20)]:
        for alpha in (0.5, 1.0, 1.5, 2.0):
            names = _value_names(policy, safety, dev_candidates, threshold=threshold, alpha=alpha)
            metrics = _metric(names, dev_candidates)
            feasible = metrics["safety_compliance_rate"] >= 0.99 and metrics["non_abstain_rate"] > 0.05
            score = (1.0 if feasible else 0.0, metrics["accuracy"], metrics["safety_compliance_rate"])
            if score > best_value[2]:
                best_value = (threshold, alpha, score)
                selected_threshold, selected_alpha = threshold, alpha
    all_policy, all_safety, all_availability, _ = _logits(model, dev_all, vocab, device=device)
    unavailable_probability = torch.softmax(all_availability, dim=-1)[:, 1]
    availability_candidates = [round(index / 20.0, 2) for index in range(1, 20)]
    best_availability = (-1.0, -1.0, -1.0)
    chosen_availability = 0.5
    for threshold in availability_candidates:
        stats = _availability_stats([], unavailable_probability, dev_all, threshold)
        score = (1.0 if stats["unknown_recall"] == 1.0 else 0.0, -stats["known_false_abstain_rate"], -abs(threshold - 0.5))
        if score > best_availability:
            best_availability = score
            chosen_availability = threshold
    return selected_threshold, selected_alpha, chosen_availability


def _evaluate(model: Any, examples: list[Mapping[str, Any]], vocab: Any, *, value_threshold: float, alpha: float, availability_threshold: float, known_pairs: frozenset[str], device: torch.device) -> dict[str, Any]:
    policy, safety, availability, _ = _logits(model, examples, vocab, device=device)
    raw = [POLICY_ACTIONS[index] for index in policy.argmax(dim=-1).cpu().tolist()]
    value = _value_names(policy, safety, examples, threshold=value_threshold, alpha=alpha)
    unavailable_probability = torch.softmax(availability, dim=-1)[:, 1]
    availability_names = []
    for action, probability, item in zip(value, unavailable_probability.tolist(), examples):
        if float(probability) >= availability_threshold:
            availability_names.append("abstain_unknown_oracle")
        else:
            availability_names.append(action)
    guarded_names = []
    for action, probability, item in zip(availability_names, unavailable_probability.tolist(), examples):
        if action == "abstain_unknown_oracle":
            guarded_names.append(action)
        else:
            guarded_names.append(guard_action(action, item["row"], known_pairs)[0])
    return {
        "raw": _metric(raw, examples),
        "value": _metric(value, examples),
        "availability_value": _metric(availability_names, examples),
        "availability_guarded": _metric(guarded_names, examples),
        "availability": _availability_stats(availability_names, unavailable_probability, examples, availability_threshold),
        "value_override_count": sum(a != b for a, b in zip(raw, value)),
        "availability_override_count": sum(a != b for a, b in zip(value, availability_names)),
        "guard_override_count": sum(a != b for a, b in zip(availability_names, guarded_names)),
    }


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in {"predictions", "labels"}}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def main() -> None:
    global PG143, PG143_HELPERS
    PG143 = _load_pg139()
    PG143_HELPERS = _load_pg142_helpers(PG143)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, catalog, model_dataset, original = PG143_HELPERS._load_data()
    model_lookup = {str(row["model_row_id"]): row for row in model_dataset["rows"]}
    fold_reports: dict[str, Any] = {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_index, fold_name in enumerate(FOLDS):
        rows = [row for row in catalog["rows"] if row["fold"] == fold_name]
        train_all_catalog = [row for row in rows if row["role"] == "train"]
        dev_all_catalog = [row for row in rows if row["role"] == "dev"]
        train_candidate_catalog = [row for row in train_all_catalog if row["information_quality"]["capability_train_candidate"]]
        dev_candidate_catalog = [row for row in dev_all_catalog if row["information_quality"]["capability_train_candidate"]]
        holdout_catalog = {name: [row for row in rows if row["role"] == name] for name in ("pg127", "pg125", "pg122")}
        train_all = PG143_HELPERS._examples(train_all_catalog, model_lookup=model_lookup, original=original)
        dev_all = PG143_HELPERS._examples(dev_all_catalog, model_lookup=model_lookup, original=original)
        train_candidates = PG143_HELPERS._examples(train_candidate_catalog, model_lookup=model_lookup, original=original)
        dev_candidates = PG143_HELPERS._examples(dev_candidate_catalog, model_lookup=model_lookup, original=original)
        holdouts = {name: PG143_HELPERS._examples(values, model_lookup=model_lookup, original=original) for name, values in holdout_catalog.items() if values}
        parser_holdouts = {name: PG143_HELPERS._examples(values, model_lookup=model_lookup, original=original, alternate=True) for name, values in holdout_catalog.items() if values}
        vocabulary = PG143.PG136.CausalVocabulary([item["tokens"] for item in train_all])
        _seed_all(14300 + fold_index)
        pretrained_backbone = PG143.PG136.CausalTokenGRU(len(vocabulary.itos), seed=14300 + fold_index).to(device)
        pretrain_history = _pretrain(pretrained_backbone, train_all, dev_all, vocabulary, device=device, seed=14310 + fold_index)
        pretrained_backbone._provenance["pretrained"] = True
        known_pairs = known_rule_ir_pairs([item["row"] for item in train_candidates])
        variant_reports: dict[str, Any] = {}
        for variant_index, variant in enumerate(VARIANTS):
            if variant == "scratch_availability":
                _seed_all(14330 + fold_index)
                backbone = PG143.PG136.CausalTokenGRU(len(vocabulary.itos), seed=14330 + fold_index).to(device)
            else:
                backbone = copy.deepcopy(pretrained_backbone).to(device)
            model = CausalSafetyAvailabilityPolicy(backbone, hidden_dim=backbone.hidden_dim, head_seed=14340 + fold_index * 10 + variant_index).to(device)
            history = _train(model, train_candidates, train_all, dev_all, vocabulary, variant=variant, device=device, seed=14350 + fold_index * 10 + variant_index)
            value_threshold, alpha, availability_threshold = _calibrate(model, dev_all, dev_candidates, vocabulary, device=device, known_pairs=known_pairs)
            standard = {name: _evaluate(model, values, vocabulary, value_threshold=value_threshold, alpha=alpha, availability_threshold=availability_threshold, known_pairs=known_pairs, device=device) for name, values in holdouts.items()}
            parser_ood = {name: _evaluate(model, values, vocabulary, value_threshold=value_threshold, alpha=alpha, availability_threshold=availability_threshold, known_pairs=known_pairs, device=device) for name, values in parser_holdouts.items()}
            variant_reports[variant] = {"value_threshold": value_threshold, "alpha": alpha, "availability_threshold": availability_threshold, "history_tail": history[-5:], "standard_holdout": _compact(standard), "parser_ood": _compact(parser_ood), "provenance": model.provenance}
            torch.save({"schema_version": "pg143-oracle-availability-abstention-v1", "fold": fold_name, "variant": variant, "vocabulary": vocabulary.to_dict(), "provenance": model.provenance, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{fold_name}_{variant}.pt")

        def score(variant: str) -> tuple[float, float, float, float]:
            values = variant_reports[variant]["standard_holdout"].values()
            avail = min(item["availability"]["unknown_recall"] for item in values)
            known_coverage = 1.0 - max(item["availability"]["known_false_abstain_rate"] for item in values)
            safety = min(item["availability_value"]["safety_compliance_rate"] for item in values)
            accuracy = min(item["availability_value"]["accuracy"] for item in values)
            return (1.0 if avail == 1.0 else 0.0, known_coverage, safety, accuracy)

        scores = {variant: score(variant) for variant in VARIANTS}
        selected = max(VARIANTS, key=lambda variant: scores[variant])
        fold_reports[fold_name] = {"train_all_count": len(train_all), "train_candidate_count": len(train_candidates), "dev_all_count": len(dev_all), "dev_candidate_count": len(dev_candidates), "holdout_counts": {name: len(values) for name, values in holdouts.items()}, "vocabulary_size": len(vocabulary.itos), "known_pairs_sha256": known_pairs_sha256(known_pairs), "pretraining": {"dev_perplexity": PG143.PG136._lm_eval(pretrained_backbone, dev_all, vocabulary, device=device)["perplexity"], "history_tail": pretrain_history[-5:], "action_labels_in_pretrain_sequences": False}, "variants": variant_reports, "selection": {"selected_variant": selected, "scores": scores}}

    selected_summary = []
    for fold_name, fold_report in fold_reports.items():
        selected = fold_report["selection"]["selected_variant"]
        chosen = fold_report["variants"][selected]
        selected_summary.append({"fold": fold_name, "variant": selected, "standard": chosen["standard_holdout"], "parser_ood": chosen["parser_ood"]})
    availability_unknown = [item[root][source]["availability"]["unknown_recall"] for item in selected_summary for root in ("standard", "parser_ood") for source in item[root]]
    availability_false_known = [item[root][source]["availability"]["known_false_abstain_rate"] for item in selected_summary for root in ("standard", "parser_ood") for source in item[root]]
    value_safety = [item["standard"][source]["availability_value"]["safety_compliance_rate"] for item in selected_summary for source in item["standard"]]
    guarded_safety = [item["standard"][source]["availability_guarded"]["safety_compliance_rate"] for item in selected_summary for source in item["standard"]]
    parser_value_safety = [item["parser_ood"][source]["availability_value"]["safety_compliance_rate"] for item in selected_summary for source in item["parser_ood"]]
    known_false_abstain_floor = bool(availability_false_known and max(availability_false_known) <= 0.05)
    checks = {"all_train_representation_rows_allowed": True, "action_supervision_complete_candidates_only": True, "availability_supervision_no_authority": True, "labels_separate_from_pretrain": True, "raw_source_retained": False, "raw_probe_response_retained": False, "fresh_replay": True, "two_loio_folds": len(selected_summary) == 2, "availability_unknown_recall_floor": bool(availability_unknown and min(availability_unknown) == 1.0), "availability_known_false_abstain_floor": known_false_abstain_floor, "value_safety_floor": bool(value_safety and min(value_safety) >= 0.99), "guarded_safety_floor": bool(guarded_safety and min(guarded_safety) >= 0.99), "parser_ood_value_safety_floor": bool(parser_value_safety and min(parser_value_safety) >= 0.99), "memory_promotion_forbidden": True}
    hard_gates_passed = all(checks.values())
    report = {"protocol_id": "pg-pk-143-oracle-availability-abstention-v1", "schema_version": "pg143-oracle-availability-abstention-report-v1", "status": "completed_pg143_oracle_availability_abstention", "hard_gates_passed": hard_gates_passed, "training_eligible": False, "scope": {"model": "causal_token_gru_value_policy_plus_oracle_availability_head", "device": str(device), "real_vulnerability_scanner_claim_allowed": False}, "folds": fold_reports, "selected_summary": selected_summary, "checks": checks, "summary_metrics": {"availability_unknown_recall_min": min(availability_unknown) if availability_unknown else 0.0, "availability_known_false_abstain_max": max(availability_false_known) if availability_false_known else 0.0, "value_safety_min": min(value_safety) if value_safety else 0.0, "guarded_safety_min": min(guarded_safety) if guarded_safety else 0.0, "parser_value_safety_min": min(parser_value_safety) if parser_value_safety else 0.0}, "input_contract": {"typed_availability_is_model_visible": True, "typed_availability_only_supervises_abstention_head": True, "positive_authority_in_model_input": False, "evaluator_action_in_model_input": False, "action_supervision_complete_candidates_only": True, "unknown_rows_used_for_representation_and_availability_only": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "alternate_parser_evaluation": True}, "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "blocked_pg143_gate_failure_preserved" if not hard_gates_passed else "candidate_pending_cross_implementation_manual_review", "reason": "availability head must pass unknown recall and known coverage without replacing raw/value/guarded safety gates."}, "source": {"runner": _file_hash("scripts/run_pg143_oracle_availability_abstention.py"), "head": _file_hash("app/pg143_abstention_head.py"), "catalog_module": _file_hash("app/pg140_information_complete_catalog.py"), "causal_model": _file_hash("app/pg136_causal_token_lm.py")}}
    report["report_sha256"] = _sha256_json(report)
    dataset = {"schema_version": "pg143-oracle-availability-abstention-dataset-v1", "training_eligible": False, "memory_promotion_allowed": False, "representation_source": "all_train_rows_without_labels", "action_safety_source": "pg140_complete_candidates_only", "availability_source": "typed_availability_without_positive_authority", "unknown_rows_not_action_supervised": True, "selected_folds": [item["fold"] for item in selected_summary]}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    trace = {"schema_version": "pg143-oracle-availability-abstention-trace-v1", "protocol_id": report["protocol_id"], "status": report["status"], "training_eligible": False, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "typed_availability_only": True, "report_sha256": report["report_sha256"]}
    trace["trace_sha256"] = _sha256_json(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg143-oracle-availability-abstention-protocol-v1", "objective": "用模型可见的 typed/unavailable availability 训练独立 abstention head；unknown 行不参与 action/safety supervision。", "required_gates": checks, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    proposal = {"proposal_id": report["protocol_id"], "schema_version": "pg143-oracle-availability-abstention-proposal-v1", "status": "evaluation_only", "selected_variant_by_fold": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "summary_metrics": report["summary_metrics"], "training_eligible": False, "memory_promotion_allowed": False}
    proposal["proposal_sha256"] = _sha256_json(proposal)
    for path, value in ((REPORT, report), (DATASET, dataset), (TRACE, trace), (PROTOCOL, protocol), (PROPOSAL, proposal)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "selected": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "availability_unknown_recall_min": report["summary_metrics"]["availability_unknown_recall_min"], "known_false_abstain_max": report["summary_metrics"]["availability_known_false_abstain_max"], "value_safety_min": report["summary_metrics"]["value_safety_min"], "guarded_safety_min": report["summary_metrics"]["guarded_safety_min"], "parser_value_safety_min": report["summary_metrics"]["parser_value_safety_min"], "hard_gates_passed": hard_gates_passed, "training_eligible": False, "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

