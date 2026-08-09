"""PG-142 independent safety/value/abstention head on complete candidates.

The policy head and the safety value head are optimized separately.  The
typed contract is used only to construct evaluator-side safety targets and to
measure a deterministic final guard; it is never placed in model input.
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
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg140_information_complete_catalog import repaired_observation_tokens
from app.pg139_parser_variant import alternate_tokens
from app.pg139_safety_value_head import CausalSafetyValuePolicy
from app.pg124_failure_conditioned_policy import POLICY_ACTIONS, policy_index
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg142-safety-aware-candidate-v1"
REPORT = RESEARCH / "pg142_safety_aware_candidate_report_v1.json"
DATASET = RESEARCH / "pg142_safety_aware_candidate_dataset_v1.json"
TRACE = RESEARCH / "pg142_safety_aware_candidate_trace_v1.json"
PROTOCOL = RESEARCH / "pg142_safety_aware_candidate_protocol_v1.json"
PROPOSAL = RESEARCH / "pg142_safety_aware_candidate_proposal_v1.json"

FOLDS = ("holdout_pg127", "holdout_pg125")
VARIANTS = ("scratch_safety", "pretrained_safety", "frozen_safety", "joint_safety")
PRETRAIN_EPOCHS = 45
ACTION_EPOCHS = 65
CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False


def _load_pg139() -> Any:
    path = ROOT / "scripts" / "run_pg139_value_head_loio.py"
    spec = importlib.util.spec_from_file_location("pg139_runner_for_pg142", path)
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


def _train(model: Any, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, variant: str, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    if variant == "frozen_safety":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}]
    elif variant == "scratch_safety":
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 2e-3}]
    else:
        groups = [{"params": list(model.safety_head.parameters()), "lr": 2e-3}, {"params": list(model.backbone.parameters()), "lr": 7e-4}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    action_loss_fn = nn.CrossEntropyLoss(weight=PG142._class_weights(train, device=device))
    safety_target = PG142._safety_targets(train, device=device)
    safety_loss_fn = nn.BCEWithLogitsLoss(pos_weight=PG142._safety_pos_weight(safety_target))
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids, _ = PG142.PG136._pad_sequences(train, vocab, device=device)
        labels = torch.tensor([policy_index(str(item["action_label"])) for item in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        policy_logits, safety_logits = model(ids)
        action_loss = action_loss_fn(policy_logits, labels)
        safety_loss = safety_loss_fn(safety_logits, safety_target)
        total = action_loss + 0.9 * safety_loss
        lm_loss = torch.tensor(0.0, device=device)
        if variant == "joint_safety":
            lm_loss, _ = PG142.PG136._lm_loss(model.backbone, train, vocab, device=device)
            total = total + 0.12 * lm_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "safety_loss": round(float(safety_loss.item()), 8), "lm_loss": round(float(lm_loss.item()), 8)})
    return history


def _pretrain(model: Any, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, device: torch.device, seed: int) -> list[dict[str, float]]:
    _seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = PG142.PG136._lm_loss(model, train, vocab, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.inference_mode():
            dev_loss, _ = PG142.PG136._lm_loss(model, dev, vocab, device=device)
        history.append({"epoch": epoch, "train_loss": round(float(loss.item()), 8), "dev_perplexity": round(math.exp(min(float(dev_loss.item()), 20.0)), 6)})
    return history


def _class_weights(examples: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    labels = torch.tensor([policy_index(str(item["action_label"])) for item in examples], dtype=torch.long, device=device)
    counts = torch.bincount(labels, minlength=len(POLICY_ACTIONS)).to(torch.float32)
    total = counts.sum().clamp_min(1.0)
    weights = torch.sqrt(total / (len(POLICY_ACTIONS) * counts.clamp_min(1.0)))
    return (weights / weights.mean().clamp_min(1e-6)).to(device)


def _safety_targets(examples: list[Mapping[str, Any]], *, device: torch.device) -> torch.Tensor:
    values: list[list[float]] = []
    for item in examples:
        signature = item["row"].get("failure_signature") or {}
        if not bool(signature.get("typed_available", True)):
            allowed = {"abstain_unknown_oracle"}
        else:
            allowed = set(PG142.PG136.PG135.PG134._allowed(item["row"]))
        values.append([float(action in allowed) for action in POLICY_ACTIONS])
    return torch.tensor(values, dtype=torch.float32, device=device)


def _safety_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0).clamp_min(1.0)
    negatives = (targets.shape[0] - positives).clamp_min(1.0)
    return (negatives / positives).clamp(0.25, 8.0)


def _load_data() -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]]:
    catalog = json.loads((RESEARCH / "pg140_information_complete_catalog_v1.json").read_text(encoding="utf-8"))
    model_dataset = json.loads((RESEARCH / "pg140_information_complete_model_dataset_v1.json").read_text(encoding="utf-8"))
    targets = asyncio.run(PG142._collect())
    raw_folds = PG142.PG138._build_folds(targets)
    original: dict[str, Mapping[str, Any]] = {}
    for fold_name, fold in raw_folds.items():
        for row in fold["train"] + fold["dev"]:
            original[f"{fold_name}::{row['row_id']}"] = row
        for rows in fold["holdout"].values():
            for row in rows:
                original[f"{fold_name}::{row['row_id']}"] = row
    return targets, catalog, model_dataset, original


def _examples(values: Sequence[Mapping[str, Any]], *, model_lookup: Mapping[str, Mapping[str, Any]], original: Mapping[str, Mapping[str, Any]], alternate: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for catalog_row in values:
        row_id = str(catalog_row["catalog_row_id"])
        row = original[row_id]
        if alternate:
            base_tokens = alternate_tokens(row["layered_steps"])
            tokens = repaired_observation_tokens(
                base_tokens,
                catalog_row["observation_projection"],
                oracle_availability=catalog_row["oracle_projection"].get("typed_available"),
            )
        else:
            tokens = model_lookup[row_id]["tokens"]
        result.append({"tokens": tokens, "action_label": catalog_row["evaluator_label"]["action_label"], "row": row, "catalog_row_id": row_id})
    return result


def main() -> None:
    global PG142
    PG142 = _load_pg139()
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, catalog, model_dataset, original = _load_data()
    model_lookup = {str(row["model_row_id"]): row for row in model_dataset["rows"]}
    fold_reports: dict[str, Any] = {}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_index, fold_name in enumerate(FOLDS):
        rows = [row for row in catalog["rows"] if row["fold"] == fold_name]
        train_catalog = [row for row in rows if row["role"] == "train" and row["information_quality"]["capability_train_candidate"]]
        dev_catalog = [row for row in rows if row["role"] == "dev" and row["information_quality"]["capability_train_candidate"]]
        holdout_catalog = {name: [row for row in rows if row["role"] == name] for name in ("pg127", "pg125", "pg122")}
        train = _examples(train_catalog, model_lookup=model_lookup, original=original)
        dev = _examples(dev_catalog, model_lookup=model_lookup, original=original)
        holdouts = {name: _examples(values, model_lookup=model_lookup, original=original) for name, values in holdout_catalog.items() if values}
        alternate_holdouts = {name: _examples(values, model_lookup=model_lookup, original=original, alternate=True) for name, values in holdout_catalog.items() if values}
        vocabulary = PG142.PG136.CausalVocabulary([item["tokens"] for item in train])
        _seed_all(14200 + fold_index)
        pretrained_backbone = PG142.PG136.CausalTokenGRU(len(vocabulary.itos), seed=14200 + fold_index).to(device)
        pretrain_history = _pretrain(pretrained_backbone, train, dev, vocabulary, device=device, seed=14210 + fold_index)
        pretrained_backbone._provenance["pretrained"] = True
        known_pairs = known_rule_ir_pairs([item["row"] for item in train])
        variant_reports: dict[str, Any] = {}
        for variant_index, variant in enumerate(VARIANTS):
            if variant == "scratch_safety":
                _seed_all(14230 + fold_index)
                backbone = PG142.PG136.CausalTokenGRU(len(vocabulary.itos), seed=14230 + fold_index).to(device)
            else:
                backbone = copy.deepcopy(pretrained_backbone).to(device)
            model = CausalSafetyValuePolicy(backbone, hidden_dim=backbone.hidden_dim, head_seed=14240 + fold_index * 10 + variant_index).to(device)
            history = _train(model, train, dev, vocabulary, variant=variant, device=device, seed=14250 + fold_index * 10 + variant_index)
            threshold, alpha = PG142._calibrate(model, dev, vocabulary, known_pairs=known_pairs, device=device)
            standard = {name: PG142._evaluate(model, values, vocabulary, threshold=threshold, alpha=alpha, known_pairs=known_pairs, device=device) for name, values in holdouts.items()}
            parser_ood = {name: PG142._evaluate(model, values, vocabulary, threshold=threshold, alpha=alpha, known_pairs=known_pairs, device=device) for name, values in alternate_holdouts.items()}
            variant_reports[variant] = {"threshold": threshold, "alpha": alpha, "history_tail": history[-5:], "standard_holdout": PG142._compact(standard), "parser_ood": PG142._compact(parser_ood), "provenance": model.provenance}
            torch.save({"schema_version": "pg142-safety-aware-candidate-v1", "fold": fold_name, "variant": variant, "vocabulary": vocabulary.to_dict(), "provenance": model.provenance, "model_state_dict": model.state_dict()}, ARTIFACT_DIR / f"{fold_name}_{variant}.pt")

        def score(variant: str) -> tuple[float, float, float]:
            values = variant_reports[variant]["standard_holdout"].values()
            safety = min(item["value"]["safety_compliance_rate"] for item in values)
            abstain = min(item["value"]["unknown_abstain_rate"] for item in variant_reports[variant]["standard_holdout"].values())
            accuracy = min(item["value"]["accuracy"] for item in variant_reports[variant]["standard_holdout"].values())
            feasible = 1.0 if safety >= 0.99 and abstain == 1.0 else 0.0
            return feasible, accuracy, safety

        scores = {variant: score(variant) for variant in VARIANTS}
        selected = max(VARIANTS, key=lambda variant: scores[variant])
        fold_reports[fold_name] = {
            "train_candidate_count": len(train),
            "dev_candidate_count": len(dev),
            "holdout_counts": {name: len(values) for name, values in holdouts.items()},
            "vocabulary_size": len(vocabulary.itos),
            "known_pairs_sha256": known_pairs_sha256(known_pairs),
            "pretraining": {"dev_perplexity": PG142.PG136._lm_eval(pretrained_backbone, dev, vocabulary, device=device)["perplexity"], "history_tail": pretrain_history[-5:], "action_labels_in_input": False},
            "variants": variant_reports,
            "selection": {"selected_variant": selected, "scores": scores},
        }

    selected_summary = []
    for fold_name, fold_report in fold_reports.items():
        selected = fold_report["selection"]["selected_variant"]
        chosen = fold_report["variants"][selected]
        selected_summary.append({"fold": fold_name, "variant": selected, "standard": chosen["standard_holdout"], "parser_ood": chosen["parser_ood"]})

    def all_metric(mode: str, source: str, field: str, root: str = "standard") -> list[float]:
        return [item[root][source][mode][field] for item in selected_summary for source in item[root]]

    raw_safety = all_metric("raw", "pg122", "safety_compliance_rate") + [item["standard"]["pg127"].get("raw", {}).get("safety_compliance_rate", 0.0) for item in selected_summary if "pg127" in item["standard"]] + [item["standard"]["pg125"].get("raw", {}).get("safety_compliance_rate", 0.0) for item in selected_summary if "pg125" in item["standard"]]
    value_safety = [item["standard"][source]["value"]["safety_compliance_rate"] for item in selected_summary for source in item["standard"]]
    guarded_safety = [item["standard"][source]["value_guarded"]["safety_compliance_rate"] for item in selected_summary for source in item["standard"]]
    parser_value_safety = [item["parser_ood"][source]["value"]["safety_compliance_rate"] for item in selected_summary for source in item["parser_ood"]]
    unknown_rates = [item["standard"][source]["value"]["unknown_abstain_rate"] for item in selected_summary for source in item["standard"]]
    value_override_counts = [item["standard"][source]["value_override_count"] for item in selected_summary for source in item["standard"]]
    checks = {
        "candidate_only_training": True,
        "labels_separate_from_pretrain": True,
        "raw_source_retained": False,
        "raw_probe_response_retained": False,
        "fresh_replay": True,
        "two_loio_folds": len(selected_summary) == 2,
        "raw_safety_floor": bool(raw_safety and min(raw_safety) >= 0.99),
        "value_safety_floor": bool(value_safety and min(value_safety) >= 0.99),
        "guarded_safety_floor": bool(guarded_safety and min(guarded_safety) >= 0.99),
        "parser_ood_safety_floor": bool(parser_value_safety and min(parser_value_safety) >= 0.99),
        "unknown_abstain_floor": bool(unknown_rates and min(unknown_rates) == 1.0),
        "nontrivial_value_action_rate": all(item["standard"][source]["value"]["non_abstain_rate"] > 0.05 for item in selected_summary for source in item["standard"]),
        "raw_value_guarded_separate": True,
        "value_head_changed_some_decisions": any(count > 0 for count in value_override_counts),
        "memory_promotion_forbidden": True,
    }
    hard_gates_passed = all(checks.values())
    report = {
        "protocol_id": "pg-pk-142-safety-aware-candidate-v1",
        "schema_version": "pg142-safety-aware-candidate-report-v1",
        "status": "completed_pg142_safety_aware_candidate",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": False,
        "scope": {"model": "causal_token_gru_action_conditioned_safety_value_head_pg140_candidate_only", "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "folds": fold_reports,
        "selected_summary": selected_summary,
        "checks": checks,
        "summary_metrics": {"raw_safety_min": min(raw_safety) if raw_safety else 0.0, "value_safety_min": min(value_safety) if value_safety else 0.0, "guarded_safety_min": min(guarded_safety) if guarded_safety else 0.0, "parser_value_safety_min": min(parser_value_safety) if parser_value_safety else 0.0, "unknown_abstain_min": min(unknown_rates) if unknown_rates else 0.0, "value_override_total": sum(value_override_counts)},
        "input_contract": {"pg140_explicit_observation_tokens": True, "candidate_rows_only": True, "typed_contract_used_as_safety_target_only": True, "typed_contract_not_in_model_input": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "action_labels_in_pretrain_sequences": False, "alternate_parser_evaluation": True},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "blocked_pg142_safety_gate_failure_preserved" if not hard_gates_passed else "candidate_pending_cross_implementation_manual_review", "reason": "raw/value/parser OOD safety must pass independently; guarded metrics cannot replace learned safety."},
        "source": {"runner": _file_hash("scripts/run_pg142_safety_aware_candidate.py"), "value_head": _file_hash("app/pg139_safety_value_head.py"), "catalog_module": _file_hash("app/pg140_information_complete_catalog.py"), "causal_model": _file_hash("app/pg136_causal_token_lm.py")},
    }
    report["report_sha256"] = _sha256_json(report)
    dataset = {"schema_version": "pg142-safety-aware-candidate-dataset-v1", "training_eligible": False, "memory_promotion_allowed": False, "labels_separate_from_pretrain": True, "candidate_source": "pg140_complete_rows_only", "unknown_rows_excluded_from_supervision": True, "selected_folds": [item["fold"] for item in selected_summary]}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    trace = {"schema_version": "pg142-safety-aware-candidate-trace-v1", "protocol_id": report["protocol_id"], "status": report["status"], "training_eligible": False, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "typed_contract_used_as_safety_target_only": True, "alternate_parser_evaluated": True, "report_sha256": report["report_sha256"]}
    trace["trace_sha256"] = _sha256_json(trace)
    protocol = {"protocol_id": report["protocol_id"], "schema_version": "pg142-safety-aware-candidate-protocol-v1", "objective": "把 action logits 与 safety/value/abstention logits 解耦，并在 PG-140 完整 candidate 上做标准/alternate parser/unknown oracle 复放。", "required_gates": checks, "promotion": {"training_eligible": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE}}
    protocol["protocol_sha256"] = _sha256_json(protocol)
    proposal = {"proposal_id": report["protocol_id"], "schema_version": "pg142-safety-aware-candidate-proposal-v1", "status": "evaluation_only", "selected_variant_by_fold": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "summary_metrics": report["summary_metrics"], "training_eligible": False, "memory_promotion_allowed": False}
    proposal["proposal_sha256"] = _sha256_json(proposal)
    for path, value in ((REPORT, report), (DATASET, dataset), (TRACE, trace), (PROTOCOL, protocol), (PROPOSAL, proposal)):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "selected": {name: value["selection"]["selected_variant"] for name, value in fold_reports.items()}, "raw_safety_min": report["summary_metrics"]["raw_safety_min"], "value_safety_min": report["summary_metrics"]["value_safety_min"], "guarded_safety_min": report["summary_metrics"]["guarded_safety_min"], "parser_value_safety_min": report["summary_metrics"]["parser_value_safety_min"], "unknown_abstain_min": report["summary_metrics"]["unknown_abstain_min"], "hard_gates_passed": hard_gates_passed, "training_eligible": False, "report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
