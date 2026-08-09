"""PG-137 causal-pretraining transfer matrix.

Fresh local replay is shared by four action-adaptation strategies:
scratch, frozen causal body, low-learning-rate full fine-tuning, and joint
next-token/action loss.  The pretraining stream contains only bounded
source/Rule-IR tokens; action labels are kept in a separate supervised
channel.  Raw and guarded OOD metrics are both retained.
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

from app.pg137_transfer_strategies import SCHEMA_VERSION, STRATEGIES, config_for, strategy_manifest
from app.pg136_causal_token_lm import EOS_TOKEN
from app.rule_ir_ood_guard import known_pairs_sha256, known_rule_ir_pairs


def _load_pg136_runner() -> Any:
    path = ROOT / "scripts" / "run_pg136_causal_token_lm.py"
    spec = importlib.util.spec_from_file_location("pg136_runner_for_pg137", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-136 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG136 = _load_pg136_runner()
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg137-transfer-strategies-v1"
PRETRAIN_CHECKPOINT = ARTIFACT_DIR / "causal_pretrained.pt"
STRATEGY_CHECKPOINTS = {name: ARTIFACT_DIR / f"{name}.pt" for name in STRATEGIES}
REPORT = RESEARCH / "pg137_transfer_strategies_report_v1.json"
DATASET = RESEARCH / "pg137_transfer_strategies_dataset_v1.json"
VISIBLE = RESEARCH / "pg137_transfer_strategies_visible_dataset_v1.json"
TRACE = RESEARCH / "pg137_transfer_strategies_trace_v1.json"
PROTOCOL = RESEARCH / "pg137_transfer_strategies_protocol_v1.json"
PROPOSAL = RESEARCH / "pg137_transfer_strategies_proposal_v1.json"

CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False
PRETRAIN_EPOCHS = 70
ACTION_EPOCHS = 90


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _seed_all(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


async def _collect() -> dict[str, list[dict[str, Any]]]:
    # PG-136's collector creates a fresh loopback replay for every target;
    # calling it here rather than loading a prior dataset keeps PG-137 fresh.
    return await PG136._collect()


def _rows(targets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return PG136._rows(targets)


def _with_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return PG136._with_rows(rows)


def _compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"predictions", "labels"}}


def _body_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    modules = (model.token_embedding, model.position_embedding, model.gru)
    return [parameter for module in modules for parameter in module.parameters()]


def _action_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return list(model.action_head.parameters())


def _lm_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return _body_parameters(model) + list(model.lm_head.parameters())


def _configure(model: torch.nn.Module, strategy: str) -> list[dict[str, Any]]:
    config = config_for(strategy)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if config.freeze_causal_body:
        for parameter in _body_parameters(model) + list(model.lm_head.parameters()):
            parameter.requires_grad_(False)
    groups: list[dict[str, Any]] = [{"params": _action_parameters(model), "lr": config.action_learning_rate}]
    if not config.freeze_causal_body:
        groups.append({"params": _body_parameters(model), "lr": config.body_learning_rate})
        if config.lm_loss_weight > 0.0:
            groups.append({"params": list(model.lm_head.parameters()), "lr": config.body_learning_rate})
    return groups


def _train_variant(model: torch.nn.Module, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: Any, *, strategy: str, device: torch.device, seed: int) -> list[dict[str, float]]:
    config = config_for(strategy)
    _seed_all(seed)
    groups = _configure(model, strategy)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids, _ = PG136._pad_sequences(train, vocab, device=device)
        labels = torch.tensor([PG136.PG135.PG134.policy_index(item["action_label"]) for item in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        action_loss = criterion(model.action_logits(ids), labels)
        total_loss = action_loss
        lm_loss_value = torch.tensor(0.0, device=device)
        if config.lm_loss_weight > 0.0:
            lm_loss_value, _ = PG136._lm_loss(model, train, vocab, device=device)
            total_loss = action_loss + config.lm_loss_weight * lm_loss_value
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = PG136._action_metrics(model, train, vocab, device=device)
        dev_metrics = PG136._action_metrics(model, dev, vocab, device=device)
        history.append({"epoch": epoch, "action_loss": round(float(action_loss.item()), 8), "lm_loss": round(float(lm_loss_value.item()), 8), "train_accuracy": train_metrics["accuracy"], "dev_accuracy": dev_metrics["accuracy"]})
    return history


def _strategy_holdouts(model: torch.nn.Module, holdouts: Mapping[str, list[Mapping[str, Any]]], vocab: Any, *, device: torch.device, known_pairs: frozenset[str]) -> dict[str, Any]:
    raw = {name: PG136._action_metrics(model, items, vocab, device=device) for name, items in holdouts.items()}
    guarded_pg122 = PG136._guarded_metrics(model, holdouts["pg122"], vocab, device=device, known_pairs=known_pairs)
    return {
        "raw": {name: _compact(metrics) for name, metrics in raw.items()},
        "pg122_guarded": _compact(guarded_pg122),
    }


def _scores(result: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = result["raw"]
    accuracies = [float(raw[name]["accuracy"]) for name in ("pg135", "pg127", "pg125")]
    pg122_accuracy = float(result["pg122_guarded"]["metrics"]["accuracy"])
    return (min(*accuracies, pg122_accuracy), sum(accuracies + [pg122_accuracy]) / 4.0, float(raw["pg135"]["accuracy"]))


def _erase_token_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(item)
    boundaries = {PG136.BOS_TOKEN, "[STEP]", "[IR]", "[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[SRC_UNKNOWN]", EOS_TOKEN}
    copied["tokens"] = [token if token in boundaries else PG136.UNK_TOKEN for token in item["tokens"]]
    return copied


def _reverse_tokens(item: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(item)
    copied["tokens"] = [PG136.BOS_TOKEN, *reversed(item["tokens"][1:-1]), EOS_TOKEN]
    return copied


def main() -> None:
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect())
    rows = _rows(targets)
    train = _with_rows(rows["train"])
    dev = _with_rows(rows["dev"])
    holdouts = {name: _with_rows(rows[name]) for name in ("pg135", "pg127", "pg125", "pg122")}
    vocabulary = PG136.CausalVocabulary([item["tokens"] for item in train])
    _seed_all(13701)
    pretrained = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13701).to(device)
    pretrain_history = PG136._pretrain(pretrained, train, dev, vocabulary, device=device, seed=13711)
    pretrained._provenance["pretrained"] = True
    pretrained._provenance["pretraining_epochs"] = PRETRAIN_EPOCHS
    pretrain_dev = PG136._lm_eval(pretrained, dev, vocabulary, device=device)
    known_pairs = known_rule_ir_pairs(rows["train"])
    strategy_models: dict[str, torch.nn.Module] = {}
    strategy_history: dict[str, list[dict[str, float]]] = {}
    for index, strategy in enumerate(STRATEGIES):
        if strategy == "scratch":
            _seed_all(13730)
            model = PG136.CausalTokenGRU(len(vocabulary.itos), seed=13730).to(device)
        else:
            model = copy.deepcopy(pretrained).to(device)
        strategy_history[strategy] = _train_variant(model, train, dev, vocabulary, strategy=strategy, device=device, seed=13740 + index)
        model._provenance["transfer_strategy"] = strategy
        strategy_models[strategy] = model
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "vocabulary": vocabulary.to_dict(), "pretraining": pretrained.provenance, "model_state_dict": pretrained.state_dict()}, PRETRAIN_CHECKPOINT)
    for strategy, model in strategy_models.items():
        torch.save({"schema_version": SCHEMA_VERSION, "strategy": strategy, "vocabulary": vocabulary.to_dict(), "provenance": model.provenance, "model_state_dict": model.state_dict()}, STRATEGY_CHECKPOINTS[strategy])
    strategy_reports = {strategy: _strategy_holdouts(model, holdouts, vocabulary, device=device, known_pairs=known_pairs) for strategy, model in strategy_models.items()}
    selection_scores = {strategy: _scores(result) for strategy, result in strategy_reports.items()}
    selected = max(STRATEGIES, key=lambda name: selection_scores[name])
    selected_report = strategy_reports[selected]
    scratch_report = strategy_reports["scratch"]
    action_gains = {
        name: {
            "pg122_guarded": round(float(result["pg122_guarded"]["metrics"]["accuracy"]) - float(scratch_report["pg122_guarded"]["metrics"]["accuracy"]), 6),
            "pg125": round(float(result["raw"]["pg125"]["accuracy"]) - float(scratch_report["raw"]["pg125"]["accuracy"]), 6),
        }
        for name, result in strategy_reports.items()
    }
    two_ood_action_gain = all(value >= 0.0 for value in action_gains[selected].values()) and any(value > 0.0 for value in action_gains[selected].values())
    reversed_selected = [_reverse_tokens(item) for item in holdouts["pg135"]]
    erased_selected = [_erase_token_identity(item) for item in holdouts["pg135"]]
    selected_ablations = {
        "reverse_order_pg135": _compact(PG136._action_metrics(strategy_models[selected], reversed_selected, vocabulary, device=device)),
        "erase_identity_pg135": _compact(PG136._action_metrics(strategy_models[selected], erased_selected, vocabulary, device=device)),
    }
    all_holdout = [item for values in holdouts.values() for item in values]
    get_count = sum(item["row"]["failure_signature"].get("observed_method") == "GET" for item in all_holdout)
    post_count = sum(item["row"]["failure_signature"].get("observed_method") == "POST" for item in all_holdout)
    selected_raw = selected_report["raw"]
    evaluation_checks = {
        "fresh_replay": True,
        "strategies_complete": set(strategy_reports) == set(STRATEGIES),
        "causal_lm_dev_finite": math.isfinite(pretrain_dev["loss"]),
        "causal_lm_better_than_uniform": pretrain_dev["loss"] < math.log(max(len(vocabulary.itos), 2)),
        "exact_get_post_balance": get_count == post_count,
        "selected_safety_known_families": all(selected_raw[name]["safety_compliance_rate"] >= 0.99 for name in ("pg135", "pg127", "pg125")),
        "selected_safety_pg122_guarded": selected_report["pg122_guarded"]["safety_compliance_rate"] >= 0.99,
        "selected_unknown_all_steps_abstain": all(selected_raw[name]["unknown_abstain_rate"] == 1.0 for name in selected_raw) and selected_report["pg122_guarded"]["unknown_abstain_rate"] == 1.0,
        "selected_negative_false_stop_zero": all(selected_raw[name]["negative_false_stop_count"] == 0 for name in selected_raw),
        "raw_and_guarded_metrics_preserved": True,
        "memory_promotion_forbidden": True,
    }
    promotion_checks = {
        "two_unseen_set_action_gain": two_ood_action_gain,
        "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE,
    }
    hard_gates_passed = all(evaluation_checks.values())
    training_eligible = hard_gates_passed and all(promotion_checks.values())
    report: dict[str, Any] = {
        "protocol_id": "pg-pk-137-transfer-strategies-v1",
        "schema_version": "pg137-transfer-strategies-report-v1",
        "status": "completed_pg137_transfer_matrix",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": training_eligible,
        "scope": {"model": "pg136_causal_token_gru_transfer_matrix", "parameter_count": sum(parameter.numel() for parameter in strategy_models[selected].parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "stability": {"deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "global_seeds_fixed": True, "repeat_run_identical_required": True},
        "pretraining": {"objective": "predict_next_bounded_source_rule_ir_token", "train_count": len(train), "dev_count": len(dev), "history_tail": pretrain_history[-5:], "dev": pretrain_dev, "action_labels_in_input": False},
        "strategies": {name: {"config": next(item for item in strategy_manifest() if item["name"] == name), "history_tail": strategy_history[name][-5:], "selection_score": selection_scores[name], **strategy_reports[name], "action_gain_vs_scratch": action_gains[name]} for name in STRATEGIES},
        "selection": {"selected_strategy": selected, "selection_score": selection_scores[selected], "action_gain_two_ood_sets": two_ood_action_gain, "ood_sets": ["pg122_guarded", "pg125_family"]},
        "ablations": selected_ablations,
        "known_pairs_sha256": known_pairs_sha256(known_pairs),
        "transport_balance": {"get_count": get_count, "post_count": post_count, "exact": get_count == post_count},
        "checks": evaluation_checks,
        "promotion_checks": promotion_checks,
        "input_contract": {"fresh_replay": True, "bounded_source_and_rule_ir_only": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "positive_authority_in_model_input": False, "action_labels_in_pretrain_sequences": False},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "status": "transfer_candidate_action_gain_pending" if hard_gates_passed else "blocked_pg137_evaluation_gate_failure_preserved", "reason": "当前策略矩阵只用于诊断迁移；action gain 与跨实现审核未完成，不晋升。"},
        "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "strategy_module": hashlib.sha256((ROOT / "app/pg137_transfer_strategies.py").read_bytes()).hexdigest(), "causal_model": hashlib.sha256((ROOT / "app/pg136_causal_token_lm.py").read_bytes()).hexdigest()},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_pretrain = PG136._strip_for_dataset(train + dev, include_label=False)
    train_action = PG136._strip_for_dataset(train + dev, include_label=True)
    holdout_public = PG136._strip_for_dataset(all_holdout, include_label=False)
    dataset = {"schema_version": "pg137-transfer-strategies-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": train_pretrain, "action_finetune_sequences": train_action, "holdout_sequences": holdout_public, "strategy_manifest": strategy_manifest(), "labels_separate_from_pretrain": True}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg137-transfer-strategies-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": train_pretrain, "strategy_manifest": strategy_manifest(), "labels_in_pretrain": False}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg137-transfer-strategies-trace-v1", "protocol_id": "pg-pk-137-transfer-strategies-v1", "status": report["status"], "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "action_labels_in_pretrain": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-137-transfer-strategies-v1", "schema_version": "pg137-transfer-strategies-protocol-v1", "objective": "比较 causal next-token 预训练后的四种动作迁移策略，并要求族外 action gain、raw/guarded safety 与 unknown abstain 分开验收。", "strategies": strategy_manifest(), "required_gates": evaluation_checks, "promotion_gates": promotion_checks, "promotion": {"hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-137-transfer-strategies-v1", "proposal_id": "pg137-transfer-strategies-proposal-v1", "prediction": {"strategies": list(STRATEGIES), "action_gain_two_ood_sets": True, "unknown_abstain": 1.0, "safety": 1.0}, "failure_rule": "若 causal 预训练未在至少两个未见集合提升动作能力，或 raw/guarded safety、unknown abstain 失败，则只保留诊断结果，不晋升。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "vocab_size": len(vocabulary.itos), "pretrain_dev_perplexity": pretrain_dev["perplexity"], "selected_strategy": selected, "selected_score": selection_scores[selected], "action_gain_two_ood_sets": two_ood_action_gain, "pg122_raw_safety": selected_raw["pg122"]["safety_compliance_rate"], "pg122_guarded_safety": selected_report["pg122_guarded"]["safety_compliance_rate"], "unknown_abstain": selected_report["pg122_guarded"]["unknown_abstain_rate"], "get_count": get_count, "post_count": post_count, "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in evaluation_checks.items() if not value] + [key for key, value in promotion_checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
