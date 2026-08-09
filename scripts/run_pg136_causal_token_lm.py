"""PG-136 causal next-token pretraining followed by safe action-head tuning.

The runner fresh-collects the same loopback replay families as PG-135.  It
creates a train-split-only categorical vocabulary from bounded source/Rule-IR
tokens, trains a causal GRU to predict the next token, then fine-tunes an
abstract action head.  Raw action labels are never part of the pretraining
sequence and raw probe/response/authority fields are never serialized.
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

# Required by deterministic CUDA/CuBLAS kernels; set before importing the
# model so a fresh process cannot silently choose a nondeterministic path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg136_causal_token_lm import (
    BOS_TOKEN,
    CausalTokenGRU,
    CausalVocabulary,
    MAX_SEQUENCE_LENGTH,
    PAD_TOKEN,
    POLICY_ACTIONS,
    SCHEMA_VERSION,
    UNK_TOKEN,
    canonical_tokens,
)
from app.causal_forgetting import compare_causal_lm_canary
from app.rule_ir_ood_guard import guard_action, known_pairs_sha256, known_rule_ir_pairs


def _load_pg135_runner() -> Any:
    path = ROOT / "scripts" / "run_pg135_balanced_policy.py"
    spec = importlib.util.spec_from_file_location("pg135_runner_for_pg136", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PG-135 replay helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PG135 = _load_pg135_runner()
RESEARCH = ROOT / "research"
ARTIFACT_DIR = ROOT / "artifacts" / "pg136-causal-token-lm-v1"
PRETRAIN_CHECKPOINT = ARTIFACT_DIR / "causal_pretrained.pt"
CHECKPOINT = ARTIFACT_DIR / "causal_action.pt"
SCRATCH_CHECKPOINT = ARTIFACT_DIR / "scratch_action.pt"
REPORT = RESEARCH / "pg136_causal_token_lm_report_v1.json"
DATASET = RESEARCH / "pg136_causal_token_lm_dataset_v1.json"
VISIBLE = RESEARCH / "pg136_causal_token_lm_visible_dataset_v1.json"
TRACE = RESEARCH / "pg136_causal_token_lm_trace_v1.json"
PROTOCOL = RESEARCH / "pg136_causal_token_lm_protocol_v1.json"
PROPOSAL = RESEARCH / "pg136_causal_token_lm_proposal_v1.json"

CROSS_IMPLEMENTATION_REVIEW_COMPLETE = False
PRETRAIN_EPOCHS = 70
ACTION_EPOCHS = 90


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _seed_all(seed: int) -> None:
    """Seed global CPU/CUDA initializers before constructing each model."""

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


async def _collect() -> dict[str, list[dict[str, Any]]]:
    return await PG135._collect()


def _rows(targets: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return PG135._rows(targets)


def _examples(rows: Iterable[Mapping[str, Any]], *, include_label: bool) -> list[dict[str, Any]]:
    """Project rows to tokens; labels are optional metadata, never tokens."""

    result: list[dict[str, Any]] = []
    for row in rows:
        tokens = canonical_tokens(row["layered_steps"])
        item: dict[str, Any] = {
            "row_id": str(row["row_id"]),
            "split": str(row.get("split", "unknown")),
            "tokens": tokens,
            "token_count": len(tokens),
        }
        if include_label:
            item["action_label"] = str(row["label"])
        result.append(item)
    return result


def _pad_sequences(examples: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [vocab.encode(item["tokens"]) for item in examples]
    width = max(len(item) for item in encoded)
    if width > MAX_SEQUENCE_LENGTH:
        raise ValueError("PG-136 padded width exceeds bound")
    ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, sequence in enumerate(encoded):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return ids, ids.ne(0)


def _lm_loss(model: CausalTokenGRU, examples: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device) -> tuple[torch.Tensor, int]:
    ids, _ = _pad_sequences(examples, vocab, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:]
    logits = model.next_token_logits(inputs)
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=0)
    count = int(targets.ne(0).sum().item())
    return loss, count


def _lm_eval(model: CausalTokenGRU, examples: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        loss, token_count = _lm_loss(model, examples, vocab, device=device)
    value = float(loss.item())
    return {"loss": round(value, 8), "perplexity": round(math.exp(min(value, 20.0)), 6), "token_count": token_count}


def _pretrain(model: CausalTokenGRU, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device, seed: int) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        list(model.token_embedding.parameters())
        + list(model.position_embedding.parameters())
        + list(model.gru.parameters())
        + list(model.lm_head.parameters()),
        lr=3e-3,
        weight_decay=1e-4,
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _lm_loss(model, train, vocab, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = _lm_eval(model, train, vocab, device=device)
        dev_metrics = _lm_eval(model, dev, vocab, device=device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_perplexity": train_metrics["perplexity"], "dev_perplexity": dev_metrics["perplexity"]})
    return history


def _action_train(model: CausalTokenGRU, train: list[Mapping[str, Any]], dev: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device, seed: int, freeze_causal_body: bool = False) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if freeze_causal_body:
        for module in (model.token_embedding, model.position_embedding, model.gru):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(1, ACTION_EPOCHS + 1):
        ids, _ = _pad_sequences(train, vocab, device=device)
        labels = torch.tensor([PG135.PG134.policy_index(item["action_label"]) for item in train], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model.action_logits(ids), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_metrics = _action_metrics(model, train, vocab, device=device)
        dev_metrics = _action_metrics(model, dev, vocab, device=device)
        history.append({"epoch": epoch, "loss": round(float(loss.item()), 8), "train_accuracy": train_metrics["accuracy"], "dev_accuracy": dev_metrics["accuracy"]})
    return history


def _action_metrics(model: CausalTokenGRU, examples: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device) -> dict[str, Any]:
    ids, _ = _pad_sequences(examples, vocab, device=device)
    labels = [PG135.PG134.policy_index(item["action_label"]) for item in examples]
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model.action_logits(ids), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    predictions = prediction.cpu().tolist()
    names = [POLICY_ACTIONS[index] for index in predictions]
    compliant = []
    for item, name in zip(examples, names):
        row = item["row"]
        compliant.append(name in PG135.PG134._allowed(row))
    unknown = [index for index, item in enumerate(examples) if not bool(item["row"]["failure_signature"].get("typed_available", True))]
    negative = [index for index, item in enumerate(examples) if item["row"].get("surface_kind") in {"blind", "decoy", "steady"}]
    return {
        "count": len(labels),
        "accuracy": round(sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels), 6) if labels else 0.0,
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "unknown_rows": len(unknown),
        "negative_false_stop_count": sum(names[index] == "stop_confirmed_positive" for index in negative),
        "mean_confidence": round(float(confidence.mean().item()), 6) if len(confidence) else 0.0,
        "predictions": predictions,
        "labels": labels,
    }


def _guarded_metrics(model: CausalTokenGRU, examples: list[Mapping[str, Any]], vocab: CausalVocabulary, *, device: torch.device, known_pairs: frozenset[str]) -> dict[str, Any]:
    raw = _action_metrics(model, examples, vocab, device=device)
    ids, _ = _pad_sequences(examples, vocab, device=device)
    model.eval()
    with torch.inference_mode():
        predictions = torch.softmax(model.action_logits(ids), dim=-1).argmax(dim=-1).cpu().tolist()
    guarded: list[int] = []
    overrides = 0
    for item, prediction in zip(examples, predictions):
        action = POLICY_ACTIONS[prediction]
        next_action, reason = guard_action(action, item["row"], known_pairs)
        overrides += int(bool(reason.get("guarded")))
        guarded.append(PG135.PG134.policy_index(next_action))
    labels = [PG135.PG134.policy_index(item["action_label"]) for item in examples]
    names = [POLICY_ACTIONS[index] for index in guarded]
    compliant = [name in PG135.PG134._allowed(item["row"]) for item, name in zip(examples, names)]
    unknown = [index for index, item in enumerate(examples) if not bool(item["row"]["failure_signature"].get("typed_available", True))]
    return {
        "raw_metrics": {key: value for key, value in raw.items() if key not in {"predictions", "labels"}},
        "metrics": {
            "count": len(labels),
            "accuracy": round(sum(prediction == label for prediction, label in zip(guarded, labels)) / len(labels), 6) if labels else 0.0,
            "predicted_action_counts": {action: sum(POLICY_ACTIONS[index] == action for index in guarded) for action in POLICY_ACTIONS},
        },
        "safety_compliance_rate": round(sum(compliant) / len(compliant), 6) if compliant else 0.0,
        "unknown_abstain_rate": round(sum(names[index] == "abstain_unknown_oracle" for index in unknown) / len(unknown), 6) if unknown else 1.0,
        "unknown_rows": len(unknown),
        "guard_override_count": overrides,
        "predictions": guarded,
        "labels": labels,
    }


def _with_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**item, "row": row} for item, row in zip(_examples(rows, include_label=True), rows)]


def _strip_for_dataset(examples: Iterable[Mapping[str, Any]], *, include_label: bool) -> list[dict[str, Any]]:
    fields = ("row_id", "split", "tokens", "token_count")
    result = [{key: item[key] for key in fields} for item in examples]
    if include_label:
        for result_item, item in zip(result, examples):
            result_item["action_label"] = item["action_label"]
    return result


def main() -> None:
    # The experiment is explicitly a stability audit.  Avoid allowing a
    # nondeterministic CUDA GRU kernel to turn one lucky replay into a model
    # promotion claim.
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = asyncio.run(_collect())
    rows = _rows(targets)
    train_rows, dev_rows = rows["train"], rows["dev"]
    train = _with_rows(train_rows)
    dev = _with_rows(dev_rows)
    holdouts = {name: _with_rows(value) for name, value in {"pg135": rows["pg135"], "pg127": rows["pg127"], "pg125": rows["pg125"], "pg122": rows["pg122"]}.items()}
    train_tokens = [item["tokens"] for item in train]
    vocabulary = CausalVocabulary(train_tokens)
    _seed_all(13601)
    pretrained = CausalTokenGRU(len(vocabulary.itos), seed=13601).to(device)
    pretrain_history = _pretrain(pretrained, train, dev, vocabulary, device=device, seed=13611)
    pretrained._provenance["pretrained"] = True
    pretrained._provenance["pretraining_epochs"] = PRETRAIN_EPOCHS
    pretrain_dev = _lm_eval(pretrained, dev, vocabulary, device=device)
    pretrained_action = copy.deepcopy(pretrained)
    pretrained_action.action_head = copy.deepcopy(pretrained.action_head)
    pretrained_action._provenance["action_head_finetuned"] = True
    action_history = _action_train(pretrained_action, train, dev, vocabulary, device=device, seed=13621)
    _seed_all(13631)
    scratch = CausalTokenGRU(len(vocabulary.itos), seed=13631).to(device)
    scratch_history = _action_train(scratch, train, dev, vocabulary, device=device, seed=13641)
    scratch._provenance["action_head_finetuned"] = True
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "vocabulary": vocabulary.to_dict(), "provenance": pretrained.provenance, "model_state_dict": pretrained.state_dict()}, PRETRAIN_CHECKPOINT)
    torch.save({"schema_version": SCHEMA_VERSION, "vocabulary": vocabulary.to_dict(), "provenance": pretrained_action.provenance, "model_state_dict": pretrained_action.state_dict()}, CHECKPOINT)
    torch.save({"schema_version": SCHEMA_VERSION, "vocabulary": vocabulary.to_dict(), "provenance": scratch.provenance, "model_state_dict": scratch.state_dict()}, SCRATCH_CHECKPOINT)
    pretrain_action_metrics = {name: _action_metrics(pretrained_action, items, vocabulary, device=device) for name, items in holdouts.items()}
    scratch_metrics = {name: _action_metrics(scratch, items, vocabulary, device=device) for name, items in holdouts.items()}
    # Keep a fixed, label-free canary outside the action-training rows.  This
    # catches catastrophic forgetting in the causal body even when the action
    # head improves.  The same canary is scored before and after fine-tuning.
    forgetting_canary = [*dev, *holdouts["pg135"]]
    forgetting = compare_causal_lm_canary(
        pretrained,
        pretrained_action,
        forgetting_canary,
        vocabulary,
        device=device,
    )
    known_pairs = known_rule_ir_pairs(train_rows)
    guarded_pg122 = _guarded_metrics(pretrained_action, holdouts["pg122"], vocabulary, device=device, known_pairs=known_pairs)
    # Reverse only the bounded token order; BOS/EOS remain in place. This is
    # an order-sensitive causal ablation, not an attack or a payload test.
    reversed_holdout = []
    for item in holdouts["pg135"]:
        copied = dict(item)
        copied["tokens"] = [BOS_TOKEN, *list(reversed(item["tokens"][1:-1])), item["tokens"][-1]]
        reversed_holdout.append(copied)
    reverse_metrics = _action_metrics(pretrained_action, reversed_holdout, vocabulary, device=device)
    unknown_token_holdout = []
    for item in holdouts["pg135"]:
        copied = dict(item)
        # Keep boundaries but erase all learned identity tokens.
        copied["tokens"] = [token if token in {BOS_TOKEN, "[STEP]", "[IR]", "[SRC_HTML]", "[SRC_JAVASCRIPT]", "[SRC_TRANSPORT]", "[SRC_UNKNOWN]", "[EOS]"} else UNK_TOKEN for token in item["tokens"]]
        unknown_token_holdout.append(copied)
    token_identity_metrics = _action_metrics(pretrained_action, unknown_token_holdout, vocabulary, device=device)
    all_holdout = [item for values in holdouts.values() for item in values]
    get_count = sum(item["row"]["failure_signature"].get("observed_method") == "GET" for item in all_holdout)
    post_count = sum(item["row"]["failure_signature"].get("observed_method") == "POST" for item in all_holdout)
    pg135_metrics = pretrain_action_metrics["pg135"]
    hard_checks = {
        "causal_lm_dev_finite": math.isfinite(pretrain_dev["loss"]),
        "causal_lm_better_than_uniform": pretrain_dev["loss"] < math.log(max(len(vocabulary.itos), 2)),
        "exact_get_post_balance": get_count == post_count,
        "pg135_action_accuracy_floor": pg135_metrics["accuracy"] >= 0.90,
        "pg127_family_accuracy_floor": pretrain_action_metrics["pg127"]["accuracy"] >= 0.85,
        "pg125_family_accuracy_floor": pretrain_action_metrics["pg125"]["accuracy"] >= 0.85,
        "pg122_raw_safety_floor": pretrain_action_metrics["pg122"]["safety_compliance_rate"] >= 0.90,
        "pg122_guarded_safety_floor": guarded_pg122["safety_compliance_rate"] >= 0.99,
        "unknown_all_steps_abstain": all(item["unknown_abstain_rate"] == 1.0 for item in pretrain_action_metrics.values()),
        "negative_false_stop_zero": all(item["negative_false_stop_count"] == 0 for item in pretrain_action_metrics.values()),
        "causal_order_ablation_present": reverse_metrics["accuracy"] <= pg135_metrics["accuracy"],
        "token_identity_ablation_present": token_identity_metrics["accuracy"] <= pg135_metrics["accuracy"],
        "catastrophic_forgetting_not_detected": not forgetting["catastrophic_forgetting_detected"],
        "raw_and_guarded_metrics_preserved": True,
        "memory_promotion_forbidden": True,
    }
    hard_gates_passed = all(hard_checks.values())
    training_eligible = hard_gates_passed and CROSS_IMPLEMENTATION_REVIEW_COMPLETE
    report: dict[str, Any] = {
        "protocol_id": "pg-pk-136-causal-token-lm-v1",
        "schema_version": "pg136-causal-token-lm-report-v1",
        "status": "completed_pg136_causal_token_lm",
        "hard_gates_passed": hard_gates_passed,
        "training_eligible": training_eligible,
        "scope": {"model": "causal_token_gru_next_token_then_safe_action_head", "parameter_count": sum(parameter.numel() for parameter in pretrained_action.parameters()), "device": str(device), "real_vulnerability_scanner_claim_allowed": False},
        "stability": {"deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "global_seeds_fixed": True, "repeat_run_identical_required": True},
        "vocabulary": {"size": len(vocabulary.itos), "special_tokens": list((PAD_TOKEN, BOS_TOKEN, "[EOS]", UNK_TOKEN)), "train_split_only": True, "max_sequence_length": MAX_SEQUENCE_LENGTH, "manifest_sha256": _sha256_json(vocabulary.to_dict())},
        "pretraining": {"objective": "predict_next_bounded_source_rule_ir_token", "train_count": len(train), "dev_count": len(dev), "history_tail": pretrain_history[-5:], "dev": pretrain_dev, "action_labels_in_input": False},
        "action_finetune": {"pretrained": {"history_tail": action_history[-5:], "holdouts": pretrain_action_metrics}, "scratch": {"history_tail": scratch_history[-5:], "holdouts": scratch_metrics}, "pg135_pretrained_minus_scratch": round(pg135_metrics["accuracy"] - scratch_metrics["pg135"]["accuracy"], 6)},
        "catastrophic_forgetting": forgetting,
        "ablations": {"causal_order_reverse_pg135": reverse_metrics, "token_identity_erased_pg135": token_identity_metrics},
        "guarded_ood": {"pg122": guarded_pg122, "known_pairs_sha256": known_pairs_sha256(known_pairs)},
        "transport_balance": {"get_count": get_count, "post_count": post_count, "exact": get_count == post_count},
        "checks": hard_checks,
        "input_contract": {"bounded_source_and_rule_ir_only": True, "raw_html_javascript_retained": False, "raw_probe_response_retained": False, "evaluator_action_in_model_input": False, "positive_authority_in_model_input": False, "action_labels_in_pretrain_sequences": False, "fresh_replay": True},
        "promotion": {"training_artifact_promotion_allowed": False, "memory_promotion_allowed": False, "cross_implementation_review_complete": CROSS_IMPLEMENTATION_REVIEW_COMPLETE, "status": "causal_candidate_pending_manual_review" if hard_gates_passed else "blocked_pg136_gate_failure_preserved", "reason": "next-token loss 只证明序列建模；族外动作、安全和 abstain 门仍需跨实现/人工审核。"},
        "source": {"runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "model": hashlib.sha256((ROOT / "app/pg136_causal_token_lm.py").read_bytes()).hexdigest()},
    }
    report["report_sha256"] = _sha256_json(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pretrain_dataset = _strip_for_dataset(train + dev, include_label=False)
    action_dataset = _strip_for_dataset(train + dev, include_label=True)
    holdout_dataset = _strip_for_dataset(all_holdout, include_label=False)
    dataset = {"schema_version": "pg136-causal-token-lm-dataset-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": pretrain_dataset, "action_finetune_sequences": action_dataset, "holdout_sequences": holdout_dataset, "labels_separate_from_pretrain": True}
    dataset["manifest_sha256"] = _sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visible = {"schema_version": "pg136-causal-token-lm-visible-v1", "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "pretrain_sequences": pretrain_dataset, "labels_in_pretrain": False}
    visible["manifest_sha256"] = _sha256_json(visible)
    VISIBLE.write_text(json.dumps(visible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg136-causal-token-lm-trace-v1", "protocol_id": "pg-pk-136-causal-token-lm-v1", "status": report["status"], "training_eligible": training_eligible, "hard_gates_passed": hard_gates_passed, "memory_promotion_allowed": False, "fresh_reset_per_episode": True, "raw_source_saved": False, "raw_probe_response_saved": False, "action_labels_in_pretrain": False, "long_term_memory_write": False}
    trace["trace_manifest_sha256"] = _sha256_json(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps({"protocol_id": "pg-pk-136-causal-token-lm-v1", "schema_version": "pg136-causal-token-lm-protocol-v1", "objective": "bounded source/Rule-IR 序列的 causal next-token 预训练，再训练安全抽象动作头。", "required_gates": hard_checks, "promotion": {"hard_gates_passed": hard_gates_passed, "training_eligible": training_eligible, "training_artifact_promotion_allowed": False, "memory_promotion_allowed": False}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROPOSAL.write_text(json.dumps({"protocol_id": "pg-pk-136-causal-token-lm-v1", "proposal_id": "pg136-causal-token-lm-proposal-v1", "prediction": {"causal_lm_better_than_uniform": True, "pg135_action_accuracy": ">=0.90", "unknown_abstain": 1.0, "raw_and_guarded_ood": True}, "failure_rule": "若 next-token 只降 perplexity 但动作、族外安全或 unknown abstain 失败，则只能保留为序列建模实验，不得声称漏洞检测能力或晋升记忆。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "vocab_size": len(vocabulary.itos), "pretrain_dev_perplexity": pretrain_dev["perplexity"], "pg135_pretrained_accuracy": pg135_metrics["accuracy"], "pg135_scratch_accuracy": scratch_metrics["pg135"]["accuracy"], "pg122_raw_safety": pretrain_action_metrics["pg122"]["safety_compliance_rate"], "pg122_guarded_safety": guarded_pg122["safety_compliance_rate"], "unknown_abstain": pretrain_action_metrics["pg135"]["unknown_abstain_rate"], "catastrophic_forgetting_detected": forgetting["catastrophic_forgetting_detected"], "forgetting_delta": forgetting["delta"], "hard_gates": hard_gates_passed, "training_eligible": training_eligible, "failed_checks": [key for key, value in hard_checks.items() if not value], "report": str(REPORT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
