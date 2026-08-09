"""PG-270: teacher-led SFT plus preference/process-reward ablation.

This experiment consumes only PG-269's abstract context/target split.  The
model never receives raw probes, response bodies, oracle fields, or route
family names as input.  A tiny conditional GRU is used deliberately: PG-270
tests whether the training signal is shaped correctly, not whether a toy
checkpoint is a web vulnerability detector.

Two candidates are trained on the same grouped split:

* ``plain_sft`` uses teacher-forced cross entropy only.
* ``guided_sft`` adds teacher process weights and a pairwise preference loss
  against deliberately corrupted abstract trajectories.

Both candidates are evaluated with the same greedy decoder and held-out
family/route split.  Promotion and memory writes remain blocked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PG269_DATASET = ROOT / "research" / "pg269_failure_guided_replay_dataset_v1.json"
PG269_AUDIT = ROOT / "research" / "pg269_failure_guided_replay_audit_v1.json"
DATASET_PATH = ROOT / "research" / "pg270_teacher_sft_dataset_v1.json"
REPORT_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg270_teacher_sft_ablation_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg270-teacher-sft"
CHECKPOINT_PATH = OUTPUT_DIR / "teacher_sft_ablation.pt"

SEED = 27001
HIDDEN_DIM = 96
EMBED_DIM = 64
EPOCHS = 260
LEARNING_RATE = 0.008
PAIRWISE_BETA = 0.7
PAD = "[PAD]"
UNK = "[UNK]"
TARGET_BOS = "[TARGET_BOS]"
TARGET_EOS = "[TARGET_EOS]"

CONTEXT_FORBIDDEN = (
    "oracle",
    "payload",
    "response_body",
    "echo_excerpt",
    "confirmed_positive",
    "outcome_class",
    "body_sha256",
)
UNSEEN_FAMILIES = {"redirect", "xxe", "serialization", "infoleak", "other"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _token_value(tokens: Iterable[str], prefix: str) -> str | None:
    values = [token.split("=", 1)[1] for token in tokens if token.startswith(prefix)]
    return values[-1] if values else None


def _contains(tokens: Iterable[str], value: str) -> bool:
    return any(token == value for token in tokens)


def _family(row: dict[str, Any]) -> str:
    return str(row.get("labels", {}).get("family_class") or "unknown")


def _split(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Group by family first, then reserve deterministic route holdouts."""

    assignments: dict[str, str] = {}
    for row in rows:
        record_id = str(row["record_id"])
        family = _family(row)
        if family in UNSEEN_FAMILIES:
            assignments[record_id] = "family_holdout"
            continue
        bucket = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16) % 6
        assignments[record_id] = "route_dev" if bucket == 0 else "train"
    # Keep the small experiment useful even if a future catalog changes the
    # hash distribution: at least one ordinary route must be held out.
    if not any(value == "route_dev" for value in assignments.values()):
        first = next(iter(assignments))
        assignments[first] = "route_dev"
    return assignments


def _repair_corruption(target: list[str], labels: dict[str, Any], index: int) -> tuple[list[str], str]:
    """Create a rejected abstract trajectory without introducing raw values."""

    chosen = list(target)
    if labels.get("final_belief") == "oracle_gap":
        # The most dangerous failure mode is declaring success without a typed
        # effect.  Change only the final decision slots.
        for position in range(len(chosen) - 1, -1, -1):
            if chosen[position] == "next_action=abstain":
                chosen[position] = "next_action=confirm_without_evidence"
                break
        for position in range(len(chosen) - 1, -1, -1):
            if chosen[position] == "final_belief=oracle_gap":
                chosen[position] = "final_belief=unsupported_positive"
                break
        return chosen, "premature_positive_without_typed_effect"

    if labels.get("repair_succeeded"):
        # A successful repair must still be preceded by diagnosis.  Remove the
        # repair action from the abstract trajectory and force an early claim.
        for position, token in enumerate(chosen):
            if token == "action=repair_probe":
                chosen[position] = "action=confirm_without_replay"
                break
        return chosen, "skipped_repair_replay"

    # For abstaining examples, skipping the negative control is a realistic
    # process error even though the final class happens to be abstain.
    for position, token in enumerate(chosen):
        if token == "action=negative_control":
            chosen[position] = "action=candidate_probe"
            break
    return chosen, "skipped_negative_control"


def _teacher_score(row: dict[str, Any]) -> dict[str, float]:
    context = list(row["context_tokens"])
    target = list(row["target_tokens"])
    labels = dict(row["labels"])
    final_belief = str(labels.get("final_belief") or "")
    has_required_context = all(
        _contains(context, token)
        for token in ("fresh_reset=1", "source_attested=1", "reference_sent=1", "negative_sent=1", "candidate_sent=1")
    )
    has_candidate = _contains(target, "action=candidate_probe")
    has_diagnosis = _contains(target, "phase=diagnose") or final_belief == "confirmed_effect"
    safe_abstain = final_belief == "oracle_gap" and _token_value(target, "next_action=") == "abstain"
    repair_attempted = bool(labels.get("repair_attempted"))
    repair_succeeded = bool(labels.get("repair_succeeded"))
    return {
        "scope_and_safety": 1.0,
        "information_completeness": 1.0 if has_required_context else 0.0,
        "probe_utility": 1.0 if has_candidate else 0.0,
        "failure_diagnosis": 1.0 if has_diagnosis else 0.5,
        "repair_quality": 1.0 if repair_succeeded else (0.85 if not repair_attempted else 0.7),
        "oracle_and_evidence_alignment": 1.0 if final_belief in {"oracle_gap", "confirmed_effect"} else 0.0,
        "calibrated_abstain": 1.0 if safe_abstain or final_belief == "confirmed_effect" else 0.0,
    }


def _weighted_score(components: dict[str, float]) -> float:
    weights = {
        "scope_and_safety": 0.15,
        "information_completeness": 0.15,
        "probe_utility": 0.15,
        "failure_diagnosis": 0.15,
        "repair_quality": 0.15,
        "oracle_and_evidence_alignment": 0.20,
        "calibrated_abstain": 0.05,
    }
    return round(sum(weights[key] * float(components[key]) for key in weights), 6)


def _load_source() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = json.loads(PG269_DATASET.read_text(encoding="utf-8"))
    audit = json.loads(PG269_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not all(bool(value) for value in audit.get("audit_checks", {}).values()):
        raise RuntimeError("PG-270 refuses a PG-269 source that did not pass independent audit")
    rows = [dict(row) for row in dataset.get("records", []) if bool(row.get("training_eligible"))]
    if len(rows) != 40:
        raise RuntimeError(f"PG-270 expects 40 eligible PG-269 rows, got {len(rows)}")
    for row in rows:
        context = [str(token) for token in row.get("context_tokens", [])]
        if any(any(term in token.casefold() for term in CONTEXT_FORBIDDEN) for token in context):
            raise RuntimeError(f"forbidden evaluator/raw token in context: {row.get('record_id')}")
        if context[-1:] != ["[CTX_END]"] or row.get("target_tokens", [None])[0] != TARGET_BOS:
            raise RuntimeError(f"malformed context/target split: {row.get('record_id')}")
    return rows, audit


def _build_dataset(rows: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    splits = _split(rows)
    sft: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    process: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record_id = str(row["record_id"])
        labels = dict(row["labels"])
        components = _teacher_score(row)
        chosen_score = _weighted_score(components)
        sft.append({
            "record_id": record_id,
            "split": splits[record_id],
            "context_tokens": list(row["context_tokens"]),
            "target_tokens": list(row["target_tokens"]),
            "labels": {key: labels.get(key) for key in ("family_class", "rule_ir_class", "final_belief", "next_action", "repair_attempted", "repair_succeeded", "step_count")},
            "teacher_components": components,
            "teacher_score": chosen_score,
        })
        rejected, reason = _repair_corruption(list(row["target_tokens"]), labels, index)
        rejected_components = dict(components)
        rejected_components["failure_diagnosis"] = min(rejected_components["failure_diagnosis"], 0.25)
        rejected_components["oracle_and_evidence_alignment"] = 0.0
        rejected_components["calibrated_abstain"] = 0.0
        rejected_score = _weighted_score(rejected_components)
        preferences.append({
            "pair_id": f"pg270-pair-{index:03d}",
            "record_id": record_id,
            "split": splits[record_id],
            "chosen_target_tokens": list(row["target_tokens"]),
            "rejected_target_tokens": rejected,
            "rejection_reason": reason,
            "chosen_teacher_score": chosen_score,
            "rejected_teacher_score": rejected_score,
            "teacher_preference": "chosen",
        })
        target = list(row["target_tokens"])
        phases = [token.split("=", 1)[1] for token in target if token.startswith("phase=")]
        step_scores = [
            {"step": step, "score": round(chosen_score, 6), "components": components}
            for step in phases
        ]
        process.append({
            "record_id": record_id,
            "split": splits[record_id],
            "step_scores": step_scores,
            "episode_score": chosen_score,
            "reward_target": {
                "repair_success": bool(labels.get("repair_succeeded")),
                "safe_abstain": labels.get("final_belief") == "oracle_gap" and labels.get("next_action") == "abstain_or_repair",
                "false_positive": False,
            },
        })
    payload = {
        "schema_version": "pg270-teacher-sft-preference-process-reward-dataset-v1",
        "source": {
            "dataset": str(PG269_DATASET.relative_to(ROOT)),
            "dataset_sha256": _sha(json.loads(PG269_DATASET.read_text(encoding="utf-8"))),
            "audit": str(PG269_AUDIT.relative_to(ROOT)),
            "audit_sha256": audit.get("audit_sha256"),
            "loopback_only": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "split_contract": {
            "family_holdout": sorted(UNSEEN_FAMILIES),
            "route_dev_hash_modulus": 6,
            "oracle_target_in_context": False,
            "family_in_context": False,
            "raw_probe_or_response_in_context": False,
        },
        "records": sft,
        "preferences": preferences,
        "process_rewards": process,
        "counts": {
            "sft": len(sft),
            "preference_pairs": len(preferences),
            "process_reward_episodes": len(process),
            "train": sum(value == "train" for value in splits.values()),
            "route_dev": sum(value == "route_dev" for value in splits.values()),
            "family_holdout": sum(value == "family_holdout" for value in splits.values()),
        },
        "contract": {
            "teacher_targets_only": True,
            "reference_answer_not_in_inference_context": True,
            "process_reward_is_label": True,
            "preference_rejected_is_abstract_only": True,
            "training_promotion_blocked": True,
            "memory_promotion_blocked": True,
        },
    }
    payload["dataset_sha256"] = _sha(payload)
    return payload


class TinyConditionalDecoder(nn.Module):
    def __init__(self, vocab_size: int, *, embed_dim: int = EMBED_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def _encode(self, context: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(context), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return hidden

    def forward(self, context: torch.Tensor, lengths: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        hidden = self._encode(context, lengths)
        decoded, _ = self.decoder(self.embedding(target_input), hidden)
        return self.output(decoded)


def _vocabulary(dataset: dict[str, Any]) -> tuple[dict[str, int], dict[int, str]]:
    tokens = {PAD, UNK}
    for row in dataset["records"]:
        tokens.update(row["context_tokens"])
        tokens.update(row["target_tokens"])
    for pair in dataset["preferences"]:
        tokens.update(pair["rejected_target_tokens"])
    ordered = [PAD, UNK] + sorted(tokens - {PAD, UNK})
    return {token: index for index, token in enumerate(ordered)}, {index: token for index, token in enumerate(ordered)}


def _tensorise(rows: list[dict[str, Any]], vocabulary: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = vocabulary[PAD]
    contexts = [[vocabulary.get(token, vocabulary[UNK]) for token in row["context_tokens"]] for row in rows]
    targets = [[vocabulary.get(token, vocabulary[UNK]) for token in row["target_tokens"]] for row in rows]
    context_len = torch.tensor([len(tokens) for tokens in contexts], dtype=torch.long)
    max_context = max(len(tokens) for tokens in contexts)
    max_target = max(len(tokens) for tokens in targets)
    context_tensor = torch.full((len(rows), max_context), pad_id, dtype=torch.long)
    target_input = torch.full((len(rows), max_target - 1), pad_id, dtype=torch.long)
    target_output = torch.full((len(rows), max_target - 1), pad_id, dtype=torch.long)
    for index, (context, target) in enumerate(zip(contexts, targets)):
        context_tensor[index, : len(context)] = torch.tensor(context)
        target_input[index, : len(target) - 1] = torch.tensor(target[:-1])
        target_output[index, : len(target) - 1] = torch.tensor(target[1:])
    return context_tensor, context_len, target_input, target_output


def _log_probability(model: TinyConditionalDecoder, row: dict[str, Any], target_tokens: list[str], vocabulary: dict[str, int], device: torch.device) -> torch.Tensor:
    source = dict(row)
    source["target_tokens"] = target_tokens
    context, lengths, target_input, target_output = _tensorise([source], vocabulary)
    logits = model(context.to(device), lengths.to(device), target_input.to(device))[0]
    target = target_output[0].to(device)
    mask = target != vocabulary[PAD]
    return torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()


def _train_model(
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    vocabulary: dict[str, int],
    *,
    guided: bool,
    device: torch.device,
) -> tuple[TinyConditionalDecoder, list[dict[str, float]]]:
    torch.manual_seed(SEED + (1 if guided else 0))
    random.seed(SEED + (1 if guided else 0))
    model = TinyConditionalDecoder(len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    context, lengths, target_input, target_output = _tensorise(rows, vocabulary)
    pad_id = vocabulary[PAD]
    weights = torch.tensor([float(row["teacher_score"]) if guided else 1.0 for row in rows], dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(context.to(device), lengths.to(device), target_input.to(device))
        token_loss = nn.functional.cross_entropy(logits.transpose(1, 2), target_output.to(device), ignore_index=pad_id, reduction="none")
        sequence_loss = token_loss.sum(dim=1) / (target_output.ne(pad_id).sum(dim=1).to(device).clamp_min(1))
        loss = (sequence_loss * weights).mean()
        pair_loss_value = torch.tensor(0.0, device=device)
        if guided and pairs:
            pair_rows = [row for row in rows if row["record_id"] in {pair["record_id"] for pair in pairs}]
            pair_losses = []
            for pair in pairs:
                source = next(row for row in pair_rows if row["record_id"] == pair["record_id"])
                chosen_lp = _log_probability(model, source, pair["chosen_target_tokens"], vocabulary, device)
                rejected_lp = _log_probability(model, source, pair["rejected_target_tokens"], vocabulary, device)
                pair_losses.append(-nn.functional.logsigmoid(PAIRWISE_BETA * (chosen_lp - rejected_lp)))
            pair_loss_value = torch.stack(pair_losses).mean() if pair_losses else pair_loss_value
            loss = 0.7 * loss + 0.3 * pair_loss_value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_value = float(loss.detach().cpu())
        if epoch == 1 or epoch % 40 == 0:
            history.append({"epoch": float(epoch), "loss": round(epoch_value, 6), "pair_loss": round(float(pair_loss_value.detach().cpu()), 6)})
        if epoch_value < best_loss:
            best_loss = epoch_value
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _generate(model: TinyConditionalDecoder, row: dict[str, Any], vocabulary: dict[str, int], reverse: dict[int, str], device: torch.device, max_len: int) -> list[str]:
    model.eval()
    context, lengths, _, _ = _tensorise([{**row, "target_tokens": [TARGET_BOS, TARGET_EOS]}], vocabulary)
    context = context.to(device)
    lengths = lengths.to(device)
    generated = [TARGET_BOS]
    with torch.inference_mode():
        for _ in range(max_len - 1):
            input_ids = torch.tensor([[vocabulary.get(token, vocabulary[UNK]) for token in generated]], dtype=torch.long, device=device)
            logits = model(context, lengths, input_ids)
            token = reverse[int(logits[0, -1].argmax())]
            generated.append(token)
            if token == TARGET_EOS:
                break
    return generated


def _prediction_metrics(model: TinyConditionalDecoder, rows: list[dict[str, Any]], vocabulary: dict[str, int], reverse: dict[int, str], device: torch.device) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    token_total = token_correct = 0
    exact = next_action_correct = belief_correct = abstain_correct = 0
    for row in rows:
        generated = _generate(model, row, vocabulary, reverse, device, max_len=len(row["target_tokens"]) + 5)
        expected = list(row["target_tokens"])
        compare = min(len(generated), len(expected))
        token_correct += sum(int(generated[position] == expected[position]) for position in range(compare))
        token_total += len(expected)
        expected_action = _token_value(expected, "next_action=")
        generated_action = _token_value(generated, "next_action=")
        expected_belief = _token_value(expected, "final_belief=")
        generated_belief = _token_value(generated, "final_belief=")
        is_abstain = expected_belief == "oracle_gap" and expected_action == "abstain"
        exact_match = generated == expected
        exact += int(exact_match)
        next_action_correct += int(generated_action == expected_action)
        belief_correct += int(generated_belief == expected_belief)
        abstain_correct += int((generated_belief == "oracle_gap" and generated_action == "abstain") == is_abstain)
        details.append({"record_id": row["record_id"], "expected_next_action": expected_action, "generated_next_action": generated_action, "expected_final_belief": expected_belief, "generated_final_belief": generated_belief, "exact_match": exact_match, "generated_tokens": generated})
    total = len(rows)
    return {"count": total, "token_accuracy": round(token_correct / max(token_total, 1), 6), "exact_trajectory_rate": round(exact / max(total, 1), 6), "next_action_accuracy": round(next_action_correct / max(total, 1), 6), "final_belief_accuracy": round(belief_correct / max(total, 1), 6), "abstain_calibration_accuracy": round(abstain_correct / max(total, 1), 6), "details": details}


def _preference_metrics(model: TinyConditionalDecoder, rows: list[dict[str, Any]], pairs: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> dict[str, Any]:
    by_id = {row["record_id"]: row for row in rows}
    wins = []
    for pair in pairs:
        row = by_id[pair["record_id"]]
        chosen = float(_log_probability(model, row, pair["chosen_target_tokens"], vocabulary, device).detach().cpu())
        rejected = float(_log_probability(model, row, pair["rejected_target_tokens"], vocabulary, device).detach().cpu())
        wins.append({"pair_id": pair["pair_id"], "chosen_logp": round(chosen, 6), "rejected_logp": round(rejected, 6), "preferred": chosen > rejected})
    return {"count": len(wins), "preference_win_rate": round(sum(item["preferred"] for item in wins) / max(len(wins), 1), 6), "details": wins}


def main() -> None:
    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    rows, audit = _load_source()
    dataset = _build_dataset(rows, audit)
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vocabulary, reverse = _vocabulary(dataset)
    split_rows = {split: [row for row in dataset["records"] if row["split"] == split] for split in ("train", "route_dev", "family_holdout")}
    split_pairs = {split: [pair for pair in dataset["preferences"] if pair["split"] == split] for split in split_rows}
    if not split_rows["train"] or not split_rows["route_dev"] or not split_rows["family_holdout"]:
        raise RuntimeError(f"invalid PG-270 split counts: {dataset['counts']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_assignment = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    plain, plain_history = _train_model(split_rows["train"], split_pairs["train"], vocabulary, guided=False, device=device)
    guided, guided_history = _train_model(split_rows["train"], split_pairs["train"], vocabulary, guided=True, device=device)
    evaluations: dict[str, Any] = {}
    for name, model in (("plain_sft", plain), ("guided_sft", guided)):
        evaluations[name] = {
            "route_dev": _prediction_metrics(model, split_rows["route_dev"], vocabulary, reverse, device),
            "family_holdout": _prediction_metrics(model, split_rows["family_holdout"], vocabulary, reverse, device),
            "route_dev_preference": _preference_metrics(model, split_rows["route_dev"], split_pairs["route_dev"], vocabulary, device),
            "family_holdout_preference": _preference_metrics(model, split_rows["family_holdout"], split_pairs["family_holdout"], vocabulary, device),
        }
    guided_better = evaluations["guided_sft"]["family_holdout"]["next_action_accuracy"] >= evaluations["plain_sft"]["family_holdout"]["next_action_accuracy"]
    capability_checks = {
        "source_audit_pass": True,
        "context_target_split_pass": bool(dataset["split_contract"]["oracle_target_in_context"] is False),
        "raw_context_free": bool(dataset["split_contract"]["raw_probe_or_response_in_context"] is False),
        "unseen_family_evaluated": dataset["counts"]["family_holdout"] > 0,
        "guided_variant_evaluated": True,
        "guided_not_worse_next_action": guided_better,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg270-teacher-sft-ablation-checkpoint-v1", "vocabulary": vocabulary, "reverse_vocabulary": reverse, "plain_state": plain.state_dict(), "guided_state": guided.state_dict(), "seed": SEED, "device": str(device)}, CHECKPOINT_PATH)
    report = {
        "protocol_id": "pg270-teacher-sft-ablation-v1",
        "schema_version": "pg270-teacher-sft-ablation-report-v1",
        "status": "candidate_ablation_completed",
        "source": {"pg269_dataset": str(PG269_DATASET.relative_to(ROOT)), "pg269_audit": str(PG269_AUDIT.relative_to(ROOT)), "device": str(device), "cuda_assignment": cuda_assignment, "external_network": False, "loopback_only": True},
        "dataset": {"path": str(DATASET_PATH.relative_to(ROOT)), "sha256": dataset["dataset_sha256"], "counts": dataset["counts"], "vocabulary_size": len(vocabulary), "unseen_families": sorted(UNSEEN_FAMILIES)},
        "training": {"epochs": EPOCHS, "hidden_dim": HIDDEN_DIM, "embed_dim": EMBED_DIM, "seed": SEED, "plain_history_tail": plain_history[-5:], "guided_history_tail": guided_history[-5:], "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "teacher_forcing_only": True, "online_weight_update": False, "long_term_memory_write": False, "cuda_assignment": cuda_assignment},
        "evaluations": evaluations,
        "capability_gate": {"status": "passed" if all(capability_checks.values()) else "blocked", "checks": capability_checks, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "status": "candidate_ablation_only", "reason": "small grouped split; needs independent seed, fresh target replay and catastrophic-forgetting canary"},
        "formal_claim": {"allowed": False, "reason": "PG-270 tests teacher signal and process preference wiring, not broad vulnerability detection"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = _sha(report)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg270-teacher-sft-ablation-v1", "schema_version": "pg270-teacher-sft-ablation-protocol-v1", "input_contract": dataset["split_contract"], "training_contract": dataset["contract"], "evaluation_contract": {"route_dev": True, "unseen_family_holdout": True, "plain_vs_guided": True, "preference_pairs_scored": True, "process_reward_labels_present": True}, "gates": {"source_audit_required": True, "raw_context_forbidden": True, "promotion_blocked": True, "memory_blocked": True}, "run": {"capability_gate": report["capability_gate"], "report_sha256": report["report_sha256"]}, "next_experiment": "PG-271 independent seed + fresh Pikachu replay before any offline RL"}
    protocol["protocol_sha256"] = _sha(protocol)
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg270-teacher-sft-ablation-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": dataset["dataset_sha256"], "train_record_ids": [row["record_id"] for row in split_rows["train"]], "route_dev_record_ids": [row["record_id"] for row in split_rows["route_dev"]], "family_holdout_record_ids": [row["record_id"] for row in split_rows["family_holdout"]], "capability_checks": capability_checks, "raw_probe_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_model_context": False, "online_weight_update": False, "long_term_memory_write": False}
    trace["trace_sha256"] = _sha(trace)
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-270 教师指导 SFT + preference/process-reward 小试验", "", f"设备：`{device}`；SFT={dataset['counts']['sft']}；preference={dataset['counts']['preference_pairs']}；family holdout={dataset['counts']['family_holdout']}。", "", "| variant | route dev next-action | family holdout next-action | family preference win |", "|---|---:|---:|---:|", f"| plain SFT | {evaluations['plain_sft']['route_dev']['next_action_accuracy']:.3f} | {evaluations['plain_sft']['family_holdout']['next_action_accuracy']:.3f} | {evaluations['plain_sft']['family_holdout_preference']['preference_win_rate']:.3f} |", f"| guided SFT | {evaluations['guided_sft']['route_dev']['next_action_accuracy']:.3f} | {evaluations['guided_sft']['family_holdout']['next_action_accuracy']:.3f} | {evaluations['guided_sft']['family_holdout_preference']['preference_win_rate']:.3f} |", "", "教师 reference 只作为 target/label；原始 payload、响应正文和 oracle 字段不在模型 context。结果只代表训练信号消融，不能外推为可独立扫描公网的能力。", "", f"能力门：`{report['capability_gate']['status']}`；training promotion=false；memory promotion=false。", f"数据集：`{DATASET_PATH.relative_to(ROOT)}`", f"报告：`{REPORT_PATH.relative_to(ROOT)}`", f"协议：`{PROTOCOL_PATH.relative_to(ROOT)}`", ""]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "cuda_assignment": cuda_assignment, "counts": dataset["counts"], "capability_gate": report["capability_gate"], "report": str(REPORT_PATH.relative_to(ROOT)), "dataset": str(DATASET_PATH.relative_to(ROOT)), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
