"""PG-278 policy study: missing-slot question binding and failure-to-repair.

The model receives abstract request/response tokens only.  It never receives a
raw probe, raw response body, evaluator/oracle value, or a real target URL.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg278_multifamily_question_dataset_v1.json"
DATASET_AUDIT = ROOT / "research" / "pg278_multifamily_question_dataset_audit_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg278-multifamily-question-policy"
CHECKPOINT = OUTPUT_DIR / "pg278_question_policies.pt"
REPORT = ROOT / "research" / "pg278_multifamily_question_policy_report_v1.json"
TRACE = ROOT / "research" / "pg278_multifamily_question_policy_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg278_multifamily_question_policy_protocol_v1.json"
MARKDOWN = ROOT / "research" / "pg278_multifamily_question_policy_report_v1.md"

PAD, UNK = "[PAD]", "[UNK]"
QUESTIONS = ("inspect_effect_channel", "inspect_control_comparison", "replay_evidence", "explain_failure")
ACTIONS = ("ask_observation", "review_evidence", "abstain")
BELIEFS = ("unresolved", "supported", "rejected")
SLOTS = (
    "dom_render_channel", "dom_control_alignment",
    "sql_response_shape", "sql_baseline_delta",
    "redirect_status_hop", "redirect_location_scope",
    "logic_outcome_transition", "logic_invariant_control",
)
MODEL_SEEDS = (27811, 27812, 27813)
EMBED, HIDDEN = 72, 160


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class Policy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED)
        self.encoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.question = nn.Linear(HIDDEN, len(QUESTIONS))
        self.action = nn.Linear(HIDDEN, len(ACTIONS))
        self.belief = nn.Linear(HIDDEN, len(BELIEFS))
        self.slot = nn.Linear(HIDDEN, len(SLOTS))

    def forward(self, context: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(context), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        state = self.norm(state[-1])
        return {"question": self.question(state), "action": self.action(state), "belief": self.belief(state), "slot": self.slot(state)}


def samples(rows: list[dict[str, Any]], mode: str, *, include_pre: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if include_pre:
            field = "coarse_pre_question_context_tokens" if mode == "coarse" else "pre_question_context_tokens"
            result.append({"record_id": row["record_id"], "pair_id": row["pair_id"], "family": row["family"], "stage": "pre", "context": row[field], "target": dict(row["targets"]["pre_question"]), "rejected": dict(row["preference_rejected"]["pre_question"][0]), "positive": False})
        result.append({"record_id": row["record_id"], "pair_id": row["pair_id"], "family": row["family"], "stage": "post", "context": row["post_observation_context_tokens"], "target": dict(row["targets"]["post_observation"]), "rejected": dict(row["preference_rejected"]["post_observation"][0]), "positive": bool(row["labels"]["expected_positive"])})
    return result


def build_vocab(items: list[dict[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for item in items:
        tokens.update(str(token) for token in item["context"])
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(tokens - {PAD, UNK}))}


def encode(items: list[dict[str, Any]], vocab: dict[str, int]) -> tuple[torch.Tensor, ...]:
    sequences = [[vocab.get(str(token), vocab[UNK]) for token in item["context"]] for item in items]
    values = torch.full((len(items), max(map(len, sequences))), vocab[PAD], dtype=torch.long)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    questions = torch.tensor([QUESTIONS.index(item["target"]["question"]) for item in items], dtype=torch.long)
    actions = torch.tensor([ACTIONS.index(item["target"]["action"]) for item in items], dtype=torch.long)
    beliefs = torch.tensor([BELIEFS.index(item["target"]["belief"]) for item in items], dtype=torch.long)
    slots = torch.tensor([SLOTS.index(item["target"]["slot"]) for item in items], dtype=torch.long)
    return values, lengths, questions, actions, beliefs, slots


def fit_sft(items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device, seed: int) -> Policy:
    torch.manual_seed(seed)
    random.seed(seed)
    model = Policy(len(vocab)).to(device)
    values, lengths, questions, actions, beliefs, slots = encode(items, vocab)
    values, lengths = values.to(device), lengths.to(device)
    questions, actions, beliefs, slots = questions.to(device), actions.to(device), beliefs.to(device), slots.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0045, weight_decay=0.012)
    best, best_loss = None, float("inf")
    for _ in range(260):
        model.train()
        output = model(values, lengths)
        loss = F.cross_entropy(output["question"], questions) + F.cross_entropy(output["action"], actions) + F.cross_entropy(output["belief"], beliefs) + F.cross_entropy(output["slot"], slots)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best)
    return model


def _all_reward(items: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    cube = torch.zeros((len(items), len(QUESTIONS), len(ACTIONS), len(BELIEFS), len(SLOTS)), device=device)
    for index, item in enumerate(items):
        tq, ta, tb, ts = QUESTIONS.index(item["target"]["question"]), ACTIONS.index(item["target"]["action"]), BELIEFS.index(item["target"]["belief"]), SLOTS.index(item["target"]["slot"])
        for q in range(len(QUESTIONS)):
            for a in range(len(ACTIONS)):
                for b in range(len(BELIEFS)):
                    for slot in range(len(SLOTS)):
                        reward = 1.0 * (q == tq) + 1.3 * (a == ta) + 1.3 * (b == tb) + 1.1 * (slot == ts)
                        if item["stage"] == "pre" and (a != ACTIONS.index("ask_observation") or b != BELIEFS.index("unresolved")):
                            reward -= 3.0
                        if item["stage"] == "post" and item["positive"] and b != BELIEFS.index("supported"):
                            reward -= 2.0
                        if item["stage"] == "post" and not item["positive"] and b == BELIEFS.index("supported"):
                            reward -= 4.0
                        cube[index, q, a, b, slot] = reward
    return cube


def conservative_update(base: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> Policy:
    model, frozen = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device).eval()
    values, lengths, questions, actions, beliefs, slots = encode(items, vocab)
    values, lengths = values.to(device), lengths.to(device)
    questions, actions, beliefs, slots = questions.to(device), actions.to(device), beliefs.to(device), slots.to(device)
    cube = _all_reward(items, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00028, weight_decay=0.012)
    for _ in range(110):
        output = model(values, lengths)
        q, a, b, slot = (output["question"].softmax(-1), output["action"].softmax(-1), output["belief"].softmax(-1), output["slot"].softmax(-1))
        expected = (q[:, :, None, None, None] * a[:, None, :, None, None] * b[:, None, None, :, None] * slot[:, None, None, None, :] * cube).sum(dim=(1, 2, 3, 4)).mean()
        anchor = F.cross_entropy(output["question"], questions) + F.cross_entropy(output["action"], actions) + F.cross_entropy(output["belief"], beliefs) + F.cross_entropy(output["slot"], slots)
        with torch.no_grad():
            reference = frozen(values, lengths)
        kl = sum(F.kl_div(F.log_softmax(output[head], -1), reference[head].softmax(-1), reduction="batchmean") for head in ("question", "action", "belief", "slot"))
        loss = -expected + 0.35 * anchor + 1.8 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def dpo_update(base: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> Policy:
    model, frozen = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device).eval()
    values, lengths, questions, actions, beliefs, slots = encode(items, vocab)
    values, lengths = values.to(device), lengths.to(device)
    questions, actions, beliefs, slots = questions.to(device), actions.to(device), beliefs.to(device), slots.to(device)
    rq = torch.tensor([QUESTIONS.index(item["rejected"]["question"]) for item in items], device=device)
    ra = torch.tensor([ACTIONS.index(item["rejected"]["action"]) for item in items], device=device)
    rb = torch.tensor([BELIEFS.index(item["rejected"]["belief"]) for item in items], device=device)
    rs = torch.tensor([SLOTS.index(item["rejected"]["slot"]) for item in items], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00024, weight_decay=0.012)
    for _ in range(110):
        output = model(values, lengths)
        logp = {head: F.log_softmax(output[head], -1) for head in ("question", "action", "belief", "slot")}
        chosen = logp["question"].gather(1, questions[:, None]).squeeze(1) + logp["action"].gather(1, actions[:, None]).squeeze(1) + logp["belief"].gather(1, beliefs[:, None]).squeeze(1) + logp["slot"].gather(1, slots[:, None]).squeeze(1)
        rejected = logp["question"].gather(1, rq[:, None]).squeeze(1) + logp["action"].gather(1, ra[:, None]).squeeze(1) + logp["belief"].gather(1, rb[:, None]).squeeze(1) + logp["slot"].gather(1, rs[:, None]).squeeze(1)
        with torch.no_grad():
            reference = frozen(values, lengths)
        kl = sum(F.kl_div(F.log_softmax(output[head], -1), reference[head].softmax(-1), reduction="batchmean") for head in ("question", "action", "belief", "slot"))
        anchor = F.cross_entropy(output["question"], questions) + F.cross_entropy(output["action"], actions) + F.cross_entropy(output["belief"], beliefs) + F.cross_entropy(output["slot"], slots)
        loss = -F.logsigmoid(0.12 * (chosen - rejected)).mean() + 0.22 * anchor + 1.4 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def predict(model: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> list[tuple[int, int, int, int]]:
    model.eval()
    values, lengths, _, _, _, _ = encode(items, vocab)
    with torch.inference_mode():
        output = model(values.to(device), lengths.to(device))
    return list(zip(output["question"].argmax(-1).cpu().tolist(), output["action"].argmax(-1).cpu().tolist(), output["belief"].argmax(-1).cpu().tolist(), output["slot"].argmax(-1).cpu().tolist()))


def _score(items: list[dict[str, Any]], predictions: list[tuple[int, int, int, int]], *, score_slot: bool = True) -> dict[str, Any]:
    pre, post = [], []
    for item, (q, a, b, s) in zip(items, predictions):
        target = item["target"]
        question_correct = QUESTIONS[q] == target["question"]
        action_correct = ACTIONS[a] == target["action"]
        belief_correct = BELIEFS[b] == target["belief"]
        slot_correct = SLOTS[s] == target["slot"]
        transition_correct = question_correct and action_correct and belief_correct and (slot_correct or not score_slot)
        record = {"question_correct": question_correct, "action_correct": action_correct, "belief_correct": belief_correct, "slot_correct": slot_correct, "transition_correct": transition_correct, "positive": item["positive"], "model_supported": BELIEFS[b] == "supported"}
        (pre if item["stage"] == "pre" else post).append(record)
    avg = lambda collection, key: round(sum(bool(item[key]) for item in collection) / max(len(collection), 1), 6)
    positives, negatives = [item for item in post if item["positive"]], [item for item in post if not item["positive"]]
    return {
        "pre_count": len(pre),
        "post_count": len(post),
        "pre_question_accuracy": avg(pre, "question_correct"),
        "pre_action_accuracy": avg(pre, "action_correct"),
        "pre_belief_accuracy": avg(pre, "belief_correct"),
        "pre_slot_accuracy": avg(pre, "slot_correct") if score_slot else None,
        "pre_transition_accuracy": avg(pre, "transition_correct"),
        "post_question_accuracy": avg(post, "question_correct"),
        "post_action_accuracy": avg(post, "action_correct"),
        "post_belief_accuracy": avg(post, "belief_correct"),
        "post_slot_accuracy": avg(post, "slot_correct") if score_slot else None,
        "post_transition_accuracy": avg(post, "transition_correct"),
        "positive_recall": round(sum(item["model_supported"] for item in positives) / max(len(positives), 1), 6),
        "negative_reject": round(sum(not item["model_supported"] for item in negatives) / max(len(negatives), 1), 6),
        "false_positive_count": sum(item["model_supported"] for item in negatives),
        "false_negative_count": sum(not item["model_supported"] for item in positives),
    }


def metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device) -> dict[str, Any]:
    items = samples(rows, mode, include_pre=True)
    return _score(items, predict(model, items, vocab, device))


def missing_metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    items = []
    for row in rows:
        target = dict(row["targets"]["pre_question"])
        items.append({"record_id": row["record_id"], "pair_id": row["pair_id"], "family": row["family"], "stage": "pre", "context": row["coarse_pre_question_context_tokens"], "target": target, "rejected": target, "positive": False})
    preds = predict(model, items, vocab, device)
    return {"count": len(items), "ask_rate": round(sum(ACTIONS[a] == "ask_observation" for _, a, _, _ in preds) / max(len(preds), 1), 6), "safe_non_supported_rate": round(sum(BELIEFS[b] != "supported" for _, _, b, _ in preds) / max(len(preds), 1), 6), "exact_slot_recovery_rate": round(sum(SLOTS[s] == item["target"]["slot"] for item, (_, _, _, s) in zip(items, preds)) / max(len(preds), 1), 6)}


def pair_flip_metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    pairs = [items for items in groups.values() if len(items) == 2]
    items = [{"record_id": row["record_id"], "pair_id": row["pair_id"], "family": row["family"], "stage": "post", "context": row["post_observation_context_tokens"], "target": dict(row["targets"]["post_observation"]), "rejected": dict(row["preference_rejected"]["post_observation"][0]), "positive": bool(row["labels"]["expected_positive"])} for pair in pairs for row in pair]
    pred_map = {item["record_id"]: pred for item, pred in zip(items, predict(model, items, vocab, device))}
    correct_pairs = 0
    for pair in pairs:
        okay = True
        for row in pair:
            q, a, b, s = pred_map[row["record_id"]]
            target = row["targets"]["post_observation"]
            okay = okay and QUESTIONS[q] == target["question"] and ACTIONS[a] == target["action"] and BELIEFS[b] == target["belief"] and SLOTS[s] == target["slot"]
        correct_pairs += int(okay)
    return {"pair_count": len(pairs), "paired_counterfactual_transition_accuracy": round(correct_pairs / max(len(pairs), 1), 6)}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(key for key, value in results[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    return {key: {"mean": round(sum(float(item[key]) for item in results) / len(results), 6), "min": round(min(float(item[key]) for item in results), 6), "max": round(max(float(item[key]) for item in results), 6)} for key in keys}


def family_holdout_metrics(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    families = sorted({str(row["family"]) for row in rows})
    for family in families:
        train, holdout = [row for row in rows if row["family"] != family and row["split"] == "implementation_train"], [row for row in rows if row["family"] == family and row["split"] == "implementation_holdout"]
        per_seed: list[dict[str, Any]] = []
        for seed in MODEL_SEEDS:
            items = samples(train, "enriched", include_pre=True)
            vocab = build_vocab(items)
            model = fit_sft(items, vocab, device, seed + 97)
            eval_items = samples(holdout, "enriched", include_pre=True)
            per_seed.append(_score(eval_items, predict(model, eval_items, vocab, device), score_slot=False))
        result[family] = aggregate(per_seed)
    return result


def main() -> None:
    started = time.perf_counter()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("PG-278 dataset audit must pass before training")
    train = [row for row in data["records"] if row["split"] == "implementation_train"]
    holdout = [row for row in data["records"] if row["split"] == "implementation_holdout"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in ("coarse_process_sft", "final_only_sft", "enriched_process_sft", "conservative_offline_update", "dpo_preference_update")}
    checkpoints: dict[str, Any] = {}
    for seed in MODEL_SEEDS:
        coarse_items = samples(train, "coarse", include_pre=True)
        coarse_vocab = build_vocab(coarse_items)
        coarse = fit_sft(coarse_items, coarse_vocab, device, seed)
        final_items = samples(train, "enriched", include_pre=False)
        final_vocab = build_vocab(final_items)
        final_only = fit_sft(final_items, final_vocab, device, seed)
        process_items = samples(train, "enriched", include_pre=True)
        process_vocab = build_vocab(process_items)
        process = fit_sft(process_items, process_vocab, device, seed)
        conservative = conservative_update(process, process_items, process_vocab, device)
        dpo = dpo_update(process, process_items, process_vocab, device)
        models = {"coarse_process_sft": (coarse, coarse_vocab, "coarse"), "final_only_sft": (final_only, final_vocab, "enriched"), "enriched_process_sft": (process, process_vocab, "enriched"), "conservative_offline_update": (conservative, process_vocab, "enriched"), "dpo_preference_update": (dpo, process_vocab, "enriched")}
        for name, (model, vocab, mode) in models.items():
            per_seed[name].append({"seed": seed, "implementation_holdout": metrics(model, holdout, vocab, mode, device), "paired_counterfactual": pair_flip_metrics(model, holdout, vocab, device), "missing_observation": missing_metrics(model, holdout, vocab, device)})
            checkpoints[f"{name}:{seed}"] = {"vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    aggregated = {name: {"implementation_holdout": aggregate([item["implementation_holdout"] for item in values]), "paired_counterfactual": aggregate([item["paired_counterfactual"] for item in values]), "missing_observation": aggregate([item["missing_observation"] for item in values])} for name, values in per_seed.items()}
    family_holdout = family_holdout_metrics(data["records"], device)
    process = aggregated["enriched_process_sft"]
    conservative = aggregated["conservative_offline_update"]
    dpo = aggregated["dpo_preference_update"]
    family_question_min = min(value["pre_question_accuracy"]["min"] for value in family_holdout.values())
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "coarse_collision_detected": int(data["projection_collision_audit"]["coarse"]["conflict_group_count"]) > 0,
        "enriched_collision_zero": int(data["projection_collision_audit"]["enriched"]["conflict_group_count"]) == 0,
        "post_observation_collision_zero": int(data["projection_collision_audit"]["post"]["conflict_group_count"]) == 0,
        "implementation_pre_transition_min": process["implementation_holdout"]["pre_transition_accuracy"]["min"] >= 0.9,
        "implementation_post_transition_min": process["implementation_holdout"]["post_transition_accuracy"]["min"] >= 0.9,
        "implementation_slot_min": process["implementation_holdout"]["pre_slot_accuracy"]["min"] >= 0.9,
        "paired_counterfactual_min": process["paired_counterfactual"]["paired_counterfactual_transition_accuracy"]["min"] >= 0.9,
        "missing_safe_min": process["missing_observation"]["safe_non_supported_rate"]["min"] >= 0.95,
        "family_holdout_question_min": family_question_min >= 0.9,
        "conservative_no_regression": conservative["implementation_holdout"]["pre_transition_accuracy"]["min"] >= process["implementation_holdout"]["pre_transition_accuracy"]["min"] and conservative["implementation_holdout"]["false_positive_count"]["max"] <= process["implementation_holdout"]["false_positive_count"]["max"],
        "dpo_no_regression": dpo["implementation_holdout"]["pre_transition_accuracy"]["min"] >= process["implementation_holdout"]["pre_transition_accuracy"]["min"] and dpo["implementation_holdout"]["false_positive_count"]["max"] <= process["implementation_holdout"]["false_positive_count"]["max"],
        "promotion_blocked": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg278-question-policy-checkpoint-v1", "assignment": assignment, "model_seeds": list(MODEL_SEEDS), "states": checkpoints}, CHECKPOINT)
    report = {
        "protocol_id": "pg278-multifamily-question-policy-v1",
        "schema_version": "pg278-multifamily-question-policy-report-v1",
        "status": "completed_controlled_multifamily_question_policy_study",
        "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "dataset_audit": str(DATASET_AUDIT.relative_to(ROOT)), "dataset_audit_sha256": audit["audit_sha256"], "device": str(device), "cuda_assignment": assignment, "external_network": False, "raw_payload_in_context": False, "raw_response_body_in_context": False, "oracle_in_context": False},
        "split": {"train_count": len(train), "holdout_count": len(holdout), "implementation_holdout": True, "families": sorted({row["family"] for row in data["records"]}), "model_seeds": list(MODEL_SEEDS)},
        "per_seed": per_seed,
        "aggregated": aggregated,
        "family_holdout_abstract_question": family_holdout,
        "hypothesis_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": "controlled_slot_binding_only" if all(checks.values()) else False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "all PG-278 examples are controlled loopback fixtures; the study has zero real multi-family gold rows"},
        "formal_conclusion": "When the missing observation is represented in the Rule-IR state, process training can be tested for question/slot binding across held-out implementations. Coarse contexts intentionally collide and are diagnostic only. Family-holdout question accuracy scores a shared abstract observation role, not unseen slot decoding; it must not be interpreted as real-application vulnerability discovery.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg278-multifamily-question-policy-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": data["dataset_sha256"], "aggregated": aggregated, "family_holdout_abstract_question": family_holdout, "hypothesis_gate": report["hypothesis_gate"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg278-multifamily-question-policy-v1", "schema_version": "pg278-multifamily-question-policy-protocol-v1", "comparators": list(per_seed), "reward": {"question_match": 1.0, "action_match": 1.3, "belief_match": 1.3, "slot_match": 1.1, "premature_non_ask": -3.0, "unsupported_positive": -4.0}, "constraints": {"behavior_kl_anchor": True, "supervised_anchor": True, "external_requests": False, "controlled_loopback_only": True}, "split": report["split"], "checks": checks, "report_sha256": report["report_sha256"], "next_experiment": "PG-279 collect real local replay records with the same slots, then test a source-heldout, frozen retention suite"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-278 多族缺失观测—失败修复策略", "", "| policy | pre transition (min) | post transition (min) | pair flip (min) | missing-safe (min) |", "|---|---:|---:|---:|---:|"]
    for name, result in aggregated.items():
        lines.append(f"| {name} | {result['implementation_holdout']['pre_transition_accuracy']['min']:.3f} | {result['implementation_holdout']['post_transition_accuracy']['min']:.3f} | {result['paired_counterfactual']['paired_counterfactual_transition_accuracy']['min']:.3f} | {result['missing_observation']['safe_non_supported_rate']['min']:.3f} |")
    lines.extend(["", f"gate=`{report['hypothesis_gate']['status']}`；可声称范围仅为受控多族 slot binding，真实靶场/长期记忆晋级仍冻结。", ""])
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": assignment, "hypothesis_gate": report["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
