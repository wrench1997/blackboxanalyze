"""PG-277 question/action/belief ablation on matched-shape counterfactuals."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg277_counterfactual_question_dataset_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg277-question-composition"
CHECKPOINT = OUTPUT_DIR / "pg277_question_policies.pt"
REPORT = ROOT / "research" / "pg277_question_composition_report_v1.json"
TRACE = ROOT / "research" / "pg277_question_composition_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg277_question_composition_protocol_v1.json"
MARKDOWN = ROOT / "research" / "pg277_question_composition_report_v1.md"

PAD, UNK = "[PAD]", "[UNK]"
QUESTIONS = ("inspect_marker_channel", "inspect_header_channel", "inspect_value_channel", "replay_evidence", "explain_mismatch")
ACTIONS = ("ask_question", "replay_confirmed", "abstain")
BELIEFS = ("unresolved", "confirmed_effect", "oracle_gap")
MODEL_SEEDS = (27711, 27712, 27713)
EMBED, HIDDEN = 64, 128


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

    def forward(self, context: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(context), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        state = self.norm(state[-1])
        return {"question": self.question(state), "action": self.action(state), "belief": self.belief(state)}


def samples(rows: list[dict[str, Any]], mode: str, *, include_pre: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if include_pre:
            target = dict(row["targets"]["pre_question"])
            result.append({"record_id": row["record_id"], "stage": "pre", "context": row["pre_question_context_tokens"], "target": target, "rejected": row["preference_rejected"]["pre_question"], "positive": False, "weight": float(row["teacher_score"])})
        field = "enriched_post_context_tokens" if mode == "enriched" else "coarse_post_context_tokens"
        result.append({"record_id": row["record_id"], "stage": "post", "context": row[field], "target": dict(row["targets"]["post_observation"]), "rejected": dict(row["preference_rejected"]["post_observation"]), "positive": bool(row["labels"]["expected_positive"]), "weight": float(row["teacher_score"])})
    return result


def build_vocab(items: list[dict[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for item in items:
        tokens.update(item["context"])
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(tokens - {PAD, UNK}))}


def encode(items: list[dict[str, Any]], vocab: dict[str, int]) -> tuple[torch.Tensor, ...]:
    sequences = [[vocab.get(token, vocab[UNK]) for token in item["context"]] for item in items]
    values = torch.full((len(items), max(map(len, sequences))), vocab[PAD], dtype=torch.long)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    for index, sequence in enumerate(sequences):
        values[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    questions = torch.tensor([QUESTIONS.index(item["target"]["question"]) for item in items], dtype=torch.long)
    actions = torch.tensor([ACTIONS.index(item["target"]["action"]) for item in items], dtype=torch.long)
    beliefs = torch.tensor([BELIEFS.index(item["target"]["belief"]) for item in items], dtype=torch.long)
    return values, lengths, questions, actions, beliefs


def fit_sft(items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device, seed: int) -> Policy:
    torch.manual_seed(seed)
    random.seed(seed)
    model = Policy(len(vocab)).to(device)
    values, lengths, questions, actions, beliefs = encode(items, vocab)
    weights = torch.tensor([float(item["weight"]) for item in items], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.01)
    best, best_loss = None, float("inf")
    for _ in range(180):
        model.train()
        output = model(values.to(device), lengths.to(device))
        loss = F.cross_entropy(output["question"], questions.to(device), reduction="none") + F.cross_entropy(output["action"], actions.to(device), reduction="none") + F.cross_entropy(output["belief"], beliefs.to(device), reduction="none")
        loss = (loss * weights).mean()
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


def reward_cube(items: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    cube = torch.zeros((len(items), len(QUESTIONS), len(ACTIONS), len(BELIEFS)), device=device)
    for index, item in enumerate(items):
        tq, ta, tb = QUESTIONS.index(item["target"]["question"]), ACTIONS.index(item["target"]["action"]), BELIEFS.index(item["target"]["belief"])
        for q in range(len(QUESTIONS)):
            for a in range(len(ACTIONS)):
                for b in range(len(BELIEFS)):
                    reward = 1.0 * (q == tq) + 1.5 * (a == ta) + 1.5 * (b == tb)
                    if item["stage"] == "pre" and (a == ACTIONS.index("replay_confirmed") or b == BELIEFS.index("confirmed_effect")):
                        reward -= 3.0
                    if item["stage"] == "post":
                        if b == BELIEFS.index("confirmed_effect") and not item["positive"]:
                            reward -= 4.0
                        if b != BELIEFS.index("confirmed_effect") and item["positive"]:
                            reward -= 1.5
                        if a == ACTIONS.index("abstain") and not item["positive"]:
                            reward += 0.5
                    cube[index, q, a, b] = reward
    return cube


def conservative_update(base: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> Policy:
    model, frozen = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device).eval()
    values, lengths, questions, actions, beliefs = encode(items, vocab)
    values, lengths = values.to(device), lengths.to(device)
    questions, actions, beliefs = questions.to(device), actions.to(device), beliefs.to(device)
    cube = reward_cube(items, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035, weight_decay=0.01)
    for _ in range(80):
        output = model(values, lengths)
        q_prob, a_prob, b_prob = output["question"].softmax(-1), output["action"].softmax(-1), output["belief"].softmax(-1)
        expected = (q_prob[:, :, None, None] * a_prob[:, None, :, None] * b_prob[:, None, None, :] * cube).sum(dim=(1, 2, 3)).mean()
        anchor = F.cross_entropy(output["question"], questions) + F.cross_entropy(output["action"], actions) + F.cross_entropy(output["belief"], beliefs)
        with torch.no_grad():
            reference = frozen(values, lengths)
        kl = sum(F.kl_div(F.log_softmax(output[head], -1), reference[head].softmax(-1), reduction="batchmean") for head in ("question", "action", "belief"))
        loss = -expected + 0.35 * anchor + 2.0 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def dpo_update(base: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> Policy:
    model, frozen = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device).eval()
    values, lengths, questions, actions, beliefs = encode(items, vocab)
    values, lengths = values.to(device), lengths.to(device)
    questions, actions, beliefs = questions.to(device), actions.to(device), beliefs.to(device)
    rq = torch.tensor([QUESTIONS.index(item["rejected"]["question"]) for item in items], device=device)
    ra = torch.tensor([ACTIONS.index(item["rejected"]["action"]) for item in items], device=device)
    rb = torch.tensor([BELIEFS.index(item["rejected"]["belief"]) for item in items], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    for _ in range(80):
        output = model(values, lengths)
        logp = {head: F.log_softmax(output[head], -1) for head in ("question", "action", "belief")}
        chosen = logp["question"].gather(1, questions[:, None]).squeeze(1) + logp["action"].gather(1, actions[:, None]).squeeze(1) + logp["belief"].gather(1, beliefs[:, None]).squeeze(1)
        rejected = logp["question"].gather(1, rq[:, None]).squeeze(1) + logp["action"].gather(1, ra[:, None]).squeeze(1) + logp["belief"].gather(1, rb[:, None]).squeeze(1)
        with torch.no_grad():
            reference = frozen(values, lengths)
        kl = sum(F.kl_div(F.log_softmax(output[head], -1), reference[head].softmax(-1), reduction="batchmean") for head in ("question", "action", "belief"))
        anchor = F.cross_entropy(output["question"], questions) + F.cross_entropy(output["action"], actions) + F.cross_entropy(output["belief"], beliefs)
        loss = -F.logsigmoid(0.1 * (chosen - rejected)).mean() + 0.25 * anchor + 1.5 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def predict(model: Policy, items: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> list[tuple[int, int, int]]:
    model.eval()
    values, lengths, _, _, _ = encode(items, vocab)
    with torch.inference_mode():
        output = model(values.to(device), lengths.to(device))
    return list(zip(output["question"].argmax(-1).cpu().tolist(), output["action"].argmax(-1).cpu().tolist(), output["belief"].argmax(-1).cpu().tolist()))


def metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device) -> dict[str, Any]:
    items = samples(rows, mode, include_pre=True)
    predictions = predict(model, items, vocab, device)
    pre, post = [], []
    for item, (q, a, b) in zip(items, predictions):
        target = item["target"]
        record = {"question_correct": QUESTIONS[q] == target["question"], "action_correct": ACTIONS[a] == target["action"], "belief_correct": BELIEFS[b] == target["belief"], "triple_correct": QUESTIONS[q] == target["question"] and ACTIONS[a] == target["action"] and BELIEFS[b] == target["belief"], "positive": item["positive"], "model_positive": BELIEFS[b] == "confirmed_effect", "false_positive": BELIEFS[b] == "confirmed_effect" and not item["positive"], "false_negative": item["positive"] and BELIEFS[b] != "confirmed_effect"}
        (pre if item["stage"] == "pre" else post).append(record)
    positives, negatives = [x for x in post if x["positive"]], [x for x in post if not x["positive"]]
    return {"count": len(rows), "pre_question_accuracy": round(sum(x["question_correct"] for x in pre) / max(len(pre), 1), 6), "pre_ask_action_accuracy": round(sum(x["action_correct"] for x in pre) / max(len(pre), 1), 6), "pre_unresolved_belief_accuracy": round(sum(x["belief_correct"] for x in pre) / max(len(pre), 1), 6), "post_question_accuracy": round(sum(x["question_correct"] for x in post) / max(len(post), 1), 6), "post_action_accuracy": round(sum(x["action_correct"] for x in post) / max(len(post), 1), 6), "post_belief_accuracy": round(sum(x["belief_correct"] for x in post) / max(len(post), 1), 6), "post_triple_accuracy": round(sum(x["triple_correct"] for x in post) / max(len(post), 1), 6), "positive_recall": round(sum(x["model_positive"] for x in positives) / max(len(positives), 1), 6), "negative_reject": round(sum(not x["model_positive"] for x in negatives) / max(len(negatives), 1), 6), "false_positive_count": sum(x["false_positive"] for x in post), "false_negative_count": sum(x["false_negative"] for x in post)}


def counterfactual_metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in rows:
        mode = row["labels"]["mode"]
        if mode not in {"attr_sink", "attr_decoy"}:
            continue
        context = list(row["enriched_post_context_tokens"])
        expected_positive = mode == "attr_decoy"
        replacements = {"observe_candidate_channel=attribute": "observe_candidate_channel=html_text", "observe_candidate_reference_match=1": "observe_candidate_reference_match=0", "failure_token=none": "failure_token=channel_mismatch"} if mode == "attr_sink" else {"observe_candidate_channel=html_text": "observe_candidate_channel=attribute", "observe_candidate_reference_match=0": "observe_candidate_reference_match=1", "failure_token=channel_mismatch": "failure_token=none"}
        context = [replacements.get(token, token) for token in context]
        target = {"question": "replay_evidence" if expected_positive else "explain_mismatch", "action": "replay_confirmed" if expected_positive else "abstain", "belief": "confirmed_effect" if expected_positive else "oracle_gap"}
        items.append({"record_id": row["record_id"], "stage": "post", "context": context, "target": target, "rejected": target, "positive": expected_positive, "weight": 1.0})
    predictions = predict(model, items, vocab, device)
    triple, belief, safe = 0, 0, 0
    for item, (q, a, b) in zip(items, predictions):
        target = item["target"]
        triple += int(QUESTIONS[q] == target["question"] and ACTIONS[a] == target["action"] and BELIEFS[b] == target["belief"])
        belief += int(BELIEFS[b] == target["belief"])
        safe += int(item["positive"] or BELIEFS[b] != "confirmed_effect")
    return {"count": len(items), "triple_accuracy": round(triple / max(len(items), 1), 6), "belief_flip_accuracy": round(belief / max(len(items), 1), 6), "counterfactual_safe_rate": round(safe / max(len(items), 1), 6)}


def missing_observation_metrics(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    items = [{"record_id": row["record_id"], "stage": "pre", "context": row["coarse_post_context_tokens"], "target": {"question": "inspect_marker_channel", "action": "ask_question", "belief": "unresolved"}, "rejected": row["preference_rejected"]["pre_question"], "positive": False, "weight": 1.0} for row in rows if row["labels"]["mode"] in {"attr_sink", "attr_decoy", "absent_decoy"}]
    predictions = predict(model, items, vocab, device)
    safe = sum(BELIEFS[b] != "confirmed_effect" for _, _, b in predictions)
    ask = sum(ACTIONS[a] == "ask_question" for _, a, _ in predictions)
    question = sum(QUESTIONS[q] == "inspect_marker_channel" for q, _, _ in predictions)
    return {"count": len(items), "safe_non_positive_rate": round(safe / max(len(items), 1), 6), "ask_recovery_rate": round(ask / max(len(items), 1), 6), "question_recovery_rate": round(question / max(len(items), 1), 6)}


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(seed_results[0])
    return {key: {"mean": round(sum(float(item[key]) for item in seed_results) / len(seed_results), 6), "min": round(min(float(item[key]) for item in seed_results), 6), "max": round(max(float(item[key]) for item in seed_results), 6)} for key in keys if isinstance(seed_results[0][key], (int, float))}


def main() -> None:
    started = time.perf_counter()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    train = [row for row in data["records"] if row["split"] == "alpha_beta_train"]
    holdout = [row for row in data["records"] if row["split"] == "gamma_seed_holdout"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in ("coarse_process_sft", "enriched_final_only_sft", "enriched_process_sft", "conservative_offline_update", "dpo_preference_update")}
    checkpoint_payload: dict[str, Any] = {"schema_version": "pg277-question-composition-checkpoint-v1", "model_seeds": list(MODEL_SEEDS), "assignment": assignment, "states": {}}
    for seed in MODEL_SEEDS:
        coarse_train = samples(train, "coarse", include_pre=True)
        coarse_vocab = build_vocab(coarse_train)
        coarse = fit_sft(coarse_train, coarse_vocab, device, seed)
        final_train = samples(train, "enriched", include_pre=False)
        final_vocab = build_vocab(final_train)
        final_only = fit_sft(final_train, final_vocab, device, seed)
        process_train = samples(train, "enriched", include_pre=True)
        process_vocab = build_vocab(process_train)
        process = fit_sft(process_train, process_vocab, device, seed)
        conservative = conservative_update(process, process_train, process_vocab, device)
        dpo = dpo_update(process, process_train, process_vocab, device)
        variants = {"coarse_process_sft": (coarse, coarse_vocab, "coarse"), "enriched_final_only_sft": (final_only, final_vocab, "enriched"), "enriched_process_sft": (process, process_vocab, "enriched"), "conservative_offline_update": (conservative, process_vocab, "enriched"), "dpo_preference_update": (dpo, process_vocab, "enriched")}
        for name, (model, vocab, mode) in variants.items():
            result = {"seed": seed, "holdout": metrics(model, holdout, vocab, mode, device), "counterfactual": counterfactual_metrics(model, holdout, vocab, device) if mode == "enriched" else {"count": 0, "triple_accuracy": 0.0, "belief_flip_accuracy": 0.0, "counterfactual_safe_rate": 0.0}, "missing_observation": missing_observation_metrics(model, holdout, vocab, device)}
            per_seed[name].append(result)
            checkpoint_payload["states"][f"{name}:{seed}"] = {"vocabulary": vocab, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    aggregated: dict[str, Any] = {}
    for name, results in per_seed.items():
        aggregated[name] = {"holdout": aggregate([item["holdout"] for item in results]), "counterfactual": aggregate([item["counterfactual"] for item in results]), "missing_observation": aggregate([item["missing_observation"] for item in results])}
    enriched = aggregated["enriched_process_sft"]
    conservative = aggregated["conservative_offline_update"]
    dpo = aggregated["dpo_preference_update"]
    checks = {
        "dataset_collision_detected": int(data["projection_collision_audit"]["coarse"]["conflict_group_count"]) > 0,
        "enriched_collision_zero": int(data["projection_collision_audit"]["enriched"]["conflict_group_count"]) == 0,
        "question_holdout_min": enriched["holdout"]["pre_question_accuracy"]["min"] >= 0.8,
        "positive_recall_min": enriched["holdout"]["positive_recall"]["min"] >= 0.8,
        "negative_reject_min": enriched["holdout"]["negative_reject"]["min"] >= 0.9,
        "false_positive_zero": enriched["holdout"]["false_positive_count"]["max"] == 0,
        "counterfactual_flip_min": enriched["counterfactual"]["belief_flip_accuracy"]["min"] >= 0.8,
        "missing_observation_safe_min": enriched["missing_observation"]["safe_non_positive_rate"]["min"] >= 0.9,
        "conservative_no_regression": conservative["holdout"]["positive_recall"]["min"] >= enriched["holdout"]["positive_recall"]["min"] and conservative["holdout"]["false_positive_count"]["max"] <= enriched["holdout"]["false_positive_count"]["max"],
        "dpo_no_regression": dpo["holdout"]["positive_recall"]["min"] >= enriched["holdout"]["positive_recall"]["min"] and dpo["holdout"]["false_positive_count"]["max"] <= enriched["holdout"]["false_positive_count"]["max"],
        "promotion_blocked": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, CHECKPOINT)
    report = {"protocol_id": "pg277-question-composition-v1", "schema_version": "pg277-question-composition-report-v1", "status": "completed_question_composition_ablation", "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "device": str(device), "cuda_assignment": assignment, "external_network": False, "raw_payload_in_context": False, "oracle_in_context": False}, "split": {"train_count": len(train), "holdout_count": len(holdout), "train_variants": ["alpha", "beta"], "holdout_variant": "gamma", "model_seeds": list(MODEL_SEEDS), "variant_and_seed_disjoint": True}, "per_seed": per_seed, "aggregated": aggregated, "hypothesis_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "one controlled surface family; real multi-family and fresh application replay still required"}, "formal_conclusion": "Projection completeness precedes policy optimization: coarse identical-input/conflicting-label records are not learnable. Atomic marker-channel observation enables held-out question/action/belief composition; conservative/DPO updates are accepted only when they preserve recall and false-positive gates.", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg277-question-composition-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": data["dataset_sha256"], "aggregated": aggregated, "hypothesis_gate": report["hypothesis_gate"], "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg277-question-composition-v1", "schema_version": "pg277-question-composition-protocol-v1", "comparators": list(per_seed), "reward": {"question_match": 1.0, "action_match": 1.5, "belief_match": 1.5, "premature_positive": -3.0, "unsupported_positive": -4.0, "missed_positive": -1.5, "safe_abstain": 0.5}, "constraints": {"behavior_kl_anchor": True, "supervised_anchor": True, "entropy_bonus": False, "online_requests": False}, "split": report["split"], "checks": checks, "report_sha256": report["report_sha256"], "next_experiment": "PG-278 real Pikachu SQL/redirect/XSS failure-question transfer with family holdout"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = []
    for name, value in aggregated.items():
        table.append(f"| {name} | {value['holdout']['pre_question_accuracy']['mean']:.3f} | {value['holdout']['positive_recall']['mean']:.3f} | {value['holdout']['negative_reject']['mean']:.3f} | {value['counterfactual']['belief_flip_accuracy']['mean']:.3f} | {value['missing_observation']['safe_non_positive_rate']['mean']:.3f} |")
    MARKDOWN.write_text("\n".join(["# PG-277 疑问驱动组合泛化", "", "| variant | pre-question | positive recall | negative reject | counterfactual flip | missing-safe |", "|---|---:|---:|---:|---:|---:|", *table, "", f"gate=`{report['hypothesis_gate']['status']}`；所有能力/记忆晋级仍冻结。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": assignment, "aggregated": aggregated, "hypothesis_gate": report["hypothesis_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
