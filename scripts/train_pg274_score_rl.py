"""PG-274: compare SFT, score-weighted SFT and constrained offline RL.

The policy consumes PG-273 generic observation/question tokens.  v1 is the
training implementation and v2 is never used for updates.  Offline RL samples
abstract actions against recorded typed-oracle labels; it never sends a new
request and never sees raw probes or response bodies.
"""

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
DATASET = ROOT / "research" / "pg273_composition_dataset_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "pg274-score-rl"
CHECKPOINT = OUTPUT_DIR / "score_rl_policy.pt"
REPORT = ROOT / "research" / "pg274_score_rl_report_v1.json"
TRACE = ROOT / "research" / "pg274_score_rl_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg274_score_rl_protocol_v1.json"
MARKDOWN = ROOT / "research" / "pg274_score_rl_report_v1.md"

SEED = 27401
EMBED_DIM = 64
HIDDEN_DIM = 128
SFT_EPOCHS = 240
RL_EPOCHS = 360
LR = 0.008
RL_LR = 0.002
PAD = "[PAD]"
UNK = "[UNK]"
ACTIONS = ("abstain", "replay_confirmed", "diagnose_failure", "candidate_probe")
BELIEFS = ("oracle_gap", "confirmed_effect")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def token_value(tokens: list[str], prefix: str) -> str | None:
    values = [token.split("=", 1)[1] for token in tokens if token.startswith(prefix)]
    return values[-1] if values else None


def split_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_all = [row for row in data["records"] if row["split"] == "implementation_v1_train"]
    holdout = [row for row in data["records"] if row["split"] == "implementation_v2_holdout"]
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for row in train_all:
        bucket = int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8], 16) % 6
        (dev if bucket == 0 else train).append(row)
    if not dev:
        dev.append(train.pop())
    return train, dev, holdout


def vocabulary(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(row["context_tokens"])
    ordered = [PAD, UNK] + sorted(tokens - {PAD, UNK})
    return {token: index for index, token in enumerate(ordered)}, {index: token for index, token in enumerate(ordered)}


def encode(rows: list[dict[str, Any]], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = vocab[PAD]
    contexts = [[vocab.get(token, vocab[UNK]) for token in row["context_tokens"]] for row in rows]
    max_len = max(len(item) for item in contexts)
    values = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([len(item) for item in contexts], dtype=torch.long)
    for index, item in enumerate(contexts):
        values[index, : len(item)] = torch.tensor(item, dtype=torch.long)
    actions = torch.tensor([ACTIONS.index(token_value(row["target_tokens"], "next_action=") or "abstain") for row in rows], dtype=torch.long)
    beliefs = torch.tensor([BELIEFS.index(token_value(row["target_tokens"], "final_belief=") or "oracle_gap") for row in rows], dtype=torch.long)
    return values, lengths, actions, beliefs


class QuestionCompositionPolicy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.encoder = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.action = nn.Linear(HIDDEN_DIM, len(ACTIONS))
        self.belief = nn.Linear(HIDDEN_DIM, len(BELIEFS))
        self.value = nn.Linear(HIDDEN_DIM, 1)

    def hidden(self, context: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(context), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        return self.norm(state[-1])

    def forward(self, context: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.hidden(context, lengths)
        return {"state": state, "action": self.action(state), "belief": self.belief(state), "value": self.value(state).squeeze(-1)}


def labels(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([ACTIONS.index(token_value(row["target_tokens"], "next_action=") or "abstain") for row in rows], dtype=torch.long),
        torch.tensor([BELIEFS.index(token_value(row["target_tokens"], "final_belief=") or "oracle_gap") for row in rows], dtype=torch.long),
    )


def evaluate(model: QuestionCompositionPolicy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    model.eval()
    values, lengths, actions, beliefs = encode(rows, vocab)
    with torch.inference_mode():
        output = model(values.to(device), lengths.to(device))
        action_prob = output["action"].softmax(-1).cpu()
        belief_prob = output["belief"].softmax(-1).cpu()
    pred_actions = action_prob.argmax(-1).tolist()
    pred_beliefs = belief_prob.argmax(-1).tolist()
    details: list[dict[str, Any]] = []
    for row, action, belief, action_values, belief_values in zip(rows, pred_actions, pred_beliefs, action_prob.tolist(), belief_prob.tolist()):
        expected_positive = bool(row["labels"]["expected_positive"])
        model_positive = BELIEFS[belief] == "confirmed_effect"
        details.append({"record_id": row["record_id"], "implementation": row["implementation"], "expected_action": ACTIONS[int(actions[len(details)])], "predicted_action": ACTIONS[action], "expected_belief": BELIEFS[int(beliefs[len(details)])], "predicted_belief": BELIEFS[belief], "expected_positive": expected_positive, "model_positive": model_positive, "next_action_correct": action == int(actions[len(details)]), "belief_correct": belief == int(beliefs[len(details)]), "false_positive": model_positive and not expected_positive, "false_negative": expected_positive and not model_positive, "action_confidence": round(float(max(action_values)), 6), "belief_confidence": round(float(max(belief_values)), 6)})
    total = len(details)
    positives = [row for row in details if row["expected_positive"]]
    negatives = [row for row in details if not row["expected_positive"]]
    return {"count": total, "positive_count": len(positives), "next_action_accuracy": round(sum(row["next_action_correct"] for row in details) / max(total, 1), 6), "belief_accuracy": round(sum(row["belief_correct"] for row in details) / max(total, 1), 6), "positive_recall": round(sum(row["model_positive"] for row in positives) / max(len(positives), 1), 6), "negative_reject": round(sum(not row["model_positive"] for row in negatives) / max(len(negatives), 1), 6), "false_positive_count": sum(row["false_positive"] for row in details), "false_negative_count": sum(row["false_negative"] for row in details), "details": details}


def train_sft(rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device, *, weighted: bool) -> tuple[QuestionCompositionPolicy, list[dict[str, float]]]:
    torch.manual_seed(SEED + (1 if weighted else 0))
    model = QuestionCompositionPolicy(len(vocab)).to(device)
    context, lengths, action_labels, belief_labels = encode(rows, vocab)
    weights = torch.tensor([float(row["teacher_score"]) if weighted else 1.0 for row in rows], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(1, SFT_EPOCHS + 1):
        model.train()
        output = model(context.to(device), lengths.to(device))
        action_loss = F.cross_entropy(output["action"], action_labels.to(device), reduction="none")
        belief_loss = F.cross_entropy(output["belief"], belief_labels.to(device), reduction="none")
        loss = ((action_loss + belief_loss) * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if epoch == 1 or epoch % 60 == 0:
            history.append({"epoch": float(epoch), "loss": round(value, 6)})
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def reward_for(action: torch.Tensor, belief: torch.Tensor, expected_action: torch.Tensor, expected_belief: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    action_match = action.eq(expected_action)
    belief_match = belief.eq(expected_belief)
    reward = action_match.float() * 1.5 + belief_match.float() * 1.5
    reward = reward + ((belief.eq(BELIEFS.index("oracle_gap")) & positive.eq(False)).float() * 0.75)
    reward = reward - ((belief.eq(BELIEFS.index("confirmed_effect")) & positive.eq(False)).float() * 3.0)
    reward = reward - ((action.eq(ACTIONS.index("abstain")) & positive.eq(True)).float() * 1.0)
    return reward


def train_rl(base: QuestionCompositionPolicy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> tuple[QuestionCompositionPolicy, list[dict[str, float]]]:
    torch.manual_seed(SEED + 2)
    random.seed(SEED + 2)
    model = copy.deepcopy(base).to(device)
    frozen = copy.deepcopy(base).to(device).eval()
    context, lengths, expected_actions, expected_beliefs = encode(rows, vocab)
    positive = torch.tensor([bool(row["labels"]["expected_positive"]) for row in rows], dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=0.02)
    history: list[dict[str, float]] = []
    for epoch in range(1, RL_EPOCHS + 1):
        model.train()
        output = model(context.to(device), lengths.to(device))
        action_dist = torch.distributions.Categorical(logits=output["action"])
        belief_dist = torch.distributions.Categorical(logits=output["belief"])
        sampled_action = action_dist.sample()
        sampled_belief = belief_dist.sample()
        reward = reward_for(sampled_action, sampled_belief, expected_actions.to(device), expected_beliefs.to(device), positive)
        advantage = (reward - output["value"].detach()).detach()
        policy_loss = -((action_dist.log_prob(sampled_action) + belief_dist.log_prob(sampled_belief)) * advantage).mean()
        value_loss = F.mse_loss(output["value"], reward)
        with torch.no_grad():
            frozen_output = frozen(context.to(device), lengths.to(device))
        kl = F.kl_div(F.log_softmax(output["action"], dim=-1), frozen_output["action"].softmax(-1), reduction="batchmean") + F.kl_div(F.log_softmax(output["belief"], dim=-1), frozen_output["belief"].softmax(-1), reduction="batchmean")
        anchor = F.cross_entropy(output["action"], expected_actions.to(device)) + F.cross_entropy(output["belief"], expected_beliefs.to(device))
        loss = policy_loss + 0.35 * value_loss + 0.02 * kl + 0.08 * anchor - 0.01 * (action_dist.entropy().mean() + belief_dist.entropy().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 90 == 0:
            history.append({"epoch": float(epoch), "loss": round(float(loss.detach().cpu()), 6), "mean_reward": round(float(reward.mean().detach().cpu()), 6), "kl": round(float(kl.detach().cpu()), 6)})
    return model, history


def main() -> None:
    started = time.perf_counter()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    train, dev, holdout = split_rows(data)
    vocab, reverse = vocabulary(train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0, "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    plain, plain_history = train_sft(train, vocab, device, weighted=False)
    weighted, weighted_history = train_sft(train, vocab, device, weighted=True)
    rl, rl_history = train_rl(weighted, train, vocab, device)
    evaluations = {
        "plain_sft": {"dev": evaluate(plain, dev, vocab, device), "v2_holdout": evaluate(plain, holdout, vocab, device)},
        "score_weighted_sft": {"dev": evaluate(weighted, dev, vocab, device), "v2_holdout": evaluate(weighted, holdout, vocab, device)},
        "score_rl": {"dev": evaluate(rl, dev, vocab, device), "v2_holdout": evaluate(rl, holdout, vocab, device)},
    }
    rl_holdout = evaluations["score_rl"]["v2_holdout"]
    capability_checks = {
        "implementation_disjoint": True,
        "rl_holdout_evaluated": True,
        "score_rl_positive_recall_min": float(rl_holdout["positive_recall"]) >= 0.5,
        "score_rl_negative_reject_min": float(rl_holdout["negative_reject"]) >= 0.8,
        "score_rl_false_positive_zero": int(rl_holdout["false_positive_count"]) == 0,
        "promotion_blocked": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg274-score-rl-policy-checkpoint-v1", "vocabulary": vocab, "reverse_vocabulary": reverse, "plain_state": plain.state_dict(), "score_weighted_state": weighted.state_dict(), "score_rl_state": rl.state_dict(), "seed": SEED, "device": str(device), "cuda_assignment": cuda_assignment, "actions": ACTIONS, "beliefs": BELIEFS}, CHECKPOINT)
    report = {
        "protocol_id": "pg274-score-rl-policy-v1",
        "schema_version": "pg274-score-rl-report-v1",
        "status": "completed_score_rl_ablation",
        "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "device": str(device), "cuda_assignment": cuda_assignment, "external_network": False, "raw_payload_in_context": False, "oracle_in_context": False},
        "dataset": {"train_count": len(train), "dev_count": len(dev), "v2_holdout_count": len(holdout), "v2_holdout_positive": sum(bool(row["labels"]["expected_positive"]) for row in holdout), "vocabulary_size": len(vocab)},
        "training": {"sft_epochs": SFT_EPOCHS, "rl_epochs": RL_EPOCHS, "seed": SEED, "plain_history_tail": plain_history[-5:], "score_weighted_history_tail": weighted_history[-5:], "rl_history_tail": rl_history[-5:], "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "online_target_requests": False, "memory_write": False},
        "evaluations": evaluations,
        "capability_gate": {"status": "passed" if all(capability_checks.values()) else "blocked", "checks": capability_checks, "claim_allowed": False},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "one v1→v2 implementation split; independent seed and fresh fixture replays still required"},
        "formal_conclusion": "score_rl is only useful if v2 compositional recall improves without false positives; otherwise representation/question-state data, not more RL steps, is the bottleneck.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg274-score-rl-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": data["dataset_sha256"], "implementation_train": "heterogeneous_surface_v1", "implementation_holdout": "heterogeneous_surface_v2", "evaluations": {key: {"v2_holdout": {metric: value for metric, value in val["v2_holdout"].items() if metric != "details"}} for key, val in evaluations.items()}, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg274-score-rl-policy-v1", "schema_version": "pg274-score-rl-protocol-v1", "algorithm": {"sft": "teacher-forced action+belief cross entropy", "score_weighted": "teacher episode score weighted CE", "offline_rl": "REINFORCE on recorded typed outcomes + value baseline + KL(anchor) + supervised anchor", "reward": {"action_match": 1.5, "belief_match": 1.5, "safe_abstain": 0.75, "unsupported_positive": -3.0, "missed_positive_abstain": -1.0}}, "split": {"train": "heterogeneous_surface_v1", "holdout": "heterogeneous_surface_v2", "implementation_disjoint": True}, "gates": {"positive_recall_min": 0.5, "negative_reject_min": 0.8, "false_positive_zero": True, "promotion_blocked": True}, "result": {"capability_gate": report["capability_gate"], "report_sha256": report["report_sha256"]}, "next_experiment": "PG-275 fresh v2 variant plus independent seed; if score_rl fails, add active-question and response-shape tokens before changing architecture"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rl_metrics = evaluations["score_rl"]["v2_holdout"]
    MARKDOWN.write_text("\n".join(["# PG-274 分数 + 约束离线 RL", "", f"v1 train={len(train)}；v2 holdout={len(holdout)}；device=`{device}`。", "", "| variant | v2 next-action | v2 belief | positive recall | negative reject | false positives |", "|---|---:|---:|---:|---:|---:|", *[f"| {name} | {value['v2_holdout']['next_action_accuracy']:.3f} | {value['v2_holdout']['belief_accuracy']:.3f} | {value['v2_holdout']['positive_recall']:.3f} | {value['v2_holdout']['negative_reject']:.3f} | {value['v2_holdout']['false_positive_count']} |" for name, value in evaluations.items()], "", f"gate=`{report['capability_gate']['status']}`；RL v2 positive recall={rl_metrics['positive_recall']:.3f}。只有独立实现复测和误报门都通过，才有资格继续扩大 RL。", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "device": str(device), "cuda_assignment": cuda_assignment, "v2": {name: {metric: value["v2_holdout"][metric] for metric in ("next_action_accuracy", "belief_accuracy", "positive_recall", "negative_reject", "false_positive_count")} for name, value in evaluations.items()}, "capability_gate": report["capability_gate"], "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
