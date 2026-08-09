"""PG-275: falsifiable representation and conservative-policy ablations.

All updates are offline over the abstract PG-273 records.  No raw probe,
response body, oracle field, or online target request enters the model.
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
OUT = ROOT / "artifacts" / "pg275-hypothesis-ablation"
REPORT = ROOT / "research" / "pg275_hypothesis_ablation_report_v1.json"
TRACE = ROOT / "research" / "pg275_hypothesis_ablation_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg275_hypothesis_ablation_protocol_v1.json"

PAD = "[PAD]"
UNK = "[UNK]"
ACTIONS = ("abstain", "replay_confirmed", "diagnose_failure", "candidate_probe")
BELIEFS = ("oracle_gap", "confirmed_effect")
SEED = 27501
EMBED = 64
HIDDEN = 128


class Policy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED)
        self.encoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.action = nn.Linear(HIDDEN, len(ACTIONS))
        self.belief = nn.Linear(HIDDEN, len(BELIEFS))

    def forward(self, context: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = self.embedding(context)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        state = self.norm(state[-1])
        return {"state": state, "action": self.action(state), "belief": self.belief(state)}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def target_labels(row: dict[str, Any]) -> tuple[int, int]:
    action = next((t.split("=", 1)[1] for t in row["target_tokens"] if t.startswith("next_action=")), "abstain")
    belief = next((t.split("=", 1)[1] for t in row["target_tokens"] if t.startswith("final_belief=")), "oracle_gap")
    # Preference negatives can contain richer trajectory labels than the
    # compact two-head policy vocabulary.  Project those labels explicitly;
    # never silently use them as a new positive class.
    if action == "confirm_without_replay":
        action = "abstain"
    if action not in ACTIONS:
        action = "candidate_probe"
    if belief == "unsupported_positive":
        belief = "confirmed_effect"
    if belief not in BELIEFS:
        belief = "oracle_gap"
    return ACTIONS.index(action), BELIEFS.index(belief)


def transformed_context(row: dict[str, Any], mode: str) -> list[str]:
    tokens = list(row["context_tokens"])
    if mode == "atomic":
        return tokens
    if mode == "minimal":
        keep_exact = {"[BOS]", "[CTX_END]"}
        keep_prefix = ("question=", "method=", "placement=", "field_bucket=", "encoding=", "observe_content=", "observe_status=")
        return [t for t in tokens if t in keep_exact or any(t.startswith(prefix) for prefix in keep_prefix)]
    if mode == "collapsed":
        shape = [t for t in tokens if t.startswith("observe_")]
        non_shape = [t for t in tokens if not t.startswith("observe_")]
        collapsed = "shape=" + "|".join(shape)
        return [t for t in non_shape if t not in ("[CTX_END]",)] + [collapsed, "[CTX_END]"]
    raise ValueError(mode)


def build_vocab(rows: list[dict[str, Any]], mode: str) -> dict[str, int]:
    values = {PAD, UNK}
    for row in rows:
        values.update(transformed_context(row, mode))
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(values - {PAD, UNK}))}


def batch(rows: list[dict[str, Any]], vocab: dict[str, int], mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [[vocab.get(t, vocab[UNK]) for t in transformed_context(row, mode)] for row in rows]
    values = torch.full((len(rows), max(map(len, sequences))), vocab[PAD], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    for index, seq in enumerate(sequences):
        values[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    labels = [target_labels(row) for row in rows]
    actions = torch.tensor([x[0] for x in labels], dtype=torch.long)
    beliefs = torch.tensor([x[1] for x in labels], dtype=torch.long)
    positive = torch.tensor([bool(row["labels"]["expected_positive"]) for row in rows], dtype=torch.bool)
    return values, lengths, actions, beliefs, positive


def split_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_train = [row for row in data["records"] if row["split"] == "implementation_v1_train"]
    holdout = [row for row in data["records"] if row["split"] == "implementation_v2_holdout"]
    train, dev = [], []
    for row in all_train:
        if int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8], 16) % 6 == 0:
            dev.append(row)
        else:
            train.append(row)
    return train, dev or train[-1:], holdout


def fit_weighted(rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device, *, weighted: bool, seed: int) -> Policy:
    torch.manual_seed(seed)
    model = Policy(len(vocab)).to(device)
    values, lengths, actions, beliefs, _ = batch(rows, vocab, mode)
    weights = torch.tensor([float(row["teacher_score"]) if weighted else 1.0 for row in rows], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01)
    best, best_loss = None, float("inf")
    for _ in range(180):
        model.train()
        out = model(values.to(device), lengths.to(device))
        loss = (F.cross_entropy(out["action"], actions.to(device), reduction="none") + F.cross_entropy(out["belief"], beliefs.to(device), reduction="none"))
        loss = (loss * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach())
            best = copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)
    return model


def reward_table(actions: torch.Tensor, beliefs: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    table = torch.zeros((len(actions), len(ACTIONS), len(BELIEFS)), device=actions.device)
    for a in range(len(ACTIONS)):
        for b in range(len(BELIEFS)):
            value = (a == actions).float() * 1.5 + (b == beliefs).float() * 1.5
            value = value + ((b == BELIEFS.index("oracle_gap")) & (~positive)).float() * 0.75
            value = value - ((b == BELIEFS.index("confirmed_effect")) & (~positive)).float() * 3.0
            value = value - ((a == ACTIONS.index("abstain")) & positive).float() * 1.0
            table[:, a, b] = value
    return table


def conservative_update(base: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device) -> Policy:
    """Low-variance expected-reward update anchored to the behavior policy."""
    model = copy.deepcopy(base).to(device)
    frozen = copy.deepcopy(base).to(device).eval()
    values, lengths, actions, beliefs, positive = batch(rows, vocab, mode)
    values, lengths = values.to(device), lengths.to(device)
    actions, beliefs, positive = actions.to(device), beliefs.to(device), positive.to(device)
    table = reward_table(actions, beliefs, positive)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00045, weight_decay=0.01)
    for _ in range(90):
        model.train()
        out = model(values, lengths)
        action_prob, belief_prob = out["action"].softmax(-1), out["belief"].softmax(-1)
        expected_reward = (action_prob[:, :, None] * belief_prob[:, None, :] * table).sum(dim=(1, 2)).mean()
        anchor = F.cross_entropy(out["action"], actions) + F.cross_entropy(out["belief"], beliefs)
        with torch.no_grad():
            ref = frozen(values, lengths)
        kl = F.kl_div(F.log_softmax(out["action"], -1), ref["action"].softmax(-1), reduction="batchmean") + F.kl_div(F.log_softmax(out["belief"], -1), ref["belief"].softmax(-1), reduction="batchmean")
        loss = -expected_reward + 0.30 * anchor + 2.0 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def dpo_update(base: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device) -> Policy:
    """Preference update with a frozen weighted-SFT reference and anchor."""
    model = copy.deepcopy(base).to(device)
    frozen = copy.deepcopy(base).to(device).eval()
    values, lengths, actions, beliefs, _ = batch(rows, vocab, mode)
    values, lengths, actions, beliefs = values.to(device), lengths.to(device), actions.to(device), beliefs.to(device)
    rejected = [target_labels({"target_tokens": row["preference"]["rejected_target_tokens"]}) for row in rows]
    reject_actions = torch.tensor([x[0] for x in rejected], device=device)
    reject_beliefs = torch.tensor([x[1] for x in rejected], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00035, weight_decay=0.01)
    for _ in range(90):
        model.train()
        out = model(values, lengths)
        with torch.no_grad():
            ref = frozen(values, lengths)
        lp_a, lp_b = F.log_softmax(out["action"], -1), F.log_softmax(out["belief"], -1)
        chosen = lp_a.gather(1, actions[:, None]).squeeze(1) + lp_b.gather(1, beliefs[:, None]).squeeze(1)
        rejected_lp = lp_a.gather(1, reject_actions[:, None]).squeeze(1) + lp_b.gather(1, reject_beliefs[:, None]).squeeze(1)
        preference_loss = -F.logsigmoid(0.10 * (chosen - rejected_lp)).mean()
        anchor = F.cross_entropy(out["action"], actions) + F.cross_entropy(out["belief"], beliefs)
        kl = F.kl_div(F.log_softmax(out["action"], -1), ref["action"].softmax(-1), reduction="batchmean") + F.kl_div(F.log_softmax(out["belief"], -1), ref["belief"].softmax(-1), reduction="batchmean")
        loss = preference_loss + 0.20 * anchor + 1.5 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model


def evaluate(model: Policy, rows: list[dict[str, Any]], vocab: dict[str, int], mode: str, device: torch.device) -> dict[str, Any]:
    model.eval()
    values, lengths, actions, beliefs, positive = batch(rows, vocab, mode)
    with torch.inference_mode():
        out = model(values.to(device), lengths.to(device))
        pred_a, pred_b = out["action"].argmax(-1).cpu(), out["belief"].argmax(-1).cpu()
    details = []
    for index, row in enumerate(rows):
        model_positive = int(pred_b[index]) == BELIEFS.index("confirmed_effect")
        details.append({"record_id": row["record_id"], "expected_positive": bool(positive[index]), "predicted_action": ACTIONS[int(pred_a[index])], "expected_action": ACTIONS[int(actions[index])], "predicted_belief": BELIEFS[int(pred_b[index])], "expected_belief": BELIEFS[int(beliefs[index])], "model_positive": model_positive, "action_correct": int(pred_a[index]) == int(actions[index]), "belief_correct": int(pred_b[index]) == int(beliefs[index]), "false_positive": model_positive and not bool(positive[index]), "false_negative": bool(positive[index]) and not model_positive})
    pos, neg = [x for x in details if x["expected_positive"]], [x for x in details if not x["expected_positive"]]
    return {"count": len(details), "positive_count": len(pos), "next_action_accuracy": round(sum(x["action_correct"] for x in details) / max(len(details), 1), 6), "belief_accuracy": round(sum(x["belief_correct"] for x in details) / max(len(details), 1), 6), "positive_recall": round(sum(x["model_positive"] for x in pos) / max(len(pos), 1), 6), "negative_reject": round(sum(not x["model_positive"] for x in neg) / max(len(neg), 1), 6), "false_positive_count": sum(x["false_positive"] for x in details), "false_negative_count": sum(x["false_negative"] for x in details), "details": details}


def main() -> None:
    started = time.perf_counter()
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    train, dev, holdout = split_rows(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    evaluations: dict[str, Any] = {}
    models: dict[str, tuple[Policy, dict[str, int], str]] = {}
    for mode in ("minimal", "atomic", "collapsed"):
        vocab = build_vocab(train, mode)
        model = fit_weighted(train, vocab, mode, device, weighted=True, seed=SEED + len(evaluations))
        name = f"weighted_sft_{mode}"
        evaluations[name] = {"mode": mode, "dev": evaluate(model, dev, vocab, mode, device), "v2_holdout": evaluate(model, holdout, vocab, mode, device)}
        models[mode] = (model, vocab, mode)
    atomic_model, atomic_vocab, _ = models["atomic"]
    conservative = conservative_update(atomic_model, train, atomic_vocab, "atomic", device)
    evaluations["conservative_offline_update"] = {"mode": "atomic", "dev": evaluate(conservative, dev, atomic_vocab, "atomic", device), "v2_holdout": evaluate(conservative, holdout, atomic_vocab, "atomic", device)}
    dpo = dpo_update(atomic_model, train, atomic_vocab, "atomic", device)
    evaluations["dpo_preference_update"] = {"mode": "atomic", "dev": evaluate(dpo, dev, atomic_vocab, "atomic", device), "v2_holdout": evaluate(dpo, holdout, atomic_vocab, "atomic", device)}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg275-hypothesis-ablation-checkpoint-v1", "atomic_weighted_state": atomic_model.state_dict(), "conservative_state": conservative.state_dict(), "dpo_state": dpo.state_dict(), "seed": SEED, "assignment": assignment}, OUT / "pg275_policies.pt")
    report = {"protocol_id": "pg275-hypothesis-ablation-v1", "schema_version": "pg275-hypothesis-ablation-report-v1", "status": "completed_hypothesis_ablation", "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "device": str(device), "cuda_assignment": assignment, "external_network": False, "raw_payload_in_context": False, "oracle_in_context": False}, "split": {"train": len(train), "dev": len(dev), "v2_holdout": len(holdout), "implementation_train": "heterogeneous_surface_v1", "implementation_holdout": "heterogeneous_surface_v2", "disjoint": True}, "evaluations": evaluations, "policy_contract": {"conservative_rl": {"online_requests": False, "entropy_bonus": False, "behavior_kl_anchor": True, "positive_miss_penalty": True, "unsupported_positive_penalty": True}, "dpo": {"online_requests": False, "frozen_reference": True, "preference_pairs": True}}, "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "claim_allowed": False, "reason": "third implementation and fresh-seed re-test still required"}, "interpretation": "Compare representation modes before policy optimization; a conservative update is only useful if it preserves weighted-SFT holdout recall and lowers no guardrail metric.", "elapsed_seconds": round(time.perf_counter() - started, 3)}
    report["report_sha256"] = sha(report)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg275-hypothesis-ablation-trace-v1", "evaluation_only": True, "training_eligible": False, "source_dataset_sha256": data["dataset_sha256"], "evaluations": {name: {"v2_holdout": {k: v for k, v in item["v2_holdout"].items() if k != "details"}} for name, item in evaluations.items()}, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "oracle_in_context": False, "memory_write": False}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg275-hypothesis-ablation-v1", "schema_version": "pg275-hypothesis-ablation-protocol-v1", "hypotheses": ["representation_bottleneck", "supervision_objective", "reward_misalignment", "composition_leakage", "memory_contamination"], "modes": ["minimal", "atomic", "collapsed"], "policy_variants": ["weighted_sft", "conservative_offline_update", "dpo_preference_update"], "gates": {"positive_recall_min": 0.5, "negative_reject_min": 0.8, "false_positive_zero": True, "promotion_blocked": True}, "report_sha256": report["report_sha256"], "next_experiment": "PG-276 third implementation plus fresh seed and old-canary replay"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": assignment, "v2": {name: {k: item["v2_holdout"][k] for k in ("next_action_accuracy", "belief_accuracy", "positive_recall", "negative_reject", "false_positive_count", "false_negative_count")} for name, item in evaluations.items()}, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
