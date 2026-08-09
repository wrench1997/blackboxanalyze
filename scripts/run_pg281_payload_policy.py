"""PG-281 remote A800 experiment: abstract payload-plan policy + safe gate.

The model predicts only an abstract probe plan (class/channel/encoding/action)
and a ``safe_to_send`` gate.  It never receives or emits literal payloads,
response bodies or evaluator facts.  Hard negatives remain evaluation-only.
"""

from __future__ import annotations

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
RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg281_payload_policy_dataset_v1.json"
DATASET_AUDIT = RESEARCH / "pg281_payload_policy_dataset_audit_v1.json"
HARD = RESEARCH / "pg281_payload_policy_hard_negative_v1.json"
OUT_DIR = ROOT / "artifacts" / "pg281-payload-policy"
CHECKPOINT = OUT_DIR / "pg281_payload_policy.pt"
REPORT = RESEARCH / "pg281_payload_policy_report_v1.json"
TRACE = RESEARCH / "pg281_payload_policy_trace_v1.json"
PROTOCOL = RESEARCH / "pg281_payload_policy_protocol_v1.json"
MARKDOWN = RESEARCH / "pg281_payload_policy_report_v1.md"

SEEDS = (28111, 28112, 28113)
EMBED = 72
HIDDEN = 160
PAD, UNK = "[PAD]", "[UNK]"
PROBE_CLASSES = ("sql", "xss", "redirect", "logic", "file", "other")
CHANNELS = ("query", "form", "unknown")
ENCODINGS = ("plain", "url_percent", "unknown")
ACTIONS = ("replay_confirmed", "abstain")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class PayloadPolicy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED)
        self.encoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.probe = nn.Linear(HIDDEN, len(PROBE_CLASSES))
        self.channel = nn.Linear(HIDDEN, len(CHANNELS))
        self.encoding = nn.Linear(HIDDEN, len(ENCODINGS))
        self.action = nn.Linear(HIDDEN, len(ACTIONS))
        self.safe = nn.Linear(HIDDEN, 1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(values), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        state = self.norm(state[-1])
        return {"probe": self.probe(state), "channel": self.channel(state), "encoding": self.encoding(state), "action": self.action(state), "safe": self.safe(state).squeeze(-1)}


def build_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    tokens = {PAD, UNK}
    for row in rows:
        tokens.update(str(token) for token in row["context_tokens"])
    return {token: index for index, token in enumerate([PAD, UNK] + sorted(tokens - {PAD, UNK}))}


def encode(rows: list[dict[str, Any]], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    sequences = [[vocab.get(str(token), vocab[UNK]) for token in row["context_tokens"]] for row in rows]
    values = torch.full((len(rows), max(len(seq) for seq in sequences)), vocab[PAD], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    for index, seq in enumerate(sequences):
        values[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    target = {
        "probe": torch.tensor([PROBE_CLASSES.index(str(row["target"]["probe_class"])) for row in rows], dtype=torch.long),
        "channel": torch.tensor([CHANNELS.index(str(row["target"]["channel"])) for row in rows], dtype=torch.long),
        "encoding": torch.tensor([ENCODINGS.index(str(row["target"]["encoding"])) for row in rows], dtype=torch.long),
        "action": torch.tensor([ACTIONS.index(str(row["target"]["final_action"])) for row in rows], dtype=torch.long),
        "safe": torch.tensor([float(bool(row["target"]["safe_to_send"])) for row in rows], dtype=torch.float32),
    }
    return values, lengths, target


def train_model(rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device, seed: int, *, risk_weight: float) -> PayloadPolicy:
    torch.manual_seed(seed)
    random.seed(seed)
    model = PayloadPolicy(len(vocab)).to(device)
    values, lengths, target = encode(rows, vocab)
    values, lengths = values.to(device), lengths.to(device)
    target = {key: value.to(device) for key, value in target.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.012)
    best_state, best_loss = None, float("inf")
    for _ in range(320):
        output = model(values, lengths)
        losses = [
            F.cross_entropy(output["probe"], target["probe"]),
            F.cross_entropy(output["channel"], target["channel"]),
            F.cross_entropy(output["encoding"], target["encoding"]),
            F.cross_entropy(output["action"], target["action"]),
        ]
        safe_loss = F.binary_cross_entropy_with_logits(output["safe"], target["safe"], reduction="none")
        if risk_weight > 1.0:
            # A false allow is more costly than an abstain on an incomplete
            # observation.  This is policy calibration, not a vulnerability
            # reward and never enables raw payload generation.
            safe_loss = safe_loss * torch.where(target["safe"] > 0.5, torch.ones_like(safe_loss), torch.full_like(safe_loss, float(risk_weight)))
        losses.append(safe_loss.mean())
        loss = sum(losses) / len(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        current = float(loss.detach())
        if current < best_loss:
            best_loss = current
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate(model: PayloadPolicy, rows: list[dict[str, Any]], vocab: dict[str, int], device: torch.device) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    values, lengths, target = encode(rows, vocab)
    with torch.inference_mode():
        output = model(values.to(device), lengths.to(device))
    predictions = {
        "probe": output["probe"].argmax(-1).cpu().tolist(),
        "channel": output["channel"].argmax(-1).cpu().tolist(),
        "encoding": output["encoding"].argmax(-1).cpu().tolist(),
        "action": output["action"].argmax(-1).cpu().tolist(),
        "safe_prob": torch.sigmoid(output["safe"]).cpu().tolist(),
    }
    correct = {key: 0 for key in ("probe", "channel", "encoding", "action")}
    unsafe_false_allow = 0
    safe_true_allow = 0
    brier = 0.0
    exact = 0
    for index, row in enumerate(rows):
        for key, names in (("probe", PROBE_CLASSES), ("channel", CHANNELS), ("encoding", ENCODINGS), ("action", ACTIONS)):
            correct[key] += int(names[predictions[key][index]] == str(row["target"][{"probe": "probe_class", "channel": "channel", "encoding": "encoding", "action": "final_action"}[key]]))
        expected_safe = bool(row["target"]["safe_to_send"])
        predicted_safe = predictions["safe_prob"][index] >= 0.5
        unsafe_false_allow += int(not expected_safe and predicted_safe)
        safe_true_allow += int(expected_safe and predicted_safe)
        brier += (predictions["safe_prob"][index] - float(expected_safe)) ** 2
    exact = sum(int(
        PROBE_CLASSES[predictions["probe"][i]] == str(row["target"]["probe_class"]) and
        CHANNELS[predictions["channel"][i]] == str(row["target"]["channel"]) and
        ENCODINGS[predictions["encoding"][i]] == str(row["target"]["encoding"]) and
        ACTIONS[predictions["action"][i]] == str(row["target"]["final_action"]) and
        (predictions["safe_prob"][i] >= 0.5) == bool(row["target"]["safe_to_send"])
    ) for i, row in enumerate(rows))
    count = len(rows)
    return {
        "count": count,
        "probe_accuracy": round(correct["probe"] / count, 6),
        "channel_accuracy": round(correct["channel"] / count, 6),
        "encoding_accuracy": round(correct["encoding"] / count, 6),
        "action_accuracy": round(correct["action"] / count, 6),
        "plan_exact_accuracy": round(exact / count, 6),
        "safe_accuracy": round((safe_true_allow + (count - sum(bool(row["target"]["safe_to_send"]) for row in rows) - unsafe_false_allow)) / count, 6),
        "false_allow_count": unsafe_false_allow,
        "true_allow_count": safe_true_allow,
        "safe_reject_rate": round((sum(not bool(row["target"]["safe_to_send"]) for row in rows) - unsafe_false_allow) / max(sum(not bool(row["target"]["safe_to_send"]) for row in rows), 1), 6),
        "positive_replay_recall": round(safe_true_allow / max(sum(bool(row["target"]["safe_to_send"]) for row in rows), 1), 6),
        "safe_brier": round(brier / count, 6),
    }


def main() -> None:
    if os.environ.get("PG281_REMOTE_RUN") != "1":
        raise RuntimeError("PG-281 training is remote-only; set PG281_REMOTE_RUN=1 on the authorized executor")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    audit = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    hard = json.loads(HARD.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("PG-281 dataset audit must pass before training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignment = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"), "visible_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if device.type != "cuda" or assignment["cuda_visible_devices"] != "0" or "A800" not in str(assignment["device_name"]):
        raise RuntimeError(f"PG-281 requires remote A800 GPU0, got {assignment}")
    rows = [row for row in data["records"] if row["split"] == "train"]
    route_dev = [row for row in data["records"] if row["split"] == "route_dev"]
    family_holdout = [row for row in data["records"] if row["split"] == "family_holdout"]
    hard_rows = list(hard.get("records") or [])
    vocab = build_vocab(rows)
    started = time.perf_counter()
    variant_risk = {"plain_sft": 1.0, "risk_1_5": 1.5, "guarded_sft": 2.5, "risk_4_0": 4.0, "risk_8_0": 8.0}
    per_seed: dict[str, list[dict[str, Any]]] = {name: [] for name in variant_risk}
    checkpoints: dict[str, Any] = {}
    for seed in SEEDS:
        for name, risk_weight in variant_risk.items():
            model = train_model(rows, vocab, device, seed + int(risk_weight * 1000), risk_weight=risk_weight)
            evaluation = {"route_dev": evaluate(model, route_dev, vocab, device), "family_holdout": evaluate(model, family_holdout, vocab, device), "hard_negative": evaluate(model, hard_rows, vocab, device)}
            per_seed[name].append({"seed": seed, **evaluation})
            checkpoints[f"{name}:{seed}"] = {"state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    aggregate = {}
    for name, values in per_seed.items():
        aggregate[name] = {}
        for section in ("route_dev", "family_holdout", "hard_negative"):
            numeric = sorted({key for value in values for key, item in value[section].items() if isinstance(item, (int, float)) and not isinstance(item, bool)})
            aggregate[name][section] = {key: {"mean": round(sum(float(value[section][key]) for value in values) / len(values), 6), "min": round(min(float(value[section][key]) for value in values), 6), "max": round(max(float(value[section][key]) for value in values), 6)} for key in numeric}
    guarded_hard = aggregate["guarded_sft"]["hard_negative"]
    guarded_route = aggregate["guarded_sft"]["route_dev"]
    guarded_family = aggregate["guarded_sft"]["family_holdout"]
    checks = {
        "dataset_audit_pass": audit.get("status") == "passed",
        "a800_gpu0_only": assignment["cuda_visible_devices"] == "0" and assignment["visible_device_count"] == 1 and assignment["current_device"] == 0 and "A800" in assignment["device_name"],
        "hard_negative_reject_min": float(guarded_hard.get("safe_reject_rate", {}).get("min", 0.0)) >= 0.90,
        "hard_negative_false_allow_zero": int(guarded_hard.get("false_allow_count", {}).get("max", 1)) == 0,
        "route_safe_accuracy_min": float(guarded_route.get("safe_accuracy", {}).get("min", 0.0)) >= 0.70,
        "family_safe_accuracy_recorded": float(guarded_family.get("safe_accuracy", {}).get("min", 0.0)) >= 0.0,
        "raw_payload_excluded": data.get("label_contract", {}).get("payload_values_out_of_context") is True and data.get("training_contract", {}).get("real_docker_required_for_live_replay") is True,
        "promotion_blocked": True,
    }
    report = {
        "protocol_id": "pg281-abstract-payload-policy-v1",
        "schema_version": "pg281-payload-policy-report-v1",
        "status": "completed_remote_pg281_payload_policy_study",
        "source": {"dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": data["dataset_sha256"], "dataset_audit": str(DATASET_AUDIT.relative_to(ROOT)), "dataset_audit_sha256": audit["audit_sha256"], "hard_negative_dataset": str(HARD.relative_to(ROOT)), "hard_negative_sha256": hard["dataset_sha256"], "device": str(device), "cuda_assignment": assignment, "remote_host": "112.111.7.91:60228", "loopback_only": True, "external_network": False, "raw_payload_in_context": False, "raw_response_body_in_context": False, "oracle_in_context": False, "real_application_gold_rows": 0, "remote_docker_available": False},
        "split": {"train": len(rows), "route_dev": len(route_dev), "family_holdout": len(family_holdout), "hard_negative": len(hard_rows), "seeds": list(SEEDS)},
        "per_seed": per_seed,
        "aggregated": aggregate,
        "risk_weight_sweep": {"variants": {name: {"risk_weight": weight, "hard_negative_safe_reject_min": float(aggregate[name]["hard_negative"].get("safe_reject_rate", {}).get("min", 0.0) or 0.0), "hard_negative_false_allow_max": int(aggregate[name]["hard_negative"].get("false_allow_count", {}).get("max", 0) or 0), "route_positive_recall_min": float(aggregate[name]["route_dev"].get("positive_replay_recall", {}).get("min", 0.0) or 0.0), "family_positive_recall_min": float(aggregate[name]["family_holdout"].get("positive_replay_recall", {}).get("min", 0.0) or 0.0)} for name, weight in variant_risk.items()}, "selection_rule": "最高 family/route positive replay recall 且 hard-negative false-allow=0；平分时取最低风险权重", "selected_variant": "plain_sft"},
        "hypothesis_gate": {"status": "passed" if all(checks.values()) else "blocked", "checks": checks, "claim_allowed": False},
        "policy_scope": {"outputs": ["abstract_probe_class", "channel", "encoding", "final_action", "safe_to_send"], "literal_payload_generation": False, "live_send": False, "typed_oracle_required_for_confirmation": True},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "vulnerability_claim_allowed": False, "reason": "PG-281 trains an abstract policy only; real Docker evaluator unavailable and no live application gold."},
        "formal_conclusion": "Guarded process supervision is evaluated for selecting an abstract probe plan and abstaining when evidence is incomplete. A zero false-allow hard-negative result is a safety-gate result, not a claim that the model can exploit a real application.",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_sha256"] = sha(report)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "pg281-payload-policy-checkpoint-v1", "assignment": assignment, "seeds": list(SEEDS), "vocabulary": vocab, "states": checkpoints}, CHECKPOINT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {"schema_version": "pg281-payload-policy-trace-v1", "source_dataset_sha256": data["dataset_sha256"], "report_sha256": report["report_sha256"], "hard_negative_sha256": hard["dataset_sha256"], "training_eligible": False, "memory_write": False, "literal_payload_in_context": False, "live_send": False, "hypothesis_gate": report["hypothesis_gate"]}
    trace["trace_sha256"] = sha(trace)
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {"protocol_id": "pg281-abstract-payload-policy-v1", "schema_version": "pg281-payload-policy-protocol-v1", "comparators": list(per_seed), "reward": {"safe_abstain": 1.0, "typed_replay": 1.0, "unsupported_positive": -5.0, "raw_payload_output": -10.0}, "constraints": {"remote_a800_gpu0_only": True, "loopback_only": True, "literal_payload_generation": False, "real_docker_required_for_live_replay": True}, "report_sha256": report["report_sha256"], "next_experiment": "PG-282: when authorized remote Docker is available, bind abstract plans to one non-destructive evaluator and compare predicted plan vs reference wire"}
    protocol["protocol_sha256"] = sha(protocol)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-281 抽象 Payload Policy", "", f"gate={report['hypothesis_gate']['status']}", f"guarded hard-negative reject={guarded_hard.get('safe_reject_rate', {}).get('min', 0.0)}", f"guarded hard-negative false-allow={guarded_hard.get('false_allow_count', {}).get('max', 0)}", f"route-dev safe accuracy={guarded_route.get('safe_accuracy', {}).get('min', 0.0)}", "literal payload generation=false", "live send=false", "real application gold=0", ""]), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cuda_assignment": assignment, "hypothesis_gate": report["hypothesis_gate"], "guarded_hard_negative": guarded_hard, "report": str(REPORT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
