"""PG-57 typed-oracle post-training with a separate Rule IR head.

Only the prefix through ``ORACLE_TARGET`` is visible to the head.  The typed
oracle provides supervision after the probe, never an input feature.  Family
names are evaluator metadata and are not predicted by this experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.causal_trace_transformer import CausalTraceTransformer  # noqa: E402


DATASET_PATH = ROOT / "research" / "pg56_causal_trace_dataset_v1.json"
PRETRAIN_REPORT_PATH = ROOT / "research" / "pg56_causal_trace_pretraining_report_v1.json"
REPORT_PATH = ROOT / "research" / "pg57_rule_ir_posttraining_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg57_rule_ir_posttraining_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg57-rule-ir-posttrainer"
CHECKPOINT_PATH = OUTPUT_DIR / "rule_ir_head.pt"
SEED = 20260803
MAX_LEN = 128
EPOCHS = 80

EFFECTS = ["abstain", "confirmed", "rejected"]
TRANSPORTS = ["GET", "POST"]
MODALITIES = ["DOM", "AST", "BOUNDARY", "REDIRECT", "CANARY", "EFFECT", "NEGATIVE_CONTROL", "OTHER"]


class RuleIRHead(nn.Module):
    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.effect = nn.Linear(d_model, len(EFFECTS))
        self.transport = nn.Linear(d_model, len(TRANSPORTS))
        self.modality = nn.Linear(d_model, len(MODALITIES))
        self.confidence = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.norm(hidden)
        return {
            "effect": self.effect(hidden),
            "transport": self.transport(hidden),
            "modality": self.modality(hidden),
            "confidence": self.confidence(hidden).squeeze(-1),
        }


def _load() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    pretrain_report = json.loads(PRETRAIN_REPORT_PATH.read_text(encoding="utf-8"))
    rows = dataset["rows"]
    return dataset, pretrain_report, [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "holdout"]


def _encode_prefix(row: dict[str, Any], vocabulary: dict[str, int]) -> tuple[torch.Tensor, int]:
    ids = [vocabulary.get(token, vocabulary["<UNK>"]) for token in row["tokens"][:MAX_LEN]]
    marker = row["tokens"].index("ORACLE_TARGET")
    marker = min(marker, len(ids) - 1)
    # The marker itself is visible; oracle modality/outcome/rule slots are not.
    prefix = torch.tensor([ids[:marker + 1]], dtype=torch.long)
    return prefix, marker


def _target(row: dict[str, Any]) -> tuple[int, int, int, float]:
    target = row["target"]
    outcome = target["outcome"]
    # A positive typed effect is only confirmable when evidence is present;
    # negative or absent evidence is rejected/abstained, never confirmed.
    effect = "confirmed" if outcome == "positive" and target.get("evidence_present") else ("rejected" if outcome == "negative" else "abstain")
    transport = "GET" if "CHANNEL_GET" in row["tokens"] else "POST"
    modality = str(target.get("modality") or "OTHER")
    if modality not in MODALITIES:
        modality = "OTHER"
    return EFFECTS.index(effect), TRANSPORTS.index(transport), MODALITIES.index(modality), int(effect == "confirmed")


def _hidden(model: CausalTraceTransformer, row: dict[str, Any], vocabulary: dict[str, int], device: torch.device) -> torch.Tensor:
    prefix, _ = _encode_prefix(row, vocabulary)
    prefix = prefix.to(device)
    mask = torch.ones_like(prefix, dtype=torch.bool)
    return model.encode(prefix, mask)[:, -1, :]


def _hidden_rows(model: CausalTraceTransformer, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> list[torch.Tensor]:
    model.eval()
    # Use no_grad (rather than inference_mode) so the cached tensors remain
    # valid autograd inputs for the trainable post-training head.
    with torch.no_grad():
        return [_hidden(model, row, vocabulary, device).detach().clone() for row in rows]


def _confirmation_scores(base: CausalTraceTransformer, head: RuleIRHead, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device, *, vectors: list[torch.Tensor] | None = None) -> list[tuple[int, float]]:
    base.eval()
    head.eval()
    scores: list[tuple[int, float]] = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            hidden = vectors[index] if vectors is not None else _hidden(base, row, vocabulary, device)
            output = head(hidden)
            probabilities = output["effect"].softmax(-1)[0]
            scores.append((int(probabilities.argmax()), float(probabilities[EFFECTS.index("confirmed")].cpu())))
    return scores


def _calibrate_confirmation_threshold(rows: list[dict[str, Any]], scores: list[tuple[int, float]]) -> tuple[float, dict[str, Any]]:
    candidates = sorted({score for _, score in scores} | {1.0}, reverse=True)
    best_threshold = 1.0
    best: dict[str, Any] | None = None
    confirmed_index = EFFECTS.index("confirmed")
    for threshold in candidates:
        accepted = [index for index, (predicted, score) in enumerate(scores) if predicted == confirmed_index and score >= threshold]
        unknown_attempts = sum(int(index in accepted and rows[index]["target"].get("unknown_family")) for index in range(len(rows)))
        negative_false_accepts = sum(int(index in accepted and _target(rows[index])[0] != confirmed_index) for index in range(len(rows)))
        true_positives = sum(int(index in accepted and _target(rows[index])[0] == confirmed_index) for index in range(len(rows)))
        metrics = {
            "threshold": threshold,
            "emitted_confirmed_count": len(accepted),
            "confirmed_true_positive_count": true_positives,
            "unknown_family_confirmed_attempts": unknown_attempts,
            "negative_confirmed_false_accept_count": negative_false_accepts,
        }
        if unknown_attempts or negative_false_accepts:
            continue
        if best is None or true_positives > best["confirmed_true_positive_count"]:
            best = metrics
            best_threshold = threshold
    return best_threshold, best or {
        "threshold": best_threshold,
        "emitted_confirmed_count": 0,
        "confirmed_true_positive_count": 0,
        "unknown_family_confirmed_attempts": 0,
        "negative_confirmed_false_accept_count": 0,
    }


def _evaluate(base: CausalTraceTransformer, head: RuleIRHead, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device, *, vectors: list[torch.Tensor] | None = None, threshold: float = 0.5) -> dict[str, Any]:
    base.eval()
    head.eval()
    counts = {"effect": 0, "transport": 0, "modality": 0, "confirmed": 0}
    totals = {key: 0 for key in counts}
    accepted: list[int] = []
    unknown_attempts = 0
    with torch.inference_mode():
        for index, row in enumerate(rows):
            hidden = vectors[index] if vectors is not None else _hidden(base, row, vocabulary, device)
            output = head(hidden)
            target_effect, target_transport, target_modality, _ = _target(row)
            effect_prob = output["effect"].softmax(-1)[0]
            predicted_effect = int(effect_prob.argmax())
            predicted_transport = int(output["transport"].argmax(-1)[0])
            predicted_modality = int(output["modality"].argmax(-1)[0])
            totals["effect"] += 1
            totals["transport"] += 1
            totals["modality"] += 1
            counts["effect"] += int(predicted_effect == target_effect)
            counts["transport"] += int(predicted_transport == target_transport)
            counts["modality"] += int(predicted_modality == target_modality)
            emit_confirmed = predicted_effect == EFFECTS.index("confirmed") and float(effect_prob[predicted_effect]) >= threshold
            if emit_confirmed:
                accepted.append(index)
                counts["confirmed"] += int(target_effect == EFFECTS.index("confirmed"))
            if row["target"].get("unknown_family") and emit_confirmed:
                unknown_attempts += 1
    metrics = {
        "count": len(rows),
        "effect_accuracy": round(counts["effect"] / max(totals["effect"], 1), 6),
        "transport_accuracy": round(counts["transport"] / max(totals["transport"], 1), 6),
        "modality_accuracy": round(counts["modality"] / max(totals["modality"], 1), 6),
        "emitted_confirmed_count": len(accepted),
        "confirmed_true_positive_count": counts["confirmed"],
        "unknown_family_confirmed_attempts": unknown_attempts,
        "unknown_family_strict_abstain": unknown_attempts == 0,
        "negative_confirmed_false_accept_count": sum(int(index in accepted and _target(rows[index])[0] != EFFECTS.index("confirmed")) for index in range(len(rows))),
        "abstain_or_reject_rate": round(1.0 - len(accepted) / max(len(rows), 1), 6),
    }
    metrics["confirmed_recall"] = round(counts["confirmed"] / max(sum(_target(row)[0] == EFFECTS.index("confirmed") for row in rows), 1), 6)
    return metrics


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dataset, pretrain_report, train_rows, dev_rows, holdout_rows = _load()
    vocabulary = torch.load(ROOT / pretrain_report["checkpoint"], map_location="cpu", weights_only=False)["vocabulary"]
    pretrain_checkpoint = torch.load(ROOT / pretrain_report["checkpoint"], map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = CausalTraceTransformer(len(vocabulary), max_len=MAX_LEN).to(device)
    base.load_state_dict(pretrain_checkpoint["model_state"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    head = RuleIRHead().to(device)
    train_vectors = _hidden_rows(base, train_rows, vocabulary, device)
    dev_vectors = _hidden_rows(base, dev_rows, vocabulary, device)
    holdout_vectors = _hidden_rows(base, holdout_rows, vocabulary, device)
    train_matrix = torch.cat(train_vectors, dim=0)
    train_targets = torch.tensor([_target(row) for row in train_rows], dtype=torch.long, device=device)
    train_confidence = train_targets[:, 3].float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.02)
    losses = {
        "effect": nn.CrossEntropyLoss(),
        "transport": nn.CrossEntropyLoss(),
        "modality": nn.CrossEntropyLoss(),
        "confidence": nn.BCEWithLogitsLoss(),
    }
    best_state = None
    best_dev_loss = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(SEED + epoch)).to(device)
        total = 0.0
        batches = 0
        for start in range(0, len(train_rows), 64):
            indexes = order[start:start + 64]
            output = head(train_matrix[indexes])
            loss = (
                losses["effect"](output["effect"], train_targets[indexes, 0])
                + losses["transport"](output["transport"], train_targets[indexes, 1])
                + losses["modality"](output["modality"], train_targets[indexes, 2])
                + losses["confidence"](output["confidence"], train_confidence[indexes])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        dev = _evaluate(base, head, dev_rows, vocabulary, device, vectors=dev_vectors)
        dev_loss = 1.0 - dev["effect_accuracy"] + (1.0 - dev["modality_accuracy"])
        history.append({"epoch": epoch, "train_loss": round(total / max(batches, 1), 6), "dev_selection_loss": round(dev_loss, 6), "dev": dev})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})
    if best_state is not None:
        head.load_state_dict(best_state)
    dev_metrics = _evaluate(base, head, dev_rows, vocabulary, device, vectors=dev_vectors)
    dev_scores = _confirmation_scores(base, head, dev_rows, vocabulary, device, vectors=dev_vectors)
    confirmation_threshold, calibration = _calibrate_confirmation_threshold(dev_rows, dev_scores)
    calibrated_dev = _evaluate(base, head, dev_rows, vocabulary, device, vectors=dev_vectors, threshold=confirmation_threshold)
    raw_holdout = _evaluate(base, head, holdout_rows, vocabulary, device, vectors=holdout_vectors)
    calibrated_holdout = _evaluate(base, head, holdout_rows, vocabulary, device, vectors=holdout_vectors, threshold=confirmation_threshold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": "pg57-rule-ir-posttrainer-checkpoint-v1",
        "base_checkpoint": pretrain_report["checkpoint"],
        "head_state": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "effect_classes": EFFECTS,
        "transport_classes": TRANSPORTS,
        "modality_classes": MODALITIES,
        "oracle_is_target_only": True,
        "family_name_in_input": False,
        "unknown_family_class": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
        "device_at_training": str(device),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    report = {
        "protocol_id": "pg-pk-57-rule-ir-oracle-posttraining-v1",
        "schema_version": "pg57-rule-ir-posttraining-report-v1",
        "base_pretraining_report": str(PRETRAIN_REPORT_PATH.relative_to(ROOT)),
        "device": str(device),
        "split_counts": {"train": len(train_rows), "dev": len(dev_rows), "holdout": len(holdout_rows)},
        "training_contract": {
            "oracle_is_target_only": True,
            "family_name_in_input": False,
            "unknown_family_class": False,
            "prefix_ends_at_oracle_target": True,
        },
        "metrics": {"dev": dev_metrics, "dev_calibrated": calibrated_dev, "holdout_raw": raw_holdout, "holdout_calibrated": calibrated_holdout},
        "calibration": {"source": "PG-57 dev only", "threshold": confirmation_threshold, "metrics": calibration},
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "history_tail": history[-5:],
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_capability_claim_allowed": False,
            "status": "quarantined_until_independent_rule_ir_and_unknown_abstain_gate",
            "reason": "PG-57 tests typed-oracle posttraining targets; it does not name vulnerability families and has no independent Rule IR holdout gate yet.",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "\n".join([
            "# PG-57 typed-oracle Rule IR 后训练",
            "",
            f"设备：`{device}`；train/dev/holdout：`{len(train_rows)}/{len(dev_rows)}/{len(holdout_rows)}`。",
            f"dev effect/modality：`{dev_metrics['effect_accuracy']:.3f}` / `{dev_metrics['modality_accuracy']:.3f}`。",
            f"盲测原始 confirmed recall：`{raw_holdout['confirmed_recall']:.3f}`；dev 安全门阈值：`{confirmation_threshold:.4f}`；门控后 recall：`{calibrated_holdout['confirmed_recall']:.3f}`；unknown confirmed attempts：`{calibrated_holdout['unknown_family_confirmed_attempts']}`；negative false accepts：`{calibrated_holdout['negative_confirmed_false_accept_count']}`。",
            "结果仍保持隔离，不进入训练集或长期记忆；下一步必须增加独立 Rule IR 族外门和未知族密度/不确定性门。",
            "",
        ]) + "",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
