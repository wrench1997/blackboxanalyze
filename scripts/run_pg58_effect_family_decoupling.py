"""PG-58: separate effect confirmation from vulnerability-family naming.

The frozen PG-56 trace representation is shared by two independent heads:

* an effect head inherited from PG-57, which may only say confirmed/rejected;
* a new family head trained only on known-family labels, which may name a
  family only after the effect and uncertainty gates pass.

No family name is present in the prefix seen by either head.  Unknown
``template_injection`` rows are excluded from family-head training and are
used only for dev/holdout gate calibration and evaluation.
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
EFFECT_CHECKPOINT_PATH = ROOT / "artifacts" / "pg57-rule-ir-posttrainer" / "rule_ir_head.pt"
REPORT_PATH = ROOT / "research" / "pg58_effect_family_decoupling_report_v1.json"
TRACE_PATH = ROOT / "research" / "pg58_effect_family_decoupling_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg58_effect_family_decoupling_report_v1.md"
OUTPUT_DIR = ROOT / "artifacts" / "pg58-effect-family-decoupling"
CHECKPOINT_PATH = OUTPUT_DIR / "family_head.pt"
SEED = 20260803
MAX_LEN = 128
EPOCHS = 80
KNOWN_FAMILIES = [
    "xss",
    "injection",
    "authentication",
    "access_control",
    "logic",
    "url_redirect",
    "input_validation",
    "command_injection",
    "ordinary_response",
]
UNKNOWN_FAMILY = "template_injection"


class FamilyNamingHead(nn.Module):
    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, len(KNOWN_FAMILIES)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden))


class EffectHead(nn.Module):
    """Shape-compatible frozen copy of PG-57's effect head."""

    def __init__(self, d_model: int = 96) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.effect = nn.Linear(d_model, 3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.effect(self.norm(hidden))


def _load_rows() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    pretrain = json.loads(PRETRAIN_REPORT_PATH.read_text(encoding="utf-8"))
    rows = dataset["rows"]
    return dataset, pretrain, [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "dev"], [row for row in rows if row["split"] == "holdout"]


def _prefix_ids(row: dict[str, Any], vocabulary: dict[str, int]) -> torch.Tensor:
    tokens = row["tokens"][:MAX_LEN]
    marker = min(tokens.index("ORACLE_TARGET"), len(tokens) - 1)
    ids = [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens[:marker + 1]]
    return torch.tensor([ids], dtype=torch.long)


def _hidden_rows(base: CausalTraceTransformer, rows: list[dict[str, Any]], vocabulary: dict[str, int], device: torch.device) -> list[torch.Tensor]:
    base.eval()
    with torch.no_grad():
        vectors = []
        for row in rows:
            prefix = _prefix_ids(row, vocabulary).to(device)
            mask = torch.ones_like(prefix, dtype=torch.bool)
            vectors.append(base.encode(prefix, mask)[:, -1, :].detach().clone())
    return vectors


def _label(row: dict[str, Any]) -> int:
    family = str(row["target"].get("family") or UNKNOWN_FAMILY)
    return KNOWN_FAMILIES.index(family)


def _scores(head: FamilyNamingHead, effect_head: EffectHead, vectors: list[torch.Tensor]) -> list[dict[str, Any]]:
    head.eval()
    effect_head.eval()
    with torch.inference_mode():
        outputs = head(torch.cat(vectors, dim=0)).softmax(-1)
        effect_outputs = effect_head(torch.cat(vectors, dim=0)).softmax(-1)
    result = []
    for index in range(len(vectors)):
        family_probability, family_index = outputs[index].max(-1)
        sorted_probability, _ = outputs[index].sort(descending=True)
        effect_probability = effect_outputs[index, 1]
        result.append({
            "predicted_family": KNOWN_FAMILIES[int(family_index)],
            "family_confidence": float(family_probability.cpu()),
            "family_margin": float((sorted_probability[0] - sorted_probability[1]).cpu()),
            "effect_confirmed_probability": float(effect_probability.cpu()),
            "effect_predicted_confirmed": int(effect_outputs[index].argmax()) == 1,
        })
    return result


def _metrics(rows: list[dict[str, Any]], scores: list[dict[str, Any]], *, effect_threshold: float, family_threshold: float, margin_threshold: float) -> dict[str, Any]:
    accepted: list[int] = []
    for index, score in enumerate(scores):
        if not score["effect_predicted_confirmed"]:
            continue
        if score["effect_confirmed_probability"] < effect_threshold:
            continue
        if score["family_confidence"] < family_threshold or score["family_margin"] < margin_threshold:
            continue
        if score["predicted_family"] == "ordinary_response":
            continue
        accepted.append(index)
    known = [index for index, row in enumerate(rows) if row["target"].get("family") in KNOWN_FAMILIES and row["target"].get("family") != "ordinary_response"]
    unknown = [index for index, row in enumerate(rows) if row["target"].get("family") == UNKNOWN_FAMILY]
    negative = [index for index, row in enumerate(rows) if row["target"].get("family") == "ordinary_response"]
    hits = [index for index in accepted if rows[index]["target"].get("family") == scores[index]["predicted_family"]]
    return {
        "count": len(rows),
        "emitted_family_count": len(accepted),
        "known_positive_count": len(known),
        "known_family_hits": len([index for index in hits if index in known]),
        "known_family_recall": round(len([index for index in hits if index in known]) / max(len(known), 1), 6),
        "known_wrong_family_count": sum(int(index in accepted and index not in hits) for index in known),
        "unknown_family_count": len(unknown),
        "unknown_misname_count": sum(int(index in accepted) for index in unknown),
        "unknown_strict_abstain": all(index not in accepted for index in unknown),
        "negative_false_accept_count": sum(int(index in accepted) for index in negative),
        "abstain_rate": round(1.0 - len(accepted) / max(len(rows), 1), 6),
    }


def _calibrate(dev_rows: list[dict[str, Any]], dev_scores: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    def candidates(values: list[float]) -> list[float]:
        # A dev-only quantile grid keeps calibration bounded while retaining
        # the extrema that can enforce a strict zero-error gate.
        unique = sorted(set(values) | {1.0}, reverse=True)
        if len(unique) <= 24:
            return unique
        positions = [round(index * (len(unique) - 1) / 23) for index in range(24)]
        return sorted({unique[position] for position in positions} | {1.0}, reverse=True)

    effect_values = candidates([score["effect_confirmed_probability"] for score in dev_scores])
    family_values = candidates([score["family_confidence"] for score in dev_scores])
    margin_values = candidates([score["family_margin"] for score in dev_scores])
    best: tuple[int, dict[str, float], dict[str, Any]] | None = None
    for effect_threshold in effect_values:
        for family_threshold in family_values:
            for margin_threshold in margin_values:
                metrics = _metrics(dev_rows, dev_scores, effect_threshold=effect_threshold, family_threshold=family_threshold, margin_threshold=margin_threshold)
                if metrics["unknown_misname_count"] or metrics["negative_false_accept_count"] or metrics["known_wrong_family_count"]:
                    continue
                score = metrics["known_family_hits"]
                if best is None or score > best[0]:
                    best = (score, {"effect_threshold": effect_threshold, "family_threshold": family_threshold, "margin_threshold": margin_threshold}, metrics)
    if best is None:
        return {"effect_threshold": 1.0, "family_threshold": 1.0, "margin_threshold": 1.0}, {"known_family_hits": 0, "unknown_misname_count": 0, "negative_false_accept_count": 0}
    return best[1], best[2]


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    dataset, pretrain, train_rows, dev_rows, holdout_rows = _load_rows()
    checkpoint = torch.load(ROOT / pretrain["checkpoint"], map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = CausalTraceTransformer(len(vocabulary), max_len=MAX_LEN).to(device)
    base.load_state_dict(checkpoint["model_state"])
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    train_rows_known = [row for row in train_rows if row["target"].get("family") in KNOWN_FAMILIES]
    train_vectors = _hidden_rows(base, train_rows_known, vocabulary, device)
    dev_vectors = _hidden_rows(base, dev_rows, vocabulary, device)
    holdout_vectors = _hidden_rows(base, holdout_rows, vocabulary, device)
    train_matrix = torch.cat(train_vectors, dim=0)
    labels = torch.tensor([_label(row) for row in train_rows_known], dtype=torch.long, device=device)
    dev_known_positions = [index for index, row in enumerate(dev_rows) if row["target"].get("family") in KNOWN_FAMILIES]
    dev_known_matrix = torch.cat([dev_vectors[index] for index in dev_known_positions], dim=0)
    dev_known_labels = torch.tensor([_label(dev_rows[index]) for index in dev_known_positions], dtype=torch.long, device=device)
    head = FamilyNamingHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_state = None
    best_dev_loss = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        generator = torch.Generator().manual_seed(SEED + epoch)
        order = torch.randperm(len(train_rows_known), generator=generator).to(device)
        total = 0.0
        batches = 0
        for start in range(0, len(order), 64):
            indexes = order[start:start + 64]
            logits = head(train_matrix[indexes])
            loss = loss_fn(logits, labels[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        # Checkpoint selection uses only known-family dev labels; unknown rows
        # remain a gate/evaluation target and never become a family class.
        with torch.inference_mode():
            dev_logits = head(dev_known_matrix)
            dev_loss = float(nn.CrossEntropyLoss()(dev_logits, dev_known_labels).cpu())
        history.append({"epoch": epoch, "train_loss": round(total / max(batches, 1), 6), "dev_family_loss": round(dev_loss, 6)})
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in head.state_dict().items()})
    if best_state is not None:
        head.load_state_dict(best_state)
    effect_checkpoint = torch.load(EFFECT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    effect_head = EffectHead().to(device)
    # PG-57 has extra transport/modality layers, but the effect state names
    # are identical and are deliberately the only ones loaded here.
    effect_head.load_state_dict({"norm.weight": effect_checkpoint["head_state"]["norm.weight"], "norm.bias": effect_checkpoint["head_state"]["norm.bias"], "effect.weight": effect_checkpoint["head_state"]["effect.weight"], "effect.bias": effect_checkpoint["head_state"]["effect.bias"]})
    dev_scores = _scores(head, effect_head, dev_vectors)
    holdout_scores = _scores(head, effect_head, holdout_vectors)
    raw_thresholds = {"effect_threshold": 0.5, "family_threshold": 0.5, "margin_threshold": 0.05}
    calibrated_thresholds, calibration = _calibrate(dev_rows, dev_scores)
    dev_raw = _metrics(dev_rows, dev_scores, **raw_thresholds)
    holdout_raw = _metrics(holdout_rows, holdout_scores, **raw_thresholds)
    dev_calibrated = _metrics(dev_rows, dev_scores, **calibrated_thresholds)
    holdout_calibrated = _metrics(holdout_rows, holdout_scores, **calibrated_thresholds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    family_checkpoint = {
        "schema_version": "pg58-family-naming-head-checkpoint-v1",
        "base_checkpoint": pretrain["checkpoint"],
        "head_state": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "known_families": KNOWN_FAMILIES,
        "unknown_family_excluded_from_training": UNKNOWN_FAMILY,
        "family_name_in_input": False,
        "effect_head_is_separate": True,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
        "device_at_training": str(device),
    }
    torch.save(family_checkpoint, CHECKPOINT_PATH)
    trace = {
        "schema_version": "pg58-effect-family-decoupling-trace-v1",
        "evaluation_only": True,
        "rows": [
            {
                "trace_id": row["trace_id"],
                "split": row["split"],
                "predicted_family": score["predicted_family"],
                "family_confidence_bucket": "high" if score["family_confidence"] >= 0.8 else "low",
                "effect_confirmed_probability_bucket": "high" if score["effect_confirmed_probability"] >= 0.8 else "low",
                "raw_probe_stored": False,
                "raw_response_stored": False,
            }
            for row, score in zip(holdout_rows, holdout_scores)
        ],
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "long_term_memory_write": False,
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "protocol_id": "pg-pk-58-effect-family-decoupling-v1",
        "schema_version": "pg58-effect-family-decoupling-report-v1",
        "device": str(device),
        "split_counts": {"train": len(train_rows_known), "dev": len(dev_rows), "holdout": len(holdout_rows)},
        "training_contract": {
            "family_head_train_known_families_only": True,
            "unknown_family_excluded_from_training": UNKNOWN_FAMILY,
            "family_name_in_input": False,
            "effect_head_separate": True,
            "effect_head_is_not_family_evidence": True,
        },
        "metrics": {
            "dev_raw": dev_raw,
            "holdout_raw": holdout_raw,
            "dev_calibrated": dev_calibrated,
            "holdout_calibrated": holdout_calibrated,
        },
        "thresholds": {
            "raw": raw_thresholds,
            "calibrated": calibrated_thresholds,
            "calibration_source": "PG-58 dev only",
            "calibration_metrics": calibration,
        },
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest(),
        "trace": str(TRACE_PATH.relative_to(ROOT)),
        "history_tail": history[-5:],
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "formal_capability_claim_allowed": False,
            "status": "quarantined_until_family_ood_gate",
            "reason": "Effect confirmation and family naming are scored separately; framed holdout must pass unknown abstain and known-family recall gates before any promotion.",
        },
    }
    report["report_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "\n".join([
            "# PG-58 effect confirmation / family naming 解耦",
            "",
            f"训练族分类样本：`{len(train_rows_known)}`；dev/holdout：`{len(dev_rows)}/{len(holdout_rows)}`；设备：`{device}`。",
            f"盲测 raw known recall：`{holdout_raw['known_family_recall']:.3f}`；wrong family：`{holdout_raw['known_wrong_family_count']}`；unknown misname：`{holdout_raw['unknown_misname_count']}`。",
            f"盲测 calibrated known recall：`{holdout_calibrated['known_family_recall']:.3f}`；unknown misname：`{holdout_calibrated['unknown_misname_count']}`；negative false accept：`{holdout_calibrated['negative_false_accept_count']}`；abstain：`{holdout_calibrated['abstain_rate']:.3f}`。",
            "效果头只负责 confirmed/rejected；族命名头只在效果门通过后运行。所有结果仍在隔离区，未进入长期记忆。",
            "",
        ]) + "",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
