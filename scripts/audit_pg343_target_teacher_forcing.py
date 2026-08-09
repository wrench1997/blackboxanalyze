"""Read-only teacher-forced audit for PG-343 target-conditioned candidates.

Greedy generation can cascade after the first wrong token.  This audit keeps
the full abstract context and target stream, but reports only bounded slot
accuracy/loss; it never writes token sequences, payloads, responses, or
evaluator values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import PAD, UNK
from app.pg295_causal_moe import CausalMoEConfig, CausalMoELanguageModel


SLOTS = ("question", "next_action", "repair_action", "transport_ref", "field_role_ref", "encoding_ref", "probe_variant_ref", "safe_to_send", "target_eos")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _vocab(vocabulary: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> dict[str, int]:
    raw = checkpoint.get("vocabulary")
    if isinstance(raw, Mapping):
        return {str(key): int(value) for key, value in raw.items()}
    tokens = [PAD, UNK, *(vocabulary.get("context_tokens") or []), *(vocabulary.get("target_tokens") or [])]
    return {str(token): index for index, token in enumerate(dict.fromkeys(tokens))}


def _rows(dataset: Mapping[str, Any], split: str) -> list[Mapping[str, Any]]:
    return [row for row in dataset.get("records", []) if isinstance(row, Mapping) and str(row.get("split")) == split]


def _audit_seed(model: CausalMoELanguageModel, rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, int], device: torch.device) -> dict[str, Any]:
    totals = {"tokens": 0, "correct": 0, "loss": 0.0}
    slot_totals = {slot: {"total": 0, "correct": 0} for slot in SLOTS}
    pad = int(vocabulary.get(PAD, 0))
    unknown = int(vocabulary.get(UNK, 1))
    model.eval()
    with torch.inference_mode():
        for row in rows:
            context = [str(token) for token in row.get("context_tokens") or []]
            target = [str(token) for token in row.get("target_tokens") or []]
            sequence = [int(vocabulary.get(token, unknown)) for token in [*context, *target]]
            if len(sequence) < 2 or len(sequence) > model.config.max_length:
                continue
            input_ids = torch.tensor(sequence[:-1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(input_ids, valid_mask=torch.ones_like(input_ids, dtype=torch.bool))
            labels = torch.tensor(sequence[1:], dtype=torch.long, device=device)
            target_start = len(context)
            target_end = len(sequence) - 1
            for offset, position in enumerate(range(target_start, target_end)):
                slot = SLOTS[offset] if offset < len(SLOTS) else "target_extra"
                prediction = int(logits[0, position].argmax(-1).item())
                label = int(labels[position].item())
                totals["tokens"] += 1
                totals["correct"] += int(prediction == label)
                totals["loss"] += float(F.cross_entropy(logits[0, position].unsqueeze(0), labels[position].unsqueeze(0), reduction="sum").item())
                slot_totals.setdefault(slot, {"total": 0, "correct": 0})
                slot_totals[slot]["total"] += 1
                slot_totals[slot]["correct"] += int(prediction == label)
    return {
        "rows": len(rows),
        "target_token_count": totals["tokens"],
        "teacher_forced_token_accuracy": round(totals["correct"] / max(totals["tokens"], 1), 6),
        "teacher_forced_mean_loss": round(totals["loss"] / max(totals["tokens"], 1), 6),
        "slot_accuracy": {slot: round(value["correct"] / max(value["total"], 1), 6) if value["total"] else None for slot, value in slot_totals.items()},
    }


def audit(*, dataset_path: Path, vocabulary_path: Path, checkpoint_path: Path, split: str) -> dict[str, Any]:
    dataset = _load(dataset_path)
    vocabulary = _load(vocabulary_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = _vocab(vocabulary, checkpoint)
    config = CausalMoEConfig(**dict(checkpoint["config"]))
    rows = _rows(dataset, split)
    by_seed: dict[str, Any] = {}
    for seed, state in sorted(dict(checkpoint.get("states") or {}).items()):
        model = CausalMoELanguageModel(vocab_size=len(vocab), config=config)
        model.load_state_dict(state)
        by_seed[str(seed)] = _audit_seed(model, rows, vocab, torch.device("cpu"))
    return {
        "schema_version": "pg343-target-teacher-forcing-audit-v1",
        "status": "diagnostic_only",
        "split": split,
        "dataset_sha256": _sha_file(dataset_path),
        "vocabulary_sha256": _sha_file(vocabulary_path),
        "checkpoint_sha256": _sha_file(checkpoint_path),
        "slot_names": list(SLOTS),
        "seeds": by_seed,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "implementation_holdout"), default="implementation_holdout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(dataset_path=args.dataset, vocabulary_path=args.vocabulary, checkpoint_path=args.checkpoint, split=args.split)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
