"""Build append-only context/target vocabularies for PG-341.

The context and target inventories stay separate in the manifest even though
the decoder uses a single bounded embedding table.  The builder includes the
full-axis inventory before any diagnostic run, so a coarse-process smoke
cannot silently define the vocabulary for the eventual page model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "research" / "pg341_target_conditioned_process_full_axis_dataset_v1.json"
DEFAULT_OUTPUT = ROOT / "research" / "pg341_target_conditioned_vocabulary_v1.json"
SPECIAL_CONTEXT = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")
SPECIAL_TARGET = ("[TARGET_BOS]", "[TARGET_EOS]")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError("dataset must be an object")
    return value


def build(dataset: Mapping[str, Any], *, base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = base if isinstance(base, Mapping) else {}
    context = {str(token) for token in base.get("context_tokens") or []}
    target = {str(token) for token in base.get("target_tokens") or []}
    for row in dataset.get("records") or []:
        if not isinstance(row, Mapping):
            continue
        context.update(str(token) for token in row.get("context_tokens") or [])
        target.update(str(token) for token in row.get("target_tokens") or [])
    context_ordered = list(SPECIAL_CONTEXT) + sorted(context - set(SPECIAL_CONTEXT))
    target_ordered = list(SPECIAL_TARGET) + sorted(target - set(SPECIAL_TARGET))
    result: dict[str, Any] = {
        "schema_version": "pg341-target-conditioned-vocabulary-v1",
        "status": "diagnostic_only",
        "append_only": True,
        "dataset_sha256": dataset.get("dataset_sha256"),
        "base_vocabulary_sha256": base.get("vocabulary_sha256"),
        "context_tokens": list(dict.fromkeys(context_ordered)),
        "target_tokens": list(dict.fromkeys(target_ordered)),
        "reserved_context": list(SPECIAL_CONTEXT),
        "reserved_target": list(SPECIAL_TARGET),
        "coverage": {
            "coarse_process_context": True,
            "full_axis_context": True,
            "ask_targets": any(str(token).startswith("question=ask") for token in target),
            "failure_repair_targets": any(str(token) in {"next_action=repair", "next_action=repair_abstract_plan"} for token in target),
            "negative_abstain_targets": "next_action=abstain" in target,
            "holdout_not_used_for_vocab": False,
        },
        "training_eligibility": {"allowed": False, "reason": "PG-341 full-axis target coverage and operator review are pending"},
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
        "vocabulary_sha256": "",
    }
    result["vocabulary_sha256"] = _sha(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PG-341 append-only target-conditioned vocabulary")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = _load(args.dataset)
    base = _load(args.base) if args.base else None
    result = build(dataset, base=base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "context": len(result["context_tokens"]), "target": len(result["target_tokens"]), "vocabulary_sha256": result["vocabulary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
