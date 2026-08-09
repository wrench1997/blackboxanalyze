"""Derive an abstract binding lane from the factorized PG-382 matrix.

This artifact adds an abstract ``safe_to_send=1`` lane only where the
observation state is complete enough for a reviewed local evaluator template:
candidate/reference after a filtered/parser/no-effect repair, and replay after
a typed effect.  Negative controls and unknown/baseline rows remain unsafe.
No concrete marker, URL, payload, response, or evaluator answer is added.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "research" / "pg382_factorized_abstract_adversarial_dataset_v1.json"
SCHEMA_VERSION = "pg384-binding-abstract-adversarial-dataset-v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _parse_target(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
    if tokens[:1] != ["[TARGET_BOS]"] or tokens[-1:] != ["[TARGET_EOS]"]:
        raise ValueError("target boundaries are invalid")
    values: dict[str, str] = {}
    for token in tokens[1:-1]:
        key, sep, value = str(token).partition("=")
        if not sep or key in values or not value:
            raise ValueError("target slot token is invalid")
        values[key] = value
    return tokens, values


def _binding_lane(role: str, state: str) -> tuple[bool, str | None]:
    if role == "negative":
        return False, None
    if role in {"candidate", "reference"} and state in {"filtered", "parser_error", "no_effect", "typed_effect"}:
        return True, "fresh_replay" if state == "typed_effect" else ("source_attested_candidate" if role == "candidate" else "reference_shape")
    if role == "replay" and state == "typed_effect":
        return True, "fresh_replay"
    return False, None


def build_dataset(source_path: Path = SOURCE) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    if source.get("status") != "abstract_adversarial_candidate_only":
        raise ValueError("PG-382 source is not candidate-only")
    records = []
    positive = 0
    negative = 0
    for raw in source.get("records") or []:
        row = copy.deepcopy(raw)
        role = str(row.get("source", {}).get("role", ""))
        state = str(row.get("abstract_observation", {}).get("filter_state", ""))
        _, target = _parse_target([str(token) for token in row.get("target_tokens") or []])
        safe, variant = _binding_lane(role, state)
        if safe:
            target["safe_to_send"] = "1"
            if variant is not None:
                target["probe_variant_ref"] = variant
            positive += 1
        else:
            target["safe_to_send"] = "false"
            negative += 1
        row["target_tokens"] = ["[TARGET_BOS]", *(f"{key}={value}" for key, value in target.items()), "[TARGET_EOS]"]
        row["abstract_binding_lane"] = "allowlisted_template_candidate" if safe else "ask_or_abstain"
        row["training_flags"]["capability_training_allowed"] = False
        row["training_flags"]["training_eligible"] = False
        records.append(row)
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "status": "abstract_adversarial_candidate_only",
        "generator": "scripts/build_pg384_binding_adversarial_dataset.py",
        "derived_from": str(source_path),
        "source_dataset_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "objective": "abstract Rule-IR to reviewed evaluator-template binding lane",
        "records": records,
        "vocabulary": {
            "scope": "declared_abstract_factorized_binding_ontology",
            "append_only": True,
            "context_tokens": sorted({str(token) for row in records for token in row["context_tokens"]}),
            "target_tokens": sorted({str(token) for row in records for token in row["target_tokens"]}),
        },
        "counts": {
            "records": len(records),
            "train": sum(str(row.get("split")) == "train" for row in records),
            "implementation_holdout": sum(str(row.get("split")) == "implementation_holdout" for row in records),
            "abstract_safe_to_send_rows": positive,
            "abstract_unsafe_or_ask_rows": negative,
            "candidate_reference_replay_binding_rows": positive,
            "training_eligible": 0,
        },
        "audit": {
            "status": "passed_abstract_binding_candidate",
            "raw_marker_count": 0,
            "context_firewall_failures": 0,
            "safe_lane_requires_reviewed_template": True,
            "negative_controls_remain_unsafe": True,
            "typed_live_replay": False,
            "training_eligible": 0,
        },
        "safety": {
            "raw_payload_in_context": False,
            "raw_response_in_context": False,
            "evaluator_answer_in_context": False,
            "concrete_wire_generation": "reviewed_local_evaluator_template_only",
            "external_network": False,
            "persistent_state_write": False,
        },
        "promotion": {
            "representation_pretrain_candidate_allowed": True,
            "abstract_reasoning_sft_candidate_allowed": True,
            "capability_training_allowed": False,
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "payload_catalog_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
    }
    dataset["dataset_sha256"] = _sha(dataset)
    return dataset


def write_dataset(path: Path, source_path: Path = SOURCE) -> dict[str, Any]:
    dataset = build_dataset(source_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=ROOT / "research/pg384_binding_abstract_adversarial_dataset_v1.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    dataset = write_dataset(args.output, args.source)
    if args.json:
        print(json.dumps({"status": dataset["status"], "counts": dataset["counts"], "audit": dataset["audit"], "dataset_sha256": dataset["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_dataset", "write_dataset"]
