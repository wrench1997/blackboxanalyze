"""Build PG-300 compositional question-policy data without wire content."""

from __future__ import annotations

import itertools
import json
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import sha256_json  # noqa: E402
from app.pg300_question_policy import audit_question_records, canonical_question_context, question_for_observation, question_record  # noqa: E402


RESEARCH = ROOT / "research"
DATASET = RESEARCH / "pg300_question_policy_dataset_v1.json"
AUDIT = RESEARCH / "pg300_question_policy_audit_v1.json"


def make_source(index: int, values: dict[str, str], surface: tuple[str, str, str], split: str, hard_negative: bool = False) -> dict:
    method, channel, status = surface
    raw = {
        "record_id": f"pg300:source:{index}",
        "split": split,
        "training_eligible": split == "train",
        "hard_negative": hard_negative,
        "context_tokens": [
            "[BOS]",
            f"surface_method={method}",
            f"surface_channel={channel}",
            f"surface_status={status}",
            "surface_order=permuted" if index % 2 else "surface_order=canonical",
            *[f"{key}={value}" for key, value in values.items()],
            "[EOS]",
        ],
    }
    return question_record(raw)


def main() -> None:
    values = {
        "typed_available": ("0", "1", "unknown"),
        "feedback_state": ("unresolved", "observable_progress", "unknown"),
        "replay_ready": ("0", "1", "unknown"),
        "evidence_present": ("0", "1", "unknown"),
    }
    # Keep several seen token values but a small Cartesian surface set; the
    # experiment is about compositional missingness, not data volume.
    surfaces = [("GET", "query", "2xx"), ("POST", "form", "2xx"), ("GET", "form", "302"), ("POST", "query", "5xx"), ("GET", "query", "5xx"), ("POST", "form", "302")]
    holdout_surfaces = {("GET", "query", "302"), ("POST", "form", "5xx")}
    holdout_values = {("unknown", "unknown", "unknown", "unknown"), ("unknown", "observable_progress", "unknown", "1")}
    records: list[dict] = []
    index = 0
    for combo in itertools.product(*values.values()):
        observation = dict(zip(values, combo))
        for surface in surfaces:
            is_holdout = surface in holdout_surfaces or tuple(combo) in holdout_values
            split = "implementation_holdout" if is_holdout else "train"
            # Two orderings and two copies make surface composition visible,
            # while the canonicalizer removes their presentation order.
            for _ in range(2):
                records.append(make_source(index, observation, surface, split))
                index += 1
    # The safe abstain/question=none branch is deliberately oversampled.  A
    # question policy that asks on every context can score high recall while
    # being unusable; this balancing is a hard-negative design choice, not a
    # hidden oracle label.
    none_rows = [row for row in records if row.get("split") == "train" and row.get("question") == "none"]
    for source_row in none_rows:
        for copy_index in range(4):
            clone = copy.deepcopy(source_row)
            clone["record_id"] = f"{source_row['record_id']}:balanced_none:{copy_index}"
            clone["record_sha256"] = sha256_json(clone)
            records.append(clone)
    # Same fully observed observation with a different hidden label is a
    # hard-negative: the question-only model must not infer a send decision.
    hard_values = {"typed_available": "1", "feedback_state": "observable_progress", "replay_ready": "1", "evidence_present": "1"}
    for surface in surfaces:
        for _ in range(2):
            records.append(make_source(index, hard_values, surface, "hard_negative_eval", hard_negative=True))
            index += 1
    audit = audit_question_records(records)
    dataset = {
        "schema_version": "pg300-question-policy-dataset-v1",
        "purpose": "short causal question token task over compositional observation slots",
        "source": {"kind": "synthetic_observation_combinations", "raw_payload_strings_stored": False, "wire_emission": False},
        "records": records,
        "counts": {
            "total": len(records),
            "train": sum(row.get("split") == "train" for row in records),
            "implementation_holdout": sum(row.get("split") == "implementation_holdout" for row in records),
            "hard_negative_eval": sum(row.get("split") == "hard_negative_eval" for row in records),
            "surface_holdout": sorted("/".join(item) for item in holdout_surfaces),
            "observation_holdout": [list(item) for item in sorted(holdout_values)],
        },
        "contract": {
            "causal_next_token": True,
            "question_token_only": True,
            "observation_slots_are_explicit": True,
            "surface_order_canonicalized": True,
            "oracle_blind": True,
            "literal_payload_strings_stored": False,
            "wire_emission_allowed": False,
            "memory_promotion_allowed": False,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_payload = {"audit_id": "pg300-question-policy-audit-v1", "schema_version": "pg300-question-policy-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], **audit}
    audit_payload["audit_sha256"] = sha256_json(audit_payload)
    AUDIT.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit_payload["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit_payload["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit_payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
