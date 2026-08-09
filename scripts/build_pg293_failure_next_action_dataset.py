"""Build PG-293 abstract failure->next-action data without raw replay values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg293_failure_next_action import normalize_record, sha256_json  # noqa: E402


RESEARCH = ROOT / "research"
SOURCES = {
    "trajectory_a": RESEARCH / "pg269_failure_guided_replay_dataset_v1.json",
    "trajectory_b": RESEARCH / "pg271_independent_seed_failure_guided_replay_dataset_v1.json",
    "repair_trajectory": RESEARCH / "pg244_failure_repair_trajectory_dataset_v1.json",
}
DATASET = RESEARCH / "pg293_failure_next_action_dataset_v1.json"
AUDIT = RESEARCH / "pg293_failure_next_action_dataset_audit_v1.json"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [dict(row) for row in list(payload.get("records") or [])]


def main() -> None:
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for group, path in SOURCES.items():
        raw = load_records(path)
        source_counts[group] = len(raw)
        for row in raw:
            if group == "trajectory_b":
                split = "source_holdout"
            elif group == "repair_trajectory":
                split = "seed_holdout" if int(row.get("seed", 0) or 0) % 2 == 0 else "train"
            else:
                split = "train"
            records.append(normalize_record(row, source_group=group, split=split))

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        key = str(row["record_sha256"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    records = deduped
    # Evaluation-only same-context opposite-target rows expose memorization.
    # They are never put in train or promoted as data.
    hard_negative_records: list[dict[str, Any]] = []
    for source_row in records:
        if not bool(source_row.get("safe_to_send")):
            continue
        target_tokens = [
            "[TARGET_BOS]",
            "next_action=abstain",
            "repair_action=none",
            "safe_to_send=0",
            "[TARGET_EOS]",
        ]
        counterfactual = {
            "schema_version": source_row["schema_version"],
            "record_id": f"pg293:hard-negative:{source_row['record_id']}",
            "source_group": "counterfactual_hard_negative",
            "split": "hard_negative_eval",
            "context_tokens": list(source_row["context_tokens"]),
            "target_tokens": target_tokens,
            "next_action": "abstain",
            "repair_action": "none",
            "safe_to_send": False,
            "hard_negative": True,
            "source_evidence_hash": sha256_json({"source_record": source_row["record_id"], "kind": "same_context_opposite_target"}),
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "route_identity_stored": False,
            "family_identity_stored": False,
            "oracle_label_in_context": False,
            "training_eligible": False,
            "memory_promotion_allowed": False,
        }
        counterfactual["record_sha256"] = sha256_json(counterfactual)
        hard_negative_records.append(counterfactual)
    records.extend(hard_negative_records)
    train = [row for row in records if row["split"] == "train"]
    holdout = [row for row in records if row["split"] != "train"]
    if not train or not holdout:
        raise RuntimeError("PG-293 requires non-empty train and holdout partitions")

    dataset = {
        "schema_version": "pg293-failure-next-action-dataset-v1",
        "purpose": "failure-conditioned abstract next-action/repair prediction",
        "source_datasets": {group: {"path": str(path.relative_to(ROOT).as_posix()), "sha256": file_sha(path)} for group, path in SOURCES.items()},
        "records": records,
        "counts": {
            "total": len(records),
            "train": len(train),
            "holdout": len(holdout),
            "source_holdout": sum(int(row["split"] == "source_holdout") for row in records),
            "seed_holdout": sum(int(row["split"] == "seed_holdout") for row in records),
            "positive_safe": sum(int(row["safe_to_send"]) for row in records),
            "hard_negative": sum(int(row["hard_negative"]) for row in records),
            "hard_negative_eval": len(hard_negative_records),
            "source_counts": source_counts,
        },
        "contract": {
            "context_is_family_free": True,
            "context_excludes_route": True,
            "context_excludes_oracle_and_outcome_labels": True,
            "target_is_abstract": True,
            "raw_payload_strings_stored": False,
            "raw_response_bodies_stored": False,
            "wire_emission_allowed": False,
            "memory_promotion_allowed": False,
            "hard_negative_eval_only": True,
            "holdout_required": True,
        },
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    forbidden = ("payload", "response_body", "raw_response", "<script", "javascript:", "family=", "route=")
    failures: list[str] = []
    for row in records:
        if row.get("raw_payload_strings_stored") is not False or row.get("raw_response_bodies_stored") is not False:
            failures.append(f"raw_retention:{row['record_id']}")
        if row.get("route_identity_stored") is not False or row.get("family_identity_stored") is not False:
            failures.append(f"identity_retention:{row['record_id']}")
        if any(any(term in str(token).casefold() for term in forbidden) for token in row.get("context_tokens", [])):
            failures.append(f"context_leak:{row['record_id']}")
        if str(row.get("split")) == "source_holdout" and row.get("source_group") != "trajectory_b":
            failures.append(f"source_split:{row['record_id']}")
        if row.get("split") == "hard_negative_eval" and row.get("training_eligible") is not False:
            failures.append(f"hard_negative_training:{row['record_id']}")
    audit = {
        "audit_id": "pg293-failure-next-action-independent-audit-v1",
        "status": "passed" if not failures else "failed",
        "dataset": str(DATASET.relative_to(ROOT).as_posix()),
        "dataset_sha256": dataset["dataset_sha256"],
        "checks": {
            "dataset_hash": dataset["dataset_sha256"] == sha256_json({key: value for key, value in dataset.items() if key != "dataset_sha256"}),
            "non_empty_train_holdout": bool(train and holdout),
            "source_holdout_present": any(row["split"] == "source_holdout" for row in records),
            "seed_holdout_present": any(row["split"] == "seed_holdout" for row in records),
            "safe_target_is_abstract": all("safe_to_send=" in " ".join(row["target_tokens"]) for row in records),
            "raw_and_identity_firewall": not failures,
            "memory_promotion_blocked": all(row["memory_promotion_allowed"] is False for row in records),
            "hard_negative_eval_only": all(row.get("training_eligible") is False for row in records if row.get("split") == "hard_negative_eval"),
        },
        "failures": failures,
        "interpretation": "PG-293 只证明抽象失败轨迹可形成 next-action 训练输入；不证明 payload 成功或真实漏洞能力。",
    }
    audit["audit_sha256"] = sha256_json(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit["status"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
