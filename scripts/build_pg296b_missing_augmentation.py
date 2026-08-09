"""Build a source-isolated PG-296B missing-pattern augmentation split."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from app.pg293_failure_next_action import TARGET_BOS, TARGET_EOS, sha256_json  # noqa: E402
from app.pg294_active_repair import audit_records  # noqa: E402
from run_pg296_missing_pattern_eval import make_row  # noqa: E402


RESEARCH = ROOT / "research"
SOURCE = RESEARCH / "pg294_active_repair_dataset_v1.json"
DATASET = RESEARCH / "pg296b_missing_augmentation_dataset_v1.json"
AUDIT = RESEARCH / "pg296b_missing_augmentation_audit_v1.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sparse_row(base: dict[str, Any], *, split: str, training_eligible: bool) -> dict[str, Any]:
    context = [
        "[BOS]",
        "phase=diagnose",
        "method=GET",
        "status=unknown",
        "field_bucket=unknown",
        "typed_available=unknown",
        "feedback_state=unknown",
        "replay_ready=unknown",
        "evidence_present=unknown",
        "[EOS]",
    ]
    row = {
        "schema_version": "pg296b-missing-pattern-v1",
        "record_id": f"pg296b:sparse_fields:{sha256_json(context + [str(base.get('record_id', ''))])[:16]}",
        "source_group": "independent_missing_pattern",
        "split": split,
        "state_id": "sparse_fields",
        "pattern": "sparse_fields",
        "context_tokens": context,
        "target_tokens": [TARGET_BOS, "next_action=recheck_oracle", "repair_action=recheck_oracle", "question=ask_typed_availability", "safe_to_send=0", TARGET_EOS],
        "next_action": "recheck_oracle",
        "repair_action": "recheck_oracle",
        "question": "ask_typed_availability",
        "safe_to_send": False,
        "hard_negative": False,
        "oracle_label_in_context": False,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "route_identity_stored": False,
        "family_identity_stored": False,
        "training_eligible": training_eligible,
        "memory_promotion_allowed": False,
    }
    row["record_sha256"] = sha256_json(row)
    return row


def mutate(row: dict[str, Any], *, split: str, training_eligible: bool) -> dict[str, Any]:
    clone = copy.deepcopy(row)
    clone["split"] = split
    clone["training_eligible"] = training_eligible
    clone["source_group"] = "augmented_observation_surface"
    clone["record_id"] = f"pg296b:{clone.get('pattern', 'pattern')}:{sha256_json(clone.get('context_tokens', []))[:16]}"
    clone["record_sha256"] = sha256_json(clone)
    return clone


def main() -> None:
    source = load(SOURCE)
    source_records = list(source.get("records") or [])
    base_train = [row for row in source_records if row.get("split") == "train" and row.get("state_id") == "missing_key"][:24]
    base_holdout = [row for row in source_records if row.get("split") in {"source_holdout", "seed_holdout"} and row.get("state_id") == "missing_key"][:24]
    if not base_train or not base_holdout:
        raise RuntimeError("PG-296B requires missing_key train and holdout source rows")
    records = [row for row in source_records if row.get("split") == "train" and row.get("training_eligible") is True]
    for base in base_train:
        for pattern in ("get_query_order_shift", "permuted_missing"):
            records.append(mutate(make_row(base, pattern), split="train", training_eligible=True))
    ood = []
    for base in base_holdout:
        ood.append(mutate(make_row(base, "post_form_decoy"), split="implementation_holdout", training_eligible=False))
        ood.append(sparse_row(base, split="implementation_holdout", training_eligible=False))
    records.extend(ood)
    # Keep the pre-registered same-context negative evaluation from PG-294.
    records.extend(row for row in source_records if row.get("split") == "hard_negative_eval")
    dataset = {
        "schema_version": "pg296b-missing-augmentation-dataset-v1",
        "purpose": "causal MoE missing-observation compositional augmentation with implementation holdout",
        "source": {"path": str(SOURCE.relative_to(ROOT).as_posix()), "sha256": source.get("dataset_sha256")},
        "records": records,
        "counts": {"total": len(records), "train": sum(int(row.get("split") == "train" and row.get("training_eligible") is True) for row in records), "implementation_holdout": len(ood), "hard_negative_eval": sum(int(row.get("split") == "hard_negative_eval") for row in records), "train_patterns": ["base", "get_query_order_shift", "permuted_missing"], "holdout_patterns": ["post_form_decoy", "sparse_fields"]},
        "contract": {"oracle_blind": True, "implementation_holdout": True, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "wire_emission_allowed": False, "memory_promotion_allowed": False},
    }
    dataset["dataset_sha256"] = sha256_json(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = audit_records(records)
    audit.update({"audit_id": "pg296b-missing-augmentation-audit-v1", "schema_version": "pg296b-missing-augmentation-audit-v1", "dataset": str(DATASET.relative_to(ROOT).as_posix()), "dataset_sha256": dataset["dataset_sha256"], "checks": {**dict(audit.get("checks") or {}), "implementation_holdout_present": bool(ood), "train_pattern_diversity": len({row.get("pattern", "base") for row in records if row.get("split") == "train"}) >= 3, "sparse_fields_holdout_only": all(row.get("pattern") != "sparse_fields" or row.get("split") == "implementation_holdout" for row in records)}})
    audit["status"] = "passed" if all(bool(value) for value in audit["checks"].values()) else "failed"
    audit["audit_sha256"] = sha256_json(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(DATASET.relative_to(ROOT)), "audit": str(AUDIT.relative_to(ROOT)), "counts": dataset["counts"], "status": audit["status"], "dataset_sha256": dataset["dataset_sha256"], "audit_sha256": audit["audit_sha256"]}, ensure_ascii=False, indent=2))
    if audit["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
