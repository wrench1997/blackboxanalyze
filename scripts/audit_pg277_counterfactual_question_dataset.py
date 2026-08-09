"""Independent audit for the PG-277 counterfactual question dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.counterfactual_surface_fixture import source_sha256  # noqa: E402

DATASET = ROOT / "research" / "pg277_counterfactual_question_dataset_v1.json"
AUDIT = ROOT / "research" / "pg277_counterfactual_question_dataset_audit_v1.json"


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    rows = list(data.get("records", []))
    train = [row for row in rows if row.get("split") == "alpha_beta_train"]
    holdout = [row for row in rows if row.get("split") == "gamma_seed_holdout"]
    coarse = dict(data.get("projection_collision_audit", {}).get("coarse") or {})
    enriched = dict(data.get("projection_collision_audit", {}).get("enriched") or {})
    check("dataset_hash", data.get("dataset_sha256") == sha({key: value for key, value in data.items() if key != "dataset_sha256"}))
    check("fixture_hash", data.get("source", {}).get("fixture_source_sha256") == source_sha256() and all(row.get("source_hash") == source_sha256() for row in rows))
    check("split_counts", len(train) == 40 and len(holdout) == 20)
    check("split_disjoint", {row.get("variant") for row in train} == {"alpha", "beta"} and {row.get("variant") for row in holdout} == {"gamma"} and not ({row.get("seed_id") for row in train} & {row.get("seed_id") for row in holdout}))
    check("positive_support", sum(bool(row.get("labels", {}).get("expected_positive")) for row in train) == 8 and sum(bool(row.get("labels", {}).get("expected_positive")) for row in holdout) == 4)
    check("matched_shape_conflicts_exist", int(coarse.get("conflict_group_count", 0)) >= 1 and int(coarse.get("conflicting_record_count", 0)) >= 1)
    check("enriched_conflicts_resolved", int(enriched.get("conflict_group_count", -1)) == 0 and data.get("projection_collision_audit", {}).get("enriched_training_allowed") is True)
    check("coarse_quarantined", data.get("projection_collision_audit", {}).get("coarse_training_allowed") is False and data.get("training_contract", {}).get("coarse_collision_records_are_diagnostic_only") is True)
    forbidden = ("oracle", "payload", "response_body", "body_sha", "typed_positive", "mode=")
    all_contexts = [row[field] for row in rows for field in ("pre_question_context_tokens", "coarse_post_context_tokens", "enriched_post_context_tokens")]
    check("context_firewall", all(not any(term in token.casefold() for term in forbidden) for context in all_contexts for token in context))
    check("question_state_present", all(any(token.startswith("unknown=") for token in row["pre_question_context_tokens"]) and any(token.startswith("observe_candidate_channel=") for token in row["enriched_post_context_tokens"]) for row in rows))
    check("evidence_complete", all(len(str(row.get("evidence_hash", ""))) == 64 for row in rows))
    check("raw_off", all(row.get("raw_payload_strings_stored") is False and row.get("raw_response_bodies_stored") is False and row.get("oracle_in_context") is False for row in rows))
    check("promotion_blocked", data.get("training_contract", {}).get("promotion_blocked") is True and data.get("training_contract", {}).get("memory_promotion_blocked") is True)
    audit = {
        "audit_id": "pg277-counterfactual-question-dataset-audit-v1",
        "status": "passed" if not failures else "failed",
        "audit_checks": {name: name not in failures for name in ("dataset_hash", "fixture_hash", "split_counts", "split_disjoint", "positive_support", "matched_shape_conflicts_exist", "enriched_conflicts_resolved", "coarse_quarantined", "context_firewall", "question_state_present", "evidence_complete", "raw_off", "promotion_blocked")},
        "counts": {"train": len(train), "holdout": len(holdout)},
        "coarse_collision": coarse,
        "enriched_collision": enriched,
        "failures": failures,
        "interpretation": "Coarse shape tokens are provably insufficient because identical projections map to conflicting labels. Marker-channel observation resolves the collision and is eligible for the model ablation; coarse records remain diagnostic-only.",
    }
    audit["audit_sha256"] = sha(audit)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
