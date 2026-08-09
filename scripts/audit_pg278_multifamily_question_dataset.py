"""Independent structural/data-lineage audit for the PG-278 collection set."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "research" / "pg278_multifamily_question_dataset_v1.json"
OUTPUT = ROOT / "research" / "pg278_multifamily_question_dataset_audit_v1.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def check(name: str, condition: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def target_signature(row: dict[str, Any]) -> str:
    return sha(dict(row["targets"]["pre_question"]))


def collision(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        key = sha(row[field])
        counts[key] += 1
        grouped[key].add(target_signature(row))
    conflicts = [key for key, values in grouped.items() if len(values) > 1]
    return {"groups": len(grouped), "conflict_groups": len(conflicts), "conflicting_rows": sum(counts[key] for key in conflicts)}


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = list(data.get("records") or [])
    recompute = dict(data)
    claimed_hash = str(recompute.pop("dataset_sha256", ""))
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family_rows[str(row.get("family", ""))].append(row)
    pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_rows[str(row.get("pair_id", ""))].append(row)
    coarse = collision(rows, "coarse_pre_question_context_tokens")
    enriched = collision(rows, "pre_question_context_tokens")
    post = collision(rows, "post_observation_context_tokens")
    prohibited = ("expected_positive=", "oracle=", "typed_positive=", "raw_payload=", "raw_response=", "case=")
    leaked_tokens = [token for row in rows for field in ("pre_question_context_tokens", "post_observation_context_tokens") for token in row.get(field, []) if any(str(token).startswith(prefix) for prefix in prohibited)]
    checks = [
        check("schema", data.get("schema_version") == "pg278-multifamily-question-dataset-v1", data.get("schema_version")),
        check("dataset_hash", claimed_hash == sha(recompute), {"claimed": claimed_hash, "computed": sha(recompute)}),
        check("record_total", len(rows) == 288, len(rows)),
        check("split_counts", sum(row.get("split") == "implementation_train" for row in rows) == 192 and sum(row.get("split") == "implementation_holdout" for row in rows) == 96, {"train": sum(row.get("split") == "implementation_train" for row in rows), "holdout": sum(row.get("split") == "implementation_holdout" for row in rows)}),
        check("four_families", set(family_rows) == {"dom_effect", "sql_differential", "redirect_contract", "logic_access"}, sorted(family_rows)),
        check("family_quota", all(len(items) == 72 and sum(bool(item.get("labels", {}).get("expected_positive")) for item in items) >= 24 and sum(not bool(item.get("labels", {}).get("expected_positive")) for item in items) >= 12 for items in family_rows.values()), {name: {"rows": len(items), "positive": sum(bool(item.get("labels", {}).get("expected_positive")) for item in items), "negative": sum(not bool(item.get("labels", {}).get("expected_positive")) for item in items)} for name, items in family_rows.items()}),
        check("implementation_seed_encoding_contract", all(len({row.get("implementation") for row in items}) == 3 and all(len({row.get("collection_seed") for row in items if row.get("implementation") == implementation}) == 3 for implementation in {row.get("implementation") for row in items}) and all(len({row.get("encoding") for row in items if row.get("implementation") == implementation and row.get("collection_seed") == seed}) == 2 for implementation in {row.get("implementation") for row in items} for seed in {row.get("collection_seed") for row in items if row.get("implementation") == implementation}) for items in family_rows.values()), {name: {"implementations": sorted({str(row.get("implementation")) for row in items}), "seeds": sorted({int(row.get("collection_seed")) for row in items}), "encodings": sorted({str(row.get("encoding")) for row in items})} for name, items in family_rows.items()}),
        check("eight_missing_slots", len({row.get("missing_observation_slot") for row in rows}) == 8 and all(sum(row.get("missing_observation_slot") == slot for row in rows) == 36 for slot in {row.get("missing_observation_slot") for row in rows}), Counter(str(row.get("missing_observation_slot")) for row in rows)),
        check("pair_contract", all(len(items) == 2 and {bool(item.get("labels", {}).get("expected_positive")) for item in items} == {False, True} and all(str(item.get("paired_opposite_record_id")) in {str(other.get("record_id")) for other in items if other is not item} for item in items) for items in pair_rows.values()), {"pairs": len(pair_rows), "invalid": sum(not (len(items) == 2 and {bool(item.get("labels", {}).get("expected_positive")) for item in items} == {False, True}) for items in pair_rows.values())}),
        check("fresh_replay_lineage", all(int(row.get("request_projection", {}).get("replay_count", 0)) == 2 and len(row.get("source", {}).get("fresh_replay_evidence_hashes") or []) == 2 and len(set(row.get("source", {}).get("fresh_replay_evidence_hashes") or [])) == 1 for row in rows), {"rows_with_two_hashes": sum(len(row.get("source", {}).get("fresh_replay_evidence_hashes") or []) == 2 for row in rows)}),
        check("complete_context_and_rejections", all(row.get("pre_question_context_tokens") and row.get("post_observation_context_tokens") and row.get("evidence_hash") and row.get("source", {}).get("source_evidence_hash") and len(row.get("preference_rejected", {}).get("pre_question") or []) >= 2 for row in rows), {"incomplete": sum(not (row.get("pre_question_context_tokens") and row.get("post_observation_context_tokens") and row.get("evidence_hash") and row.get("source", {}).get("source_evidence_hash") and len(row.get("preference_rejected", {}).get("pre_question") or []) >= 2) for row in rows)}),
        check("logic_abstract_request_condition", all(dict(row.get("request_projection", {}).get("abstract_condition") or {}).get("route_role") == "protected_access" for row in family_rows.get("logic_access", [])), {"logic_rows": len(family_rows.get("logic_access", [])), "with_condition": sum(bool(dict(row.get("request_projection", {}).get("abstract_condition") or {})) for row in family_rows.get("logic_access", []))}),
        check("no_raw_oracle_leak", not leaked_tokens and all(not row.get("raw_payload_strings_stored") and not row.get("raw_response_bodies_stored") and not row.get("oracle_in_context") for row in rows), {"leaked_token_count": len(leaked_tokens), "sample": leaked_tokens[:8]}),
        check("controlled_collision_reproduced", coarse["conflict_groups"] > 0 and coarse["conflicting_rows"] == len(rows), coarse),
        check("enriched_collision_eliminated", enriched["conflict_groups"] == 0, enriched),
        check("post_observation_collision_eliminated", post["conflict_groups"] == 0, post),
        check("no_memory_promotion", not any(bool(row.get("memory_promotion_allowed")) for row in rows) and int(data.get("source", {}).get("real_multifamily_gold_rows", -1)) == 0, {"real_multifamily_gold_rows": data.get("source", {}).get("real_multifamily_gold_rows"), "promotion_values": sorted({bool(row.get("memory_promotion_allowed")) for row in rows})}),
    ]
    passed = all(item["passed"] for item in checks)
    audit = {
        "schema_version": "pg278-multifamily-question-dataset-audit-v1",
        "status": "passed" if passed else "failed",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": claimed_hash,
        "checks": checks,
        "independent_collision_recalculation": {"coarse": coarse, "enriched": enriched, "post": post},
        "scope": {"controlled_loopback_only": True, "real_multifamily_gold_rows": 0, "training_use": "controlled_research_only", "promotion": "blocked"},
        "conclusion": "The collection contract is structurally satisfied and the intended missing-slot collision is reproduced. This proves data lineage and a controlled slot-binding setup only; it does not prove real-application vulnerability discovery.",
    }
    audit["audit_sha256"] = sha(audit)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "audit": str(OUTPUT.relative_to(ROOT)), "audit_sha256": audit["audit_sha256"], "failed": [item["name"] for item in checks if not item["passed"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
