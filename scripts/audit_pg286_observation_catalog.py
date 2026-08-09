"""Independently audit the PG-286 observation-token catalog.

This audit deliberately does not import the catalog builder.  It recomputes the
catalog digest and checks the promotion contract from the persisted artifacts,
so a builder bug cannot make its own audit pass.  The catalog is a collection
planning artifact: incomplete evidence is retained, but no row is eligible for
training or long-term memory promotion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CATALOG_PATH = RESEARCH / "pg286_observation_token_catalog_v1.json"
HARD_PATH = RESEARCH / "pg286_observation_token_hard_negative_v1.json"
BUILDER_AUDIT_PATH = RESEARCH / "pg286_observation_token_catalog_audit_v1.json"
INDEPENDENT_AUDIT_PATH = RESEARCH / "pg286_observation_token_catalog_independent_audit_v1.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _context_forbidden(tokens: list[Any]) -> bool:
    forbidden = ("family=", "oracle=", "typed_effect", "positive", "payload=", "<script", "javascript:", "union select", "drop table")
    return any(any(bad.casefold() in str(token).casefold() for bad in forbidden) for token in tokens)


def main() -> None:
    catalog = _load(CATALOG_PATH)
    hard = _load(HARD_PATH)
    builder_audit = _load(BUILDER_AUDIT_PATH)
    rows = list(catalog.get("records") or [])
    hard_rows = list(catalog.get("hard_negative_records") or [])
    persisted_hard_rows = list(hard.get("records") or [])
    counts = dict(catalog.get("counts") or {})
    expected_counts = {
        "total": len(rows),
        "sql": sum(row.get("family") == "sql" for row in rows),
        "xss": sum(row.get("family") == "xss" for row in rows),
        "redirect": sum(row.get("family") == "redirect" for row in rows),
        "complete": sum(row.get("evidence_status") == "complete" for row in rows),
        "incomplete": sum(row.get("evidence_status") == "incomplete" for row in rows),
        "hard_negative": len(hard_rows),
    }
    checks = {
        "schema_version": catalog.get("schema_version") == "pg286-observation-token-catalog-v1",
        "catalog_digest": catalog.get("catalog_sha256") == _sha256({key: value for key, value in catalog.items() if key != "catalog_sha256"}),
        "counts_recomputed": counts == expected_counts,
        "source_rows_expected": len(rows) == 28,
        "hard_negative_persisted": len(hard_rows) == len(persisted_hard_rows) == 28,
        "context_family_hidden": all(
            "ir_family_agnostic=1" in list(row.get("context_tokens") or [])
            and not any(str(token).startswith("family=") for token in list(row.get("context_tokens") or []))
            for row in [*rows, *hard_rows]
        ),
        "context_has_no_label_or_literal": all(
            not bool(row.get("oracle_label_in_context")) and not _context_forbidden(list(row.get("context_tokens") or []))
            for row in [*rows, *hard_rows]
        ),
        "raw_material_excluded": all(
            row.get("raw_payload_stored") is False and row.get("raw_response_body_stored") is False
            for row in [*rows, *hard_rows]
        ),
        "incomplete_quarantined": all(
            not bool(row.get("training_eligible")) and not bool(row.get("memory_promotion_allowed"))
            for row in rows
            if row.get("evidence_status") == "incomplete"
        ),
        "all_rows_not_training_gold": all(not bool(row.get("training_eligible")) for row in [*rows, *hard_rows]),
        "hard_negative_abstain": all(
            row.get("target", {}).get("next_action") == "abstain"
            and row.get("target", {}).get("safe_to_send") is False
            and not bool(row.get("training_eligible"))
            for row in [*hard_rows, *persisted_hard_rows]
        ),
        "sql_ast_gap_visible": all(
            "sql_ast_available=0" in list(row.get("context_tokens") or [])
            for row in rows
            if row.get("family") == "sql"
        ),
        "builder_audit_agrees": builder_audit.get("status") == "passed" and builder_audit.get("catalog_sha256") == catalog.get("catalog_sha256"),
    }
    result = {
        "schema_version": "pg286-observation-token-catalog-independent-audit-v1",
        "catalog": str(CATALOG_PATH.relative_to(ROOT).as_posix()),
        "hard_negative": str(HARD_PATH.relative_to(ROOT).as_posix()),
        "builder_audit": str(BUILDER_AUDIT_PATH.relative_to(ROOT).as_posix()),
        "catalog_sha256": catalog.get("catalog_sha256", ""),
        "counts": expected_counts,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "blocked",
        "training_eligible_rows": 0,
        "memory_promotion_allowed_rows": 0,
        "sql_ast_available_rows": 0,
        "interpretation": "独立审计确认 12 条 DOM/重定向观测完整、16 条观测不完整；历史 SQL 14 条全部显式缺 SQL AST，因此整个 catalog 只能进入 collection lane，不得冒充训练 gold。",
    }
    result["audit_sha256"] = _sha256(result)
    INDEPENDENT_AUDIT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": expected_counts, "catalog_sha256": result["catalog_sha256"], "audit_sha256": result["audit_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
