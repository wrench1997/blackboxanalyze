# -*- coding: utf-8 -*-
"""Convert PG-266 grounded replay into abstract training records.

The human catalog contains exact local-lab wires.  This converter is the
information firewall: only surface/method/Rule-IR/outcome tokens and hashes
enter the next-token/auxiliary-head training dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "research" / "pg266_pikachu_payload_grounding_catalog_v1.json"
DATASET = ROOT / "research" / "pg267_payload_grounding_augmented_dataset_v1.json"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _rule_token(route: dict[str, Any]) -> tuple[str, str]:
    route_id = str(route.get("id", ""))
    family = str(route.get("family", ""))
    if route_id == "sql-blind-boolean-get":
        return "blind_boolean", "sql_boolean"
    if route_id == "sql-widebyte-post":
        return "widebyte_escape_boundary", "sql_widebyte"
    if family == "sql":
        return "syntax_boundary", "sql_syntax"
    if family == "xss":
        return "dom_marker", "dom_marker"
    return "oracle_gap", "oracle_gap"


def _record(entry: dict[str, Any], index: int) -> dict[str, Any]:
    route = dict(entry.get("route") or {})
    oracle = dict(entry.get("oracle") or {})
    confirmed = bool(oracle.get("confirmed_positive"))
    family = str(route.get("family", "other"))
    method = str(route.get("method", "GET")).upper()
    rule_token, rule_class = _rule_token(route)
    outcome = "typed_effect" if confirmed else "oracle_gap"
    lane = "gold" if confirmed else "hard_negative"
    fields = list(route.get("fields") or [])
    tokens = [
        "[BOS]",
        "phase=observe",
        f"surface={family}_payload_surface",
        f"method={method}",
        f"field_bucket={min(len(fields), 8)}",
        f"channel={'query' if method == 'GET' else 'form'}",
        "candidate_sent=1",
        "reference_sent=1",
        "negative_sent=1",
        "fresh_reset=1",
        "source_attested=1",
        f"rule_ir={rule_token}",
        f"oracle={str(route.get('oracle', 'typed_local_effect'))}",
        f"candidate_variant={str((entry.get('ai') or {}).get('variant', 'candidate'))}",
        f"outcome={outcome}",
        f"negative_clean={int(not bool((entry.get('negative') or {}).get('browser_oracle', {}).get('executed')))}",
        f"next_action={'replay_confirmed' if confirmed else 'abstain_or_repair'}",
        "phase=diagnose",
        f"family={family}",
        f"lane={lane}",
        "phase=replay",
        f"replay_expected={'typed' if confirmed else 'abstain'}",
        "[EOS]",
    ]
    classification_position = tokens.index(f"rule_ir={rule_token}")
    source = "pg267_pg266_payload_grounding"
    seed = 26701 + index
    return {
        "source": source,
        "seed": seed,
        "route": str(route.get("path", "")),
        "surface_class": f"{family}_payload_surface",
        "method": method,
        "lane": lane,
        "lane_index": index,
        "repair_action": "replay_confirmed" if confirmed else "abstain",
        "repair_index": 0,
        "failure_kind": "typed_effect" if confirmed else "oracle_gap",
        "replay_expected": "typed" if confirmed else "abstain",
        "classification_position": classification_position,
        "tokens": tokens,
        "trajectory_hash": _digest({"tokens": tokens, "route": route.get("path"), "seed": seed}),
        "quality_reasons": ["pg266_ai_reference_negative_fresh_oracle_complete", "raw_payload_excluded"],
        "source_evidence_hash": str(oracle.get("evidence_hash", "")),
        "route_source_sha256": str((entry.get("source") or {}).get("source_sha256", "")),
        "model_self_error_detected": False,
        "payload_grounded_eligible": True,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "next_rule_class": rule_token,
        "family_class": "dom" if family == "xss" else "sql" if family == "sql" else "other",
        "channel_class": "query" if method == "GET" else "form",
        "pair_role": "single",
        "source_role": "observed",
        "source_lane": "pg266_audited_local",
        "rule_ir_class": rule_class,
        "source_record_id": str(entry.get("record_id", f"pg266-{index}")),
        "belief_class": "confirmed_effect" if confirmed else "needs_reference",
        "probe_class": "replay_confirm" if confirmed else "negative_control",
        "unknown_abstain_class": "continue_family" if confirmed else "abstain_unknown",
    }


def main() -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    if catalog.get("status") != "completed_human_review_catalog":
        raise RuntimeError("PG-266 catalog is not complete")
    entries = [row for row in list(catalog.get("entries") or []) if isinstance(row, dict)]
    records = [_record(row, index) for index, row in enumerate(entries)]
    dataset = {
        "schema_version": "pg267-payload-grounding-augmented-dataset-v1",
        "source_catalog": str(CATALOG.relative_to(ROOT)),
        "source_catalog_sha256": str(catalog.get("catalog_sha256", "")),
        "records": records,
        "counts": {
            "records": len(records),
            "gold": sum(int(row["lane"] == "gold") for row in records),
            "hard_negative": sum(int(row["lane"] == "hard_negative") for row in records),
            "get": sum(int(row["method"] == "GET") for row in records),
            "post": sum(int(row["method"] == "POST") for row in records),
        },
        "contract": {
            "surface_method_field_tokens": True,
            "rule_ir_and_failure_tokens": True,
            "payload_strings_excluded": True,
            "response_bodies_excluded": True,
            "oracle_target_off_input": True,
            "fresh_reset_required": True,
            "source_evidence_hash_required": True,
            "training_promotion_allowed": False,
            "memory_promotion_allowed": False,
            "vulnerability_claim_allowed": False,
        },
        "dataset_sha256": "",
    }
    dataset["dataset_sha256"] = _digest(dataset)
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed_pg267_dataset_build", "counts": dataset["counts"], "dataset": str(DATASET.relative_to(ROOT)), "dataset_sha256": dataset["dataset_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
