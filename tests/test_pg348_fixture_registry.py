from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"challenge_id", "local_path", "mechanism_id", "surface_template_id", "implementation_group", "transport_method", "parameter_role", "encoding_chain", "response_shape", "redirect_shape", "script_surface", "synthetic_oracle_kind", "source_hash"}
FORBIDDEN_CONTEXT_KEYS = {"payload", "raw_payload", "response_body", "oracle_answer", "evaluator_answer", "credentials"}


def test_pg348_registry_has_500_distinct_local_instances() -> None:
    data = json.loads((ROOT / "fixtures" / "pg348" / "registry_v1.json").read_text(encoding="utf-8"))
    rows = data["records"]
    assert len(rows) >= 500
    assert len({row["challenge_id"] for row in rows}) == len(rows)
    assert len({row["source_hash"] for row in rows}) == len(rows)
    assert data["counts"]["mechanism_families"] >= 20
    assert data["counts"]["surface_templates"] >= 10
    assert data["counts"]["transport_variants"] >= 4
    assert data["counts"]["get_count"] > 0 and data["counts"]["post_count"] > 0
    assert data["counts"]["all_urls_loopback_only"] is True
    assert data["counts"]["external_network_records"] == 0
    assert data["counts"]["state_write_records"] == 0


def test_pg348_records_have_required_fields_and_closed_promotion() -> None:
    data = json.loads((ROOT / "fixtures" / "pg348" / "registry_v1.json").read_text(encoding="utf-8"))
    assert all(REQUIRED <= set(row) for row in data["records"])
    assert all(row.get("raw_source_for_evaluator_only") is True and row.get("training_context_raw") is False for row in data["records"])
    assert all(row.get("training_eligible") is False for row in data["records"])
    assert not any(key in row for row in data["records"] for key in FORBIDDEN_CONTEXT_KEYS)
    assert all(value is False for value in data["promotion"].values())
