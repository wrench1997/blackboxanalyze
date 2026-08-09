from __future__ import annotations

from pathlib import Path

from scripts.collect_pg361_syntax_typed_rows import collect
from app.pg348_dynamic_runtime import load_registry


def test_live_wrapper_adds_syntax_oracle_and_negative_slots_without_raw():
    registry = load_registry(Path("fixtures/pg348/registry_v1.json"))
    dataset, sidecars, report = collect(registry, max_records=1)
    assert report["status"] == "completed_dynamic_syntax_diagnostic_only"
    assert len(dataset["records"]) == 5
    assert dataset["counts"]["training_eligible_rows"] == 0
    for row in dataset["records"]:
        target = row["target_projection"]
        assert target["syntax_category_ref"] in {
            "marker",
            "delimiter_boundary",
            "structured_value",
            "expression_node",
            "boolean_branch",
            "parser_node",
            "state_transition",
            "redirect_control",
        }
        assert target["oracle_ref"] in {"typed_effect", "negative_no_effect", "unknown"}
        assert target["negative_control_presence_ref"] in {"matched_triplet", "unknown"}
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
    assert sidecars["promotion"]["training_allowed"] is False
