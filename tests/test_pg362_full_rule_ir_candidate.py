from __future__ import annotations

from scripts import run_pg362_a800_full_rule_ir_candidate as runner


def test_pg362_uses_append_only_pg361_slot_order() -> None:
    assert runner.BASE.SCHEMA_VERSION == "pg362-a800-full-rule-ir-candidate-v1"
    assert runner.BASE.SEEDS == (36201, 36202, 36203)
    assert runner.BASE.TARGET_KEY_ORDER[-5:] == (
        "probe_variant_ref",
        "safe_to_send",
        "payload_shape_ref",
        "oracle_ref",
        "negative_control_presence_ref",
    )


def test_pg362_keeps_raw_fragments_out_of_the_target_prefix_contract() -> None:
    assert "syntax_category_ref=" in runner.BASE.TARGET_PREFIXES
    assert all(not token.startswith("payload=") for token in runner.BASE.TARGET_PREFIXES)
