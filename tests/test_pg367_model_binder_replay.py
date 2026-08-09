from __future__ import annotations

from scripts.run_pg367_model_binder_replay import _normalize_rule_ir, _parse_target


def test_model_target_parser_requires_full_abstract_slots() -> None:
    target = [
        "[TARGET_BOS]",
        "transport_ref=get_query",
        "field_role_ref=query_term",
        "encoding_ref=url_percent",
        "probe_variant_ref=source_attested_candidate",
        "safe_to_send=1",
        "payload_shape_ref=query_marker",
        "oracle_ref=typed_effect",
        "syntax_category_ref=delimiter_boundary",
        "[TARGET_EOS]",
    ]
    slots = _parse_target(target)
    assert slots is not None
    rule = _normalize_rule_ir(slots)
    assert rule is not None
    assert rule["transport_ref"] == "get_query"


def test_model_raw_slot_is_not_bindable() -> None:
    slots = _parse_target([
        "[TARGET_BOS]", "transport_ref=get_query", "field_role_ref=query_term",
        "encoding_ref=url_percent", "probe_variant_ref=source_attested_candidate",
        "safe_to_send=1", "payload_shape_ref=raw_payload", "oracle_ref=typed_effect",
        "syntax_category_ref=marker", "[TARGET_EOS]",
    ])
    assert slots is not None
    assert _normalize_rule_ir(slots) is None
