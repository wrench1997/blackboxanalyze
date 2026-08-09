from __future__ import annotations

import hashlib

import pytest

from scripts.run_pg384_model_selected_binder_replay import (
    _catalog,
    _scrub,
    _ROLE_VARIANT,
    normalize_rule_ir,
)


def _safe_slots() -> dict[str, str]:
    return {
        "question": "none",
        "ask_reason": "none",
        "next_action": "select_probe_variant",
        "repair_action": "none",
        "transport_ref": "get_query",
        "field_role_ref": "display_text",
        "encoding_ref": "identity",
        "syntax_category_ref": "marker",
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": "1",
        "payload_shape_ref": "html_text_marker",
        "oracle_ref": "reflection",
        "negative_control_presence_ref": "unknown",
    }


def test_normalize_rule_ir_only_allows_abstract_safe_slots() -> None:
    normalized = normalize_rule_ir(_safe_slots())
    assert normalized is not None
    assert normalized["safe_to_send"] is True
    assert "raw_value" not in normalized
    assert normalize_rule_ir({**_safe_slots(), "safe_to_send": "false"}) is None
    assert normalize_rule_ir({**_safe_slots(), "probe_variant_ref": "arbitrary_literal"}) is None


def test_catalog_is_reviewed_placeholder_and_hashes_match() -> None:
    catalog = _catalog(_safe_slots(), "pg384_test")
    entry = catalog["templates"][0]
    assert entry["template"] == "{{MARKER}}"
    assert entry["template_sha256"] == hashlib.sha256(b"{{MARKER}}").hexdigest()
    assert entry["local_only"] is True
    assert entry["non_destructive"] is True


def test_role_variants_are_bounded() -> None:
    assert set(_ROLE_VARIANT) == {"candidate", "reference", "negative", "replay"}
    assert _ROLE_VARIANT["negative"] == "unsupported_variant"


def test_scrub_rejects_wire_or_raw_keys() -> None:
    _scrub({"abstract": "payload_shape_ref=html_text_marker"})
    with pytest.raises(ValueError):
        _scrub({"body": "should never persist"})
    with pytest.raises(ValueError):
        _scrub({"wire": "GET /"})

