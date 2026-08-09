from __future__ import annotations

import json

from app.pg389_js_chain_projection import CHAIN_CASES, CHAIN_VARIANTS, project_chain_case, project_js_chain_source
from scripts.audit_pg389_js_chain_dataset import audit_dataset
from scripts.build_pg389_js_chain_dataset import build_dataset


def test_pg389_case_projection_contains_ordered_chain_and_no_raw_material() -> None:
    projection = project_chain_case({"case_ref": "double_decode_order"})
    assert projection["context_tokens"][0] == "[BOS]"
    assert "decoder_step_2=percent_decode" in projection["context_tokens"]
    assert "decoder_step_3=percent_decode" in projection["context_tokens"]
    assert projection["target_tokens"][0] == "[TARGET_BEGIN]"
    assert projection["source_text_stored"] is False
    text = json.dumps(projection, ensure_ascii=False).casefold()
    for marker in ("http://", "https://", "wire=", "payload=", "response_body=", "<script", "javascript:"):
        assert marker not in text


def test_pg389_source_projection_orders_js_steps_and_blocks_persistence() -> None:
    source = """
    const raw = new URLSearchParams(location.search).get('q');
    const decoded = decodeURIComponent(raw || '').trim();
    if (decoded.includes('safe')) preview.textContent = decoded;
    """
    projected = project_js_chain_source(source)
    assert projected["decode_filter_context"]["decoder_chain"][:4] == ["query_parse", "field_extract", "percent_decode", "trim"]
    assert projected["decode_filter_context"]["filter_stage"] == "guard_or_filter_observed"
    assert projected["safe_to_send"] is True
    assert source not in json.dumps(projected, ensure_ascii=False)


def test_pg389_unsafe_source_fails_closed() -> None:
    projected = project_js_chain_source("const value = location.hash; localStorage.setItem('x', value); eval(value);")
    assert projected["safe_to_send"] is False
    assert "next_action=ask" in projected["target_tokens"]
    assert projected["javascript_surface"]["persistent_write"] is True


def test_pg389_dataset_has_split_isolation_and_expected_matrix() -> None:
    dataset = build_dataset()
    assert dataset["counts"] == {
        "records": 288,
        "train": 144,
        "implementation_holdout": 144,
        "cases": 12,
        "implementations": 2,
        "seeds": 3,
        "roles": 4,
        "GET": 120,
        "POST": 168,
        "typed_evaluator_observed": 0,
        "training_eligible": 0,
    }
    assert len({row["record_ref_sha256"] for row in dataset["rows"]}) == 288
    assert all(row["training_eligible"] is False for row in dataset["rows"])


def test_pg389_audit_passes_candidate_contract() -> None:
    report = audit_dataset(build_dataset())
    assert report["status"] == "passed_candidate_audit"
    assert report["failures"] == []
    assert report["split_isolation"] == {"cross_split_context_overlap": 0, "cross_split_context_target_overlap": 0}
    assert report["coverage"]["decoder_chain"]["unique"] >= 8


def test_pg389_variants_are_explicit() -> None:
    assert CHAIN_VARIANTS == ("chain_order_a", "chain_order_b")
    assert len(CHAIN_CASES) == 12
    assert all(project_chain_case(case, variant=variant)["chain_variant"] == variant for case in CHAIN_CASES for variant in CHAIN_VARIANTS)
