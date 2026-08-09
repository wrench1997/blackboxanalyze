from __future__ import annotations

import json

from scripts.build_pg387_ctf_frontend_context_dataset import build_dataset


def test_pg387_dataset_has_train_holdout_and_role_matrix() -> None:
    dataset = build_dataset()
    assert dataset["status"] == "abstract_ctf_candidate_only"
    assert dataset["counts"] == {
        "records": 512,
        "train": 256,
        "implementation_holdout": 256,
        "cases": 16,
        "implementations": 2,
        "seeds": 4,
        "roles": 4,
        "get": 0,
        "typed_evaluator_observed": 0,
        "training_eligible": 0,
        "GET": 256,
        "POST": 256,
    }
    assert len({row["record_ref_sha256"] for row in dataset["rows"]}) == 512
    assert all(row["training_eligible"] is False for row in dataset["rows"])


def test_pg387_dataset_has_no_raw_source_or_wire_material() -> None:
    text = json.dumps(build_dataset(), ensure_ascii=False, sort_keys=True)
    for marker in ("http://", "https://", "wire=", "response_body=", "<script", "javascript:", "payload="):
        assert marker not in text
    assert '"raw_probe": false' in text
    assert '"source_text_stored": false' in text


def test_pg387_dataset_carries_semantic_js_overlay_tokens() -> None:
    dataset = build_dataset()
    row = next(item for item in dataset["rows"] if "client_normalizer_order" in " ".join(item["context_tokens"]))
    tokens = set(row["context_tokens"])
    assert "js_source=location_search" in tokens
    assert "js_parser=url_search_params" in tokens
    assert "js_filter_shape=blocklist_or_regex" in tokens
    assert "js_guard_shape=conditional" in tokens
    assert row["javascript_context"]["source_text_stored"] is False
