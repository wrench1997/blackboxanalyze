from __future__ import annotations

import pytest

from app.pg301_payload_assembly import target_map
from app.pg305_live_evaluator import (
    abstract_projection,
    context_tokens,
    missing_question_contexts,
    surface_slots,
    typed_evidence,
)


def test_surface_slots_are_shared_and_method_bound() -> None:
    assert surface_slots("GET") == {
        "surface_method": "GET",
        "surface_field_role": "query_param",
        "surface_encoding": "url_percent",
    }
    assert surface_slots("POST")["surface_field_role"] == "form_field"


def test_missing_question_contexts_keep_unknown_explicit() -> None:
    rows = missing_question_contexts("GET")
    assert len(rows) == 7
    assert any("typed_available=unknown" in row["context_tokens"] for row in rows)
    assert all("route=" not in " ".join(row["context_tokens"]) for row in rows)
    assert target_map(rows[0]["target_tokens"])["safe_to_send"] == "0"
    assert target_map(rows[-1]["target_tokens"])["safe_to_send"] == "1"


def test_context_does_not_store_hidden_labels() -> None:
    tokens = context_tokens(
        "POST",
        typed_available="1",
        feedback_state="negative_control_clear",
        replay_ready="1",
        evidence_present="1",
        negative_control="1",
        fresh_reset="1",
    )
    joined = " ".join(tokens)
    assert "family=" not in joined
    assert "oracle=" not in joined
    assert "payload" not in joined
    assert "surface_method=POST" in joined


def test_context_constructor_rejects_metadata_smuggling() -> None:
    with pytest.raises(ValueError):
        context_tokens("GET", history_action="family=xss")
    with pytest.raises(ValueError):
        context_tokens("GET", failure_class="<script>alert(1)</script>")


def test_evidence_and_projection_are_bounded() -> None:
    projection = abstract_projection({"status_class": "2xx", "body_length": 300, "content_type": "text/html"}, effect_marker="none")
    assert set(projection) == {"status_class", "shape_sha256", "redirect_hops", "backend_observed", "effect_marker"}
    evidence = typed_evidence(
        effect_type="dom_effect",
        typed_effect_confirmed=False,
        negative_control_clean=True,
        reference_agreement=False,
        replay_consistent=False,
        evaluator_id="test",
    )
    assert len(evidence["evidence_sha256"]) == 64
    assert evidence["non_destructive"] is True
