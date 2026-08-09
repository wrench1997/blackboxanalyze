from __future__ import annotations

from scripts.audit_pg366_context_identifiability import audit_document


def _row(context: list[str], target: str) -> dict[str, object]:
    return {
        "context_tokens": context,
        "target_tokens": ["[TARGET_BOS]", f"next_action={target}", "[TARGET_EOS]"],
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "split": "train",
    }


def test_exact_context_shortcut_is_reported_without_rewriting_rows() -> None:
    report = audit_document(
        {"records": [_row(["document_presence=observed", "chunk_digest=a"], "repair"), _row(["document_presence=observed", "chunk_digest=b"], "abstain")]},
        source_path="fixture.json",
        source_sha256="a" * 64,
    )
    assert report["status"] == "diagnostic_shortcut_risk"
    assert report["shortcut_risk"]["exact_context_conditional_entropy_zero"] is True
    assert report["promotion"]["training_allowed"] is False


def test_presence_projection_retains_ambiguity_signal() -> None:
    report = audit_document(
        {"records": [_row(["document_presence=observed", "chunk_digest=a"], "repair"), _row(["document_presence=observed", "chunk_digest=b"], "abstain")]},
        source_path="fixture.json",
        source_sha256="b" * 64,
    )
    exact = report["entropy"]["exact_context"]
    presence = report["entropy"]["presence_only"]
    assert exact["conditional_target_entropy_bits"] == 0.0
    assert presence["conditional_target_entropy_bits"] > 0.0


def test_raw_context_is_rejected() -> None:
    try:
        audit_document(
            {"records": [_row(["document_presence=observed", "payload=forbidden"], "repair")]},
            source_path="fixture.json",
            source_sha256="c" * 64,
        )
    except ValueError as error:
        assert str(error) == "no_valid_rows"
    else:  # pragma: no cover - defensive
        raise AssertionError("raw context must fail closed")
