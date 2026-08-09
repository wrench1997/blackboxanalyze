import pytest

from app.regex_evidence_oracle import evaluate_allowlisted_regex


def test_regex_evidence_is_bounded_and_does_not_store_raw_text():
    evidence = evaluate_allowlisted_regex(
        text="PG25_CANARY_123",
        pattern_id="escaped_marker_reflection",
        marker="PG25_CANARY_123",
    )
    assert evidence["matched"] is True
    assert evidence["match_count"] == 1
    assert evidence["raw_text_stored"] is False
    assert evidence["capture_groups_stored"] is False
    assert evidence["evidence_hash"]


def test_regex_evidence_rejects_unallowlisted_patterns_and_unbounded_marker():
    with pytest.raises(ValueError, match="allow-listed"):
        evaluate_allowlisted_regex(text="anything", pattern_id="arbitrary_regex")
    with pytest.raises(ValueError, match="bounded marker"):
        evaluate_allowlisted_regex(text="anything", pattern_id="escaped_marker_reflection", marker="not safe!")
