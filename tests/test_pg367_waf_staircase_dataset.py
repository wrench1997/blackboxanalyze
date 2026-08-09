from __future__ import annotations

from scripts.build_pg367_waf_staircase_dataset import build


def test_dataset_contains_get_post_failure_and_repair_rows() -> None:
    document = build()
    counts = document["counts"]
    assert counts["records"] > 0
    assert counts["get_rows"] > 0
    assert counts["post_rows"] > 0
    assert counts["failure_rows"] > 0
    assert counts["repair_rows"] > 0
    assert document["promotion"]["training_allowed"] is False


def test_context_has_waf_tokens_but_no_raw_fields() -> None:
    document = build()
    for row in document["records"]:
        assert any(token.startswith("waf_") for token in row["context_tokens"])
        assert row["context_firewall"] == {"forbidden_token_count": 0, "sidecars_off_context": True}
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert "payload=" not in " ".join(row["context_tokens"])


def test_each_repair_row_records_one_axis_transition() -> None:
    document = build()
    repair_rows = [row for row in document["records"] if "failure_transition" in row]
    assert repair_rows
    assert all(row["failure_transition"]["changed_axis"] in {"encoding", "syntax", "shape"} for row in repair_rows)
