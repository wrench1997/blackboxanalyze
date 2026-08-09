from __future__ import annotations

from scripts.audit_pg367_waf_staircase import audit
from scripts.build_pg367_waf_staircase_dataset import build


def test_waf_audit_passes_but_never_promotes() -> None:
    result = audit(build())
    assert result["status"] == "passed_diagnostic_only"
    assert result["counts"]["get_rows"] > 0
    assert result["counts"]["post_rows"] > 0
    assert result["counts"]["repair_action_changed"] == result["counts"]["repair_rows"]
    assert result["counts"]["negative_clean_rows"] == result["counts"]["negative_rows"]
    assert result["promotion"]["training_allowed"] is False


def test_waf_audit_blocks_raw_context() -> None:
    document = build()
    document["records"][0]["context_tokens"].append("payload=raw")
    result = audit(document)
    assert result["status"] == "blocked_incomplete"
    assert result["counts"]["raw_hits"] == 1
