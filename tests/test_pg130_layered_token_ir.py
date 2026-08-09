from __future__ import annotations

import json
from pathlib import Path


def test_pg130_layered_token_ir_smoke_is_engineering_only() -> None:
    report = json.loads(Path("research/pg130_layered_token_ir_report_v1.json").read_text(encoding="utf-8"))
    assert report["audited_steps"] == 72
    assert report["get_post_covered"] is True
    assert report["all_steps_validated"] is True
    assert report["failure_adjusted_steps"] > 0
    assert report["forward_baseline_steps"] > 0
    assert report["raw_source_saved"] is False
    assert report["raw_javascript_saved"] is False
    assert report["training_eligible"] is False
    assert report["memory_promotion_allowed"] is False
