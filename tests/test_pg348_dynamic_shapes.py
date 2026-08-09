from __future__ import annotations

import json
from pathlib import Path

from scripts.collect_pg348_dynamic_shapes import collect


ROOT = Path(__file__).resolve().parents[1]


def test_pg348_dynamic_shape_trace_has_get_post_and_four_roles() -> None:
    registry = json.loads((ROOT / "fixtures" / "pg348" / "registry_v1.json").read_text(encoding="utf-8"))
    trace = collect(registry)
    assert trace["counts"] == {"rows": 2080, "records": 520, "roles": 4, "get_rows": 1240, "post_rows": 840, "fresh_reset_rows": 2080, "typed_positive": 0, "training_eligible": 0}
    assert all(row["typed_available"] is False and row["training_eligible"] is False for row in trace["records"])
    assert all(row["fresh_reset"] is True and row["persistent_storage"] is False for row in trace["records"])
    assert all("body" not in row and "payload" not in row for row in trace["records"])


def test_pg348_dynamic_shape_trace_keeps_promotion_closed() -> None:
    registry = json.loads((ROOT / "fixtures" / "pg348" / "registry_v1.json").read_text(encoding="utf-8"))
    trace = collect(registry)
    assert trace["status"] == "completed_dynamic_diagnostic_only"
    assert all(value is False for value in trace["promotion"].values())
