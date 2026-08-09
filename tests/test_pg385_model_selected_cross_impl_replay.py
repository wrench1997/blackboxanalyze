from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_pg385_model_selected_cross_impl_replay import run_cross_impl


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research/pg385_model_selected_cross_impl_replay_v1.json"
CHECKPOINT = ROOT / "artifacts/pg385-variant-selector/pg385_variant_seed_38503.pt"
NODE_FIXTURE = ROOT / "fixtures/pg385/impl_b/server.js"


def test_cross_impl_report_is_loopback_only_and_promotion_closed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "completed_model_selected_cross_implementation_loopback_only"
    assert report["counts"] == {
        "rows": 4,
        "implementations": 2,
        "methods_get": 2,
        "methods_post": 2,
        "model_variant_selected": 4,
        "candidate_typed": 4,
        "reference_typed": 4,
        "negative_violation": 0,
        "replay_typed": 4,
    }
    assert report["execution"] == {
        "external_network": False,
        "docker_started": False,
        "target_contacted": True,
        "loopback_only": True,
        "raw_wire_stored": False,
    }
    assert all(value is False for value in report["promotion"].values())
    assert report["model_boundary"]["model_emits_raw_string"] is False
    assert report["model_boundary"]["evaluator_last_hop_canary_binding"] is True


@pytest.mark.skipif(not CHECKPOINT.exists() or not NODE_FIXTURE.exists(), reason="PG-385 selector checkpoint or Node fixture is not present")
def test_cross_impl_replay_runs_model_selected_get_post_matrix() -> None:
    report, wires = run_cross_impl(checkpoint=CHECKPOINT, show_wire=True)
    assert report["status"] == "completed_model_selected_cross_implementation_loopback_only"
    assert report["counts"]["implementations"] == 2
    assert report["counts"]["methods_get"] == 2
    assert report["counts"]["methods_post"] == 2
    assert report["counts"]["negative_violation"] == 0
    assert report["counts"]["candidate_typed"] == 4
    assert report["counts"]["reference_typed"] == 4
    assert report["counts"]["replay_typed"] == 4
    assert len(wires) == 20  # baseline + candidate/reference/negative/replay per row
    assert report["execution"]["docker_started"] is False
    assert report["execution"]["external_network"] is False
