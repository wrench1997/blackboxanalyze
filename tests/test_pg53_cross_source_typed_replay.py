import json
import re
from pathlib import Path


REPORT = Path("research/pg53_cross_source_typed_replay_report_v1.json")
PROTOCOL = Path("research/pg53_cross_source_typed_replay_protocol_v1.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pg53_has_cross_source_get_post_typed_positive_and_negative_evidence():
    report = _load(REPORT)
    metrics = report["metrics"]
    assert report["status"] == "completed"
    assert metrics["case_count"] == 270
    assert metrics["confirmed_positive_count"] == 240
    assert metrics["confirmed_negative_count"] == 30
    assert metrics["get_post_covered"] == {"GET": 135, "POST": 135}
    assert metrics["oracle_family_binding_match_count"] == 240
    assert metrics["negative_control_pass_count"] == 270
    assert report["cross_source"]["family_replay_rate"] == 1.0
    assert report["cross_source"]["mismatch_cell_count"] == 0
    assert report["cross_source"]["mismatch_cells"] == []
    assert report["cross_source"]["negative_control_pass_rate"] == 1.0
    assert set(report["by_implementation"]) == {"pg34", "pg35", "pg36"}
    assert report["by_implementation"]["pg34"]["confirmed_positive_count"] == 48
    assert report["by_implementation"]["pg35"]["confirmed_positive_count"] == 96
    assert report["by_implementation"]["pg36"]["confirmed_positive_count"] == 96


def test_pg53_keeps_model_score_separate_and_exposes_current_weakness():
    report = _load(REPORT)
    metrics = report["metrics"]
    # The existing five-family checkpoint is intentionally not treated as an
    # oracle.  A useful experiment must expose its errors instead of hiding
    # them behind the typed label.
    assert metrics["model_candidate_count"] > 0
    assert metrics["model_family_misclassification_count"] > 0
    assert metrics["model_false_positive_count"] > 0
    assert metrics["model_family_match_count"] < metrics["confirmed_positive_count"]
    assert report["model_visibility"]["oracle_visible_before_probe"] is False
    assert report["model_visibility"]["family_label_in_features"] is False
    assert report["model_visibility"]["raw_values_in_features"] is False


def test_pg53_rows_are_fresh_hashed_and_do_not_persist_attack_material():
    report = _load(REPORT)
    rows = report["rows"]
    assert rows
    for row in rows:
        assert row["raw_payload_stored"] is False
        assert row["raw_response_body_stored"] is False
        assert re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"])
        assert row["negative_control"]["matched"] is True
        for reset in row["fresh_reset"].values():
            if reset is None:
                continue
            assert reset["fresh_target"] is True
            assert reset["completed"] is True
            assert reset["state_change_allowed"] is False
            assert reset["external_network"] is False
        assert row["candidate"]["oracle"]["positive"] is (row["decision"] == "confirmed_positive")
        if row["screen"] is not None:
            assert row["screen"]["oracle"]["positive"] is False
    text = json.dumps(report, ensure_ascii=False).casefold()
    for forbidden in ("<svg", "onload", "union select", "pg52missing", "password"):
        assert forbidden not in text
    assert report["training_boundary"]["training_eligible"] is False
    assert report["training_boundary"]["long_term_memory_write"] is False
    assert report["formal_claim"]["allowed"] is False


def test_pg53_protocol_declares_independence_and_evaluator_only_labels():
    protocol = _load(PROTOCOL)
    assert len(protocol["independent_sources"]) == 5
    assert {item["implementation"] for item in protocol["independent_sources"]} == {"pg34", "pg35", "pg36"}
    assert protocol["sampling_seeds"] == [5301, 5307, 5311]
    assert protocol["methods"] == ["GET", "POST"]
    assert protocol["oracle_contract"]["oracle_is_label_not_model_feature"] is True
    assert protocol["promotion_gate"]["training_eligible_before_fresh_holdout"] is False
