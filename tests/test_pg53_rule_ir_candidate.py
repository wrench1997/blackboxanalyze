import json
from pathlib import Path

from app.pg53_rule_ir_candidate import PG53_MODEL_FAMILIES, abstract_pg53_rule_ir
from app.rule_ir_decoder import validate_abstract_rule_ir


REPORT = Path("research/pg53_rule_ir_candidate_report_v1.json")
CHECKPOINT = Path("artifacts/pg53-rule-ir-candidate/decoder.pt")


def _load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_pg53_candidate_uses_source_holdout_and_is_quarantined():
    report = _load()
    funnel = json.loads(Path("research/pg53_web_feature_funnel_dataset_v1.json").read_text(encoding="utf-8"))
    assert report["training"]["source"] == "pg35"
    assert report["training"]["holdout_source"] == "pg36"
    assert report["training"]["train_rows"] == 72
    assert report["training"]["dev_rows"] == 36
    assert report["training"]["holdout_rows"] == 108
    assert report["training"]["oracle_in_features"] is False
    assert report["training"]["selected_features"] == funnel["accepted_features"]
    assert set(report["training"]["selected_features"]) == {
        "geometry_change_presence_control",
        "geometry_true_boolean_delta_ratio_control",
        "geometry_array_item_count",
        "geometry_nonzero_numeric_count",
        "geometry_numeric_count",
        "geometry_array_count",
    }
    assert report["holdout"]["calibrated"]["false_accept_count"] == 0
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["formal_claim_allowed"] is False
    assert CHECKPOINT.exists()


def test_pg53_candidate_emits_only_grammar_checked_non_executable_templates():
    assert set(PG53_MODEL_FAMILIES) == {
        "xss",
        "injection",
        "authentication",
        "access_control",
        "logic",
        "url_redirect",
        "input_validation",
        "command_injection",
        "ordinary_response",
    }
    for family in PG53_MODEL_FAMILIES:
        rule = abstract_pg53_rule_ir(family)
        if rule is not None:
            validate_abstract_rule_ir(rule)


def test_pg53_candidate_report_has_no_raw_attack_material():
    text = REPORT.read_text(encoding="utf-8").casefold()
    for forbidden in ("<svg", "onload", "union select", "password", "pg52missing"):
        assert forbidden not in text
