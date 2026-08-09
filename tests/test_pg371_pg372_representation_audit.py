import json
from pathlib import Path

from scripts.audit_pg371_representation_entropy import audit
from scripts.build_pg372_failure_repair_dataset import SLOTS, build

ROOT = Path(__file__).resolve().parents[1]


def _row(split: str, *, failure: bool, repair: str = "none", syntax: str = "marker") -> dict:
    context = ["document_presence=observed", "dom_tag=form", "javascript_presence=observed"]
    if failure:
        context += ["failure_signature=blocked_pattern", "failure_process_step=repair"]
    values = {
        "question": "ask_failure" if failure else "none",
        "ask_reason": "failure_feedback" if failure else "none",
        "next_action": "repair" if failure else "select_probe_variant",
        "repair_action": repair,
        "transport_ref": "get_query",
        "field_role_ref": "query_term",
        "encoding_ref": "identity",
        "syntax_category_ref": syntax,
        "probe_variant_ref": "source_attested_candidate",
        "safe_to_send": "0" if failure else "1",
        "payload_shape_ref": "query_marker",
        "oracle_ref": "unknown" if failure else "typed_effect",
        "negative_control_presence_ref": "matched_triplet",
    }
    target = ["[TARGET_BOS]", *[f"{slot}={values[slot]}" for slot in SLOTS], "[TARGET_EOS]"]
    return {
        "split": split,
        "context_tokens": context,
        "target_tokens": target,
        "context_firewall": {"forbidden_token_count": 0, "sidecars_off_context": True},
        "raw_payload_stored": False,
        "raw_response_body_stored": False,
        "oracle_answer_in_context": False,
    }


def test_pg371_real_audit_blocks_random_baseline_and_holdout_gap_without_rows() -> None:
    datasets = {
        "pg362": json.loads((ROOT / "research/pg362_full_rule_ir_dataset_v1.json").read_text(encoding="utf-8-sig")),
        "pg367": json.loads((ROOT / "research/pg367_waf_staircase_dataset_v2.json").read_text(encoding="utf-8-sig")),
    }
    candidate = json.loads((ROOT / "research/pg370_multitask_moe_candidate_v1.json").read_text(encoding="utf-8-sig"))
    report = audit(datasets, candidate)
    assert report["status"] == "blocked_entropy_or_leakage"
    assert report["entropy_baseline"]["trained_baseline_required"] is True
    assert report["recommended_training_sequence"]["stage_1"] == "train_only_next_token_pretrain"
    if not report["entropy_baseline"]["comparison_valid"]:
        assert report["entropy_baseline"]["random_initialization_suspected"] is True
    assert report["train_only_vocabulary"]["holdout_target_unknown_count"] >= 1
    assert report["declared_ontology_inventory"]["slot_order"]
    assert report["holdout_contract"]["holdout_precedence_dedupe"] is True
    assert "records" not in report
    assert report["promotion"]["training_allowed"] is False


def test_pg372_builds_abstract_failure_repair_pairs_only() -> None:
    datasets = {"pg362": {"records": [_row("train", failure=False), _row("train", failure=True, repair="encoding")]}, "pg367": {"records": []}}
    report = build(datasets)
    assert report["counts"]["paired_failure_repair_groups"] == 1
    assert report["counts"]["failure_rows"] == 1
    assert report["counts"]["repair_rows"] == 1
    assert all(row["paired_failure_repair"] for row in report["records"])
    assert report["promotion"]["memory_promotion_allowed"] is False
    encoded = json.dumps(report, ensure_ascii=False)
    assert "payload=" not in encoded
    assert "wire=" not in encoded
    assert all(row["training_eligible"] is False for row in report["records"])


def test_pg372_holdout_unknown_values_are_blocked_not_leaked() -> None:
    datasets = {"pg362": {"records": [_row("train", failure=False), _row("implementation_holdout", failure=True, repair="syntax", syntax="redirect_control")]}, "pg367": {"records": []}}
    report = build(datasets)
    assert report["status"] == "blocked_incomplete"
    assert "holdout_vocabulary_gap" in report["failures"]
    assert report["counts"]["holdout_unknown_target_tokens"] >= 1
    assert "redirect_control" not in report["vocabulary"]["target_tokens"]
    assert "target_inventory_sha256" in report["declared_ontology_inventory"]
    assert report["train_only_gap"]["blocked"] is True
