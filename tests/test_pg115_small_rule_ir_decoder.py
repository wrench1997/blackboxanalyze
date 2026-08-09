import hashlib
import json
from pathlib import Path

from app.pg115_small_rule_ir_decoder import RULE_IR_BY_DECISION, validate_abstract_rule_ir


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg115_small_decoder_trial_runs_on_gpu_or_cpu_and_passes_hard_ood_checks():
    report = _load("pg115_small_rule_ir_decoder_report_v1.json")
    assert report["status"] == "completed_pg115_small_rule_ir_decoder_trial"
    assert report["scope"]["parameter_count"] == 4612
    assert report["scope"]["epochs"] == 30
    assert report["training"]["dev_metrics"]["accuracy"] >= 0.95
    assert report["blind_pg114"]["family_holdout_confirm_recall"] > 0.0
    assert report["blind_pg114"]["decoy_false_accept_count"] == 0
    assert report["blind_pg114"]["withheld_oracle_abstain_rate"] > 0.0
    assert all(report["checks"].values())
    assert report["promotion"]["checkpoint_written"] is True
    assert report["promotion"]["training_artifact_promotion_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert (ROOT / "artifacts" / "pg115-small-rule-ir-decoder-v1" / "model.pt").exists()


def test_pg115_training_fixture_has_disjoint_seeds_and_no_evaluator_input_leakage():
    dataset = _load("pg115_small_rule_ir_train_dataset_v1.json")
    assert dataset["training_eligible"] is True
    assert dataset["memory_promotion_allowed"] is False
    assert dataset["model_input_family_free"] is True
    assert dataset["model_input_oracle_blind"] is True
    assert dataset["raw_probe_strings_stored"] is False
    assert dataset["raw_response_bodies_stored"] is False
    assert len(dataset["rows"]) == 800
    train_seeds = {row["source_seed"] for row in dataset["rows"] if row["split"] == "train"}
    dev_seeds = {row["source_seed"] for row in dataset["rows"] if row["split"] == "dev"}
    assert train_seeds.isdisjoint(dev_seeds)
    forbidden = ("family", "oracle", "positive_authority", "evaluator", "target_instance_slot", "probe_ref")
    for row in dataset["rows"]:
        model_text = json.dumps(row["model_input"], ensure_ascii=False).casefold()
        assert not any(token in model_text for token in forbidden)
        assert row["training_eligible"] is True
        assert row["memory_promotion_allowed"] is False


def test_pg115_rule_ir_outputs_are_grammar_checked_and_protocol_is_explicit():
    for expression in RULE_IR_BY_DECISION.values():
        validate_abstract_rule_ir(expression)
    protocol = _load("pg115_small_rule_ir_decoder_protocol_v1.json")
    assert protocol["model"]["feature_dim"] == 40
    assert protocol["model"]["previous_checkpoint_reuse_forbidden"] is True
    assert protocol["split"]["pg114_excluded_from_training"] is True
    assert protocol["hard_metrics"]["all_abstain_is_not_success"] is True
    assert protocol["promotion"]["memory_promotion_allowed"] is False


def test_pg115_report_source_hashes_match_current_code():
    report = _load("pg115_small_rule_ir_decoder_report_v1.json")
    for key, relative_path in {
        "decoder": "app/pg115_small_rule_ir_decoder.py",
        "runner": "scripts/run_pg115_small_rule_ir_decoder.py",
        "pg114_report": "research/pg114_family_holdout_replay_report_v1.json",
    }.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == report["source"][key]

