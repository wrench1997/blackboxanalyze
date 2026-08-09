import hashlib
import json
import re
from pathlib import Path

from app.active_goal_label_inducer import ActiveGoalLabelInducer, proposal_digest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "research" / name).read_text(encoding="utf-8"))


def test_pg103_induces_generic_goal_labels_and_passes_unknown_family_gate():
    proposal = _load("pg103_auto_goal_label_active_probe_proposal_v1.json")
    report = _load("pg103_auto_goal_label_active_probe_report_v1.json")
    protocol = _load("pg103_auto_goal_label_active_probe_protocol_v1.json")
    assert proposal["schema_version"] == "active-auto-goal-label-inducer-v1"
    assert proposal["proposal_inputs"]["oracle_visible"] is False
    assert proposal["proposal_inputs"]["family_visible"] is False
    assert proposal["audit"]["vulnerability_family_names_not_generated"] is True
    assert proposal["supported_slots"] == [f"p{i}" for i in range(8)]
    assert len(proposal["labels"]) == 10
    assert proposal["goal"]["budget"]["requires_get_post_pair"] is True
    assert report["status"] == "passed_generic_goal_label_diagnostic"
    guarded = report["metrics"]["guarded_proposal"]
    assert guarded["all_evaluation"]["known_confirm_recall"] == 1.0
    assert guarded["all_evaluation"]["known_label_consistency"] == 1.0
    assert guarded["all_evaluation"]["false_accept_count"] == 0
    assert guarded["pg42"]["unknown_family_strict_abstain"] is True
    assert guarded["pg76_unseen_family"]["unknown_family_strict_abstain"] is True
    goal = report["metrics"]["guarded_goal"]
    assert goal["paired_episode_count"] == 273
    assert goal["known_repeat_goal_completion_rate"] == 1.0
    assert goal["known_label_consistency_rate"] == 1.0
    assert goal["negative_false_completion_count"] == 0
    assert goal["unknown_strict_abstain_rate"] == 1.0
    assert report["raw_failure_visible"]["failure_present"] is True
    assert report["raw_failure_visible"]["unknown_family_misname_count"] == 48
    assert report["promotion"]["training_allowed"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert protocol["proposal_contract"]["vulnerability_family_generation_forbidden"] is True
    assert protocol["fresh_evaluation_contract"]["row_count"] == 24


def test_pg103_visible_artifacts_are_evaluator_blind_and_fresh_replay_attested():
    report = _load("pg103_auto_goal_label_active_probe_report_v1.json")
    dataset = _load("pg103_auto_goal_label_active_probe_visible_dataset_v1.json")
    trace = _load("pg103_auto_goal_label_active_probe_trace_v1.json")
    assert dataset["evaluation_only"] is True
    assert dataset["training_eligible"] is False
    assert trace["training_eligible"] is False
    assert dataset["model_input_contract"]["oracle_is_label_not_feature"] is True
    assert dataset["model_input_contract"]["family_label_in_features"] is False
    assert dataset["model_input_contract"]["proposal_is_generic_effect_only"] is True
    assert len(dataset["rows"]) == 546
    assert len(trace["steps"]) == 546
    assert trace["evaluator_labels_in_trace"] is False
    assert all("evaluator_label" not in row for row in dataset["rows"])
    assert all(not row["raw_probe_strings_stored"] and not row["raw_response_body_stored"] for row in dataset["rows"])
    assert sorted({row["method"] for row in dataset["rows"]}) == ["GET", "POST"]
    assert all(row["negative_control_matched"] for row in dataset["rows"])
    assert all(row["fresh_reset"] for row in dataset["rows"])
    assert len({row["fresh_reset"]["target_instance_id"] for row in dataset["rows"]}) == 546
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]) for row in dataset["rows"])
    for row in dataset["rows"]:
        assert not any(key in row["model_input"] for key in ("family", "oracle", "raw_body", "route_template_id"))
        assert row["model_input"]["probe_order"] == [f"p{i}" for i in range(9)]
    assert dataset["long_term_memory_write"] is False
    text = json.dumps({"proposal": _load("pg103_auto_goal_label_active_probe_proposal_v1.json"), "dataset": dataset, "trace": trace}, ensure_ascii=False).casefold()
    for forbidden in ("workflow_invariant", "<script", "onerror", "union select", "markup_candidate", "operator_like", "template_candidate"):
        assert forbidden not in text
    for name, digest in report["source"]["source_hashes"].items():
        path = {
            "pg101_input_dataset": "research/pg101_active_probe_signature_visible_dataset_v1.json",
            "pg101_input_report": "research/pg101_active_probe_signature_report_v1.json",
            "pg76_fixture": "app/pg76_unknown_triplet_fixture.py",
            "active_probe_module": "app/active_probe_signature.py",
            "inducer_module": "app/active_goal_label_inducer.py",
            "runner": "scripts/run_pg103_auto_goal_label_active_probe.py",
        }[name]
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_pg103_inducer_rejects_no_effect_and_abstains_on_unseen_or_ambiguous_slots():
    dataset = _load("pg103_auto_goal_label_active_probe_visible_dataset_v1.json")
    train = [row for row in dataset["rows"] if row["source"] == "pg42"][:0]
    frozen = _load("pg101_active_probe_signature_visible_dataset_v1.json")
    train_rows = [{"model_input": row["model_input"]} for row in frozen["rows"] if row["role"] == "train"]
    inducer = ActiveGoalLabelInducer().fit(train_rows)
    unknown_signature = next(row["model_input"] for row in frozen["rows"] if row["role"] == "family_holdout")
    assert inducer.predict(unknown_signature)["decision"] == "abstain"
    negative_signature = next(row["model_input"] for row in dataset["rows"] if not any(row["model_input"]["delta_pattern"]))
    assert inducer.predict(negative_signature)["decision"] == "reject"
    ambiguous = json.loads(json.dumps(unknown_signature))
    ambiguous["delta_pattern"][0] = True
    assert inducer.predict(ambiguous)["decision"] == "abstain"
    proposal = inducer.proposal()
    digest_input = dict(proposal)
    digest_input.pop("proposal_sha256", None)
    assert proposal_digest(digest_input) == proposal["proposal_sha256"]
