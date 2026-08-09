from pathlib import Path

from app.dom_oracle import run_dom_oracle
from app.payload_learner import PayloadLearner, generate_payload_candidates
from app.sql_ast_oracle import run_sql_ast_oracle


def test_payload_learner_generates_visible_safe_probes_not_unrestricted_strings():
    dom = generate_payload_candidates("xss", path="/playground", marker="sift-auto-01")
    sql = generate_payload_candidates("injection", path="/api/search", marker="sift-auto-02")
    assert len(dom) == 2
    assert len(sql) >= 7
    assert dom[0]["payload"]["probe_kind"] == "inert_dom_markup"
    assert "sift-auto-01" in dom[0]["payload"]["probe"]
    assert all(item["payload"]["safety"]["does_not_execute"] for item in sql)


def test_payload_learner_updates_from_oracle_feedback_and_persists_checkpoint(tmp_path: Path):
    candidates = generate_payload_candidates("xss", path="/playground", marker="sift-auto-03")
    learner = PayloadLearner(seed=7)
    chosen = learner.select(candidates)
    evidence = run_dom_oracle('<span data-sift-marker="sift-auto-03">sift-auto-03</span>', marker="sift-auto-03").to_dict()
    feedback = learner.observe(chosen, status="observable_success", evidence=evidence)
    assert feedback["policy_uses_evaluator"] is False
    assert learner.summary()["observable_success_count"] == 1
    checkpoint_path = tmp_path / "payload-learner.json"
    learner.save(checkpoint_path)
    restored = PayloadLearner.load(checkpoint_path)
    assert restored.summary()["attempt_count"] == 1


def test_sql_payload_learning_keeps_evaluator_separate():
    candidate = generate_payload_candidates("injection", path="/api/search", marker="sift-auto-04")[0]
    evidence = run_sql_ast_oracle("operator_like").to_dict()["evidence"]
    learner = PayloadLearner()
    feedback = learner.observe(candidate, status="evaluator_confirmed", evidence=evidence, evaluator_confirmed=True)
    assert feedback["evaluator_confirmed"] is True
    assert feedback["policy_uses_evaluator"] is False


def test_memory_replay_prefers_observed_success_over_unseen_candidates():
    candidates = generate_payload_candidates("xss", path="/playground", marker="sift-auto-05")
    learner = PayloadLearner(seed=19)
    successful = candidates[1]
    evidence = run_dom_oracle(
        successful["payload"]["probe"],
        transforms=["html_entity_decode", "html_entity_decode"],
        marker=successful["payload"]["marker"],
    ).to_dict()
    learner.observe(successful, status="observable_success", evidence=evidence)
    chosen = learner.select_replay([candidates[0], successful])
    assert chosen["candidate_id"] == successful["candidate_id"]
    assert chosen["selection_mode"] == "memory_replay"
