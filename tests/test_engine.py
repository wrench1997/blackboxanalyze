from app.rule_ir import evaluate, truthy_result
from app.scenarios import get_scenario
from app.search import search_rules, suggest_query
from app.synthetic_curriculum import generate_curriculum
from app.maturity import evaluate_research_maturity, triage_scale_failure


def test_rule_ir_parity_color():
    scenario = get_scenario("parity_color")
    rule = scenario["hidden_rule"]
    assert truthy_result(rule, {"input": {"color": "red", "number": 4}, "context": {}, "state": {}})
    assert not truthy_result(rule, {"input": {"color": "red", "number": 3}, "context": {}, "state": {}})
    assert not truthy_result(rule, {"input": {"color": "blue", "number": 4}, "context": {}, "state": {}})


def test_search_recovers_simple_rule():
    scenario = get_scenario("parity_color")
    observations = []
    for color in ["red", "blue"]:
        for number in range(6):
            envelope = {"input": {"color": color, "number": number}, "context": {}, "state": {}}
            observations.append({**envelope, "output": truthy_result(scenario["hidden_rule"], envelope)})
    candidates = search_rules(scenario["fields"], observations, max_depth=3, beam_width=160, history_depth=0)
    assert candidates
    assert candidates[0].accuracy == 1.0
    assert "input.color" in candidates[0].to_dict()["pretty"]


def test_sequence_history():
    scenario = get_scenario("sequence_lock")
    history = [{"input": {"action": "knock"}, "context": {}, "state": {}, "output": False}]
    envelope = {"input": {"action": "open"}, "context": {}, "state": {}}
    assert truthy_result(scenario["hidden_rule"], envelope, history)


def test_active_query_returns_unseen_case():
    scenario = get_scenario("parity_color")
    observations = [
        {"input": {"color": "red", "number": 2}, "context": {}, "state": {}, "output": True},
        {"input": {"color": "blue", "number": 2}, "context": {}, "state": {}, "output": False},
        {"input": {"color": "red", "number": 3}, "context": {}, "state": {}, "output": False},
    ]
    candidates = search_rules(scenario["fields"], observations, max_depth=2, beam_width=80, history_depth=0)
    suggestion = suggest_query(scenario["fields"], candidates, observations)
    assert suggestion is not None


def test_js_research_corpus_behaviors():
    truthy = get_scenario("js_truthy_access")
    assert truthy_result(truthy["hidden_rule"], {"input": {"role": "guest", "quota": -1}, "context": {}, "state": {}})
    assert not truthy_result(truthy["hidden_rule"], {"input": {"role": "guest", "quota": 0}, "context": {}, "state": {}})

    redirect = get_scenario("js_substring_redirect")
    assert truthy_result(redirect["hidden_rule"], {"input": {"next": "https://trusted.com.evil.test/phish"}, "context": {}, "state": {}})
    assert not truthy_result(redirect["hidden_rule"], {"input": {"next": "https://example.com/home"}, "context": {}, "state": {}})

    coupon = get_scenario("js_boundary_coupon")
    assert not truthy_result(coupon["hidden_rule"], {"input": {"member": True, "total": 100}, "context": {}, "state": {}})
    assert truthy_result(coupon["hidden_rule"], {"input": {"member": True, "total": 101}, "context": {}, "state": {}})


def test_synthetic_curriculum_is_reproducible_and_verified():
    first = generate_curriculum(12, traces_per_program=6, seed=17)
    second = generate_curriculum(12, traces_per_program=6, seed=17)
    assert first == second
    assert {row["split"] for row in first} == {"train", "validation", "test"}
    assert all(row["verification"]["verified_counterexample_count"] >= 1 for row in first)

from app.closure import analyze_closure, detect_conflicts


def test_conflicting_visible_context_detected():
    observations = [
        {"input": {"x": 1}, "context": {}, "state": {}, "output": True},
        {"input": {"x": 1}, "context": {}, "state": {}, "output": False},
    ]
    result = detect_conflicts(observations, stateful=False)
    assert result["conflict_group_count"] == 1
    assert result["max_output_entropy"] == 1.0


def test_episode_history_prevents_false_conflict():
    observations = [
        {"episode_id": "a", "step": 0, "input": {"action": "knock"}, "context": {}, "state": {}, "output": False},
        {"episode_id": "a", "step": 1, "input": {"action": "open"}, "context": {}, "state": {}, "output": True},
        {"episode_id": "b", "step": 0, "input": {"action": "wait"}, "context": {}, "state": {}, "output": False},
        {"episode_id": "b", "step": 1, "input": {"action": "open"}, "context": {}, "state": {}, "output": False},
    ]
    result = detect_conflicts(observations, stateful=True, history_depth=1)
    assert result["conflict_group_count"] == 0


def test_closure_open_on_partial_domain_and_closed_on_full_domain():
    scenario = get_scenario("parity_color")
    partial = []
    for color in ["red", "blue"]:
        for number in range(6):
            envelope = {"input": {"color": color, "number": number}, "context": {}, "state": {}}
            partial.append({**envelope, "output": truthy_result(scenario["hidden_rule"], envelope)})
    candidates = search_rules(scenario["fields"], partial, max_depth=3, beam_width=180, history_depth=0)
    report = analyze_closure(
        scenario=scenario,
        observations=partial,
        raw_candidates=[candidate.to_dict() for candidate in candidates],
        history_depth=0,
    )
    assert report["closure_status"] == "open"
    assert report["hypotheses"]["best_disagreement_case"] is not None

    full = []
    for color in ["red", "blue", "green"]:
        for number in range(13):
            envelope = {"input": {"color": color, "number": number}, "context": {}, "state": {}}
            full.append({**envelope, "output": truthy_result(scenario["hidden_rule"], envelope)})
    candidates = search_rules(scenario["fields"], full, max_depth=3, beam_width=180, history_depth=0)
    report = analyze_closure(
        scenario=scenario,
        observations=full,
        raw_candidates=[candidate.to_dict() for candidate in candidates],
        history_depth=0,
    )
    assert report["closure_status"] in {"identified", "observationally_closed"}
    assert report["hypotheses"]["behavior_class_count"] == 1
    assert report["coverage"]["envelope_coverage"] == 1.0


def test_terminal_scc_is_confirmed_deadlock_when_all_actions_tested():
    scenario = {
        "name": "deadlock",
        "stateful": True,
        "fields": [
            {"path": "input.action", "type": "enum", "domain": ["left", "right"]},
        ],
    }
    observations = [
        {
            "episode_id": "run",
            "step": 0,
            "input": {"action": "left"},
            "context": {},
            "state": {"room": "trap"},
            "state_after": {"room": "trap"},
            "output": False,
        },
        {
            "episode_id": "run",
            "step": 1,
            "input": {"action": "right"},
            "context": {},
            "state": {"room": "trap"},
            "state_after": {"room": "trap"},
            "output": False,
        },
    ]
    always_false = {"op": "const", "value": False}
    report = analyze_closure(
        scenario=scenario,
        observations=observations,
        raw_candidates=[{"expr": always_false}],
        history_depth=0,
        goal_mode="observation_goal",
    )
    assert report["closure_status"] == "deadlocked"
    assert report["state_graph"]["deadlock"] is not None


def test_research_maturity_gate_blocks_premature_scaling():
    result = evaluate_research_maturity(
        {
            "reproducible": True,
            "independent_seeds": 1,
            "family_holdout_runs": 1,
            "ablation_supports_mechanism": True,
            "preregistered_target_met": True,
            "guardrails_passed": True,
            "data_lineage_complete": True,
        }
    )
    assert result["state"] == "research_only"
    assert result["failed_checks"] == ["independent_seeds"]


def test_scale_failure_triage_separates_experiment_and_engineering():
    engineering = triage_scale_failure(
        {
            "single_node_passes_but_distributed_fails": True,
            "data_hash_or_lineage_mismatch": True,
        }
    )
    assert engineering["classification"] == "engineering_capability_problem"

    mixed = triage_scale_failure(
        {
            "family_holdout_regression": True,
            "nondeterministic_pipeline": True,
        }
    )
    assert mixed["classification"] == "mixed"


def test_structured_url_and_dom_rule_ir():
    origin_rule = {
        "op": "origin_eq",
        "left": {"op": "field", "path": "input.origin"},
        "right": {"op": "const", "value": "https://trusted.com"},
    }
    assert truthy_result(origin_rule, {"input": {"origin": "https://trusted.com:443"}})
    assert not truthy_result(origin_rule, {"input": {"origin": "https://eviltrusted.com"}})

    tag_count_rule = {"op": "html_tag_count", "arg": {"op": "field", "path": "input.payload"}}
    assert evaluate(tag_count_rule, {"input": {"payload": "plain text"}}) == 0
    assert evaluate(tag_count_rule, {"input": {"payload": "<span>probe</span>"}}) == 1
    assert evaluate(tag_count_rule, {"input": {"payload": "&lt;span&gt;encoded&lt;/span&gt;"}}) == 0


def test_frontend_holdout_families_have_verified_counterexamples():
    rows = generate_curriculum(12, traces_per_program=20, seed=20260801)
    by_family = {row["family"]: row for row in rows}
    for family in ("postmessage_origin", "dom_sink_injection"):
        assert family in by_family
        assert by_family[family]["split"] == "test"
        assert by_family[family]["verification"]["verified_counterexample_count"] >= 1


def test_generalization_rule_ir_primitives():
    assert evaluate({"op": "casefold", "arg": {"op": "field", "path": "input.role"}}, {"input": {"role": "ADMIN"}}) == "admin"
    assert evaluate({"op": "to_number", "arg": {"op": "field", "path": "input.amount"}}, {"input": {"amount": " 0x10 "}}) == 16.0
    decoded = {"op": "html_entity_decode", "arg": {"op": "field", "path": "input.payload"}}
    assert evaluate(decoded, {"input": {"payload": "&lt;b&gt;x&lt;/b&gt;"}}) == "<b>x</b>"
    assert truthy_result({"op": "html_creates_nodes", "arg": decoded}, {"input": {"payload": "&lt;b&gt;x&lt;/b&gt;"}})


def test_generalization_matrix_families_are_complete_holdouts():
    families = {
        "url_scheme_downgrade",
        "dom_double_decode",
        "unicode_casefold_role",
        "numeric_string_coercion",
        "compound_origin_role",
        "state_replay_window",
    }
    rows = generate_curriculum(18, traces_per_program=20, seed=20261001)
    by_family = {row["family"]: row for row in rows}
    assert families <= set(by_family)
    for family in families:
        row = by_family[family]
        assert row["split"] == "test"
        assert row["generalization"]["axis"] != "baseline"
        assert row["verification"]["verified_counterexample_count"] >= 1
