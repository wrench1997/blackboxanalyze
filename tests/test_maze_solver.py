from app.juice_shop_adapter import agent_observation
from app.maze_solver import (
    MazeGraph,
    assess_authentication_exit,
    assess_rule_exit,
    state_fingerprint,
    state_projection,
)


def obs(path: str, status: int, *, cookie: bool = False, semantic: str = "stable", headers=None):
    return agent_observation(
        action={"method": "GET", "path": path},
        status_code=status,
        response_headers=headers or {"content-type": "application/json"},
        response_summary={
            "body_length": len(semantic),
            "semantic_body_sha256": semantic,
            "json_shape": {"data": "list"},
            "cookie_jar_changed": cookie,
        },
    )


def test_public_200_is_not_an_authentication_exit():
    assessment = assess_authentication_exit(obs("/public", 200), obs("/public", 200))
    assert assessment["status"] == "not_goal"
    assert "baseline_was_not_denied" in assessment["reasons"]


def test_auth_exit_requires_same_resource_session_signal_and_recheck():
    baseline = obs("/api/private", 401, headers={"www-authenticate": "Bearer"})
    login_after = obs("/api/login", 200, cookie=True)
    first = obs("/api/private", 200)
    second = obs("/api/private", 200)
    one_try = assess_authentication_exit(baseline, first, auth_transition=login_after)
    assert one_try["status"] == "candidate"
    stable = assess_authentication_exit(
        baseline,
        first,
        auth_transition=login_after,
        protected_rechecks=[second],
    )
    assert stable["status"] == "observable_success"
    assert stable["evaluator_confirmed"] is False


def test_redirect_and_reflection_do_not_count_as_browser_xss_exit():
    redirect = assess_rule_exit(
        "url_redirect",
        visible_evidence={"candidate_signal": True, "location_origin_changed": True},
    )
    assert redirect["status"] == "candidate"
    xss = assess_rule_exit(
        "xss",
        visible_evidence={"candidate_signal": True, "dom_change": True},
    )
    assert xss["status"] == "candidate"
    assert "browser_sink_observed" in xss["missing_evidence"]


def test_rule_exit_replay_is_family_specific():
    result = assess_rule_exit(
        "logic",
        visible_evidence={"candidate_signal": True, "invariant_violation": True, "state_replay": True},
        rechecks=[{"invariant_violation": True, "state_replay": True}],
    )
    assert result["status"] == "observable_success"


def test_maze_records_forward_loop_dead_end_and_hides_body_preview():
    start = obs("/start", 200, semantic="start")
    middle = obs("/middle", 200, semantic="middle")
    graph = MazeGraph()
    graph.record_transition(start, {"method": "GET", "path": "/middle"}, middle)
    loop = graph.record_transition(middle, {"method": "GET", "path": "/start"}, start)
    assert loop["kind"] == "loop"
    graph.mark_dead_end(middle, "same_state_after_probe")
    graph.enqueue(middle, {"method": "GET", "path": "/dead"}, priority=0.4, reason="unseen")
    assert graph.frontier.pop()["reason"] == "unseen"
    state = state_projection(start)
    assert "body_preview" not in state
    assert graph.nodes[state_fingerprint(start)].dead_end is False
