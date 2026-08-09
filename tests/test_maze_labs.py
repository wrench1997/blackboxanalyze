from app.maze_labs import get_maze_lab, public_maze_labs
from app.rule_ir import truthy_result
from app.scenarios import get_scenario


def test_maze_registry_covers_distinct_vulnerability_families():
    labs = public_maze_labs()
    assert len(labs) >= 6
    assert {lab["family"] for lab in labs} >= {"access_control", "url_redirect", "logic", "xss", "injection"}
    assert all("exit_oracle" in lab and "dead_ends" in lab for lab in labs)


def test_dom_and_sql_labs_are_synthetic_and_deterministic():
    dom = get_scenario("js_dom_sink_injection")
    assert truthy_result(dom["hidden_rule"], {"input": {"text": "plain marker"}, "context": {}, "state": {}}) is False
    assert truthy_result(dom["hidden_rule"], {"input": {"text": "<span data-sift-marker>inert</span>"}, "context": {}, "state": {}}) is True
    sql = get_scenario("js_sql_structure_boundary")
    assert truthy_result(sql["hidden_rule"], {"input": {"fragment_class": "plain"}, "context": {}, "state": {}}) is False
    assert truthy_result(sql["hidden_rule"], {"input": {"fragment_class": "operator_like"}, "context": {}, "state": {}}) is True
    assert get_maze_lab("maze_sql_boundary")["safety"] == "synthetic_no_database"

