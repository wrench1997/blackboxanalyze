from app.dom_oracle import run_dom_oracle
from app.maze_solver import assess_rule_exit
from app.sql_ast_oracle import run_sql_ast_oracle


def test_dom_oracle_is_inert_and_records_source_transform_sink_evidence():
    plain = run_dom_oracle("plain marker", marker="sift-marker").to_dict()
    assert plain["browser_sink_observed"] is True
    assert plain["dom_change"] is False
    assert plain["script_execution"] is False
    assert plain["network_access"] is False

    markup = run_dom_oracle('<span data-sift-marker="sift-marker">inert</span>', marker="sift-marker").to_dict()
    assert markup["dom_change"] is True
    assert markup["marker_hits"] == 1
    assert markup["script_execution"] is False
    assert markup["evidence_hash"]


def test_dom_oracle_requires_browser_sink_and_dom_delta_for_exit():
    evidence = run_dom_oracle('<span data-sift-marker="sift-marker">inert</span>').to_dict()
    result = assess_rule_exit("xss", visible_evidence=evidence, rechecks=[evidence])
    assert result["status"] == "observable_success"


def test_sql_ast_oracle_distinguishes_value_from_interpreter_boundary():
    plain = run_sql_ast_oracle("plain").to_dict()["evidence"]
    assert plain["controlled_differential"] is True
    assert plain["interpreter_boundary"] is False

    boundary = run_sql_ast_oracle("operator_like").to_dict()["evidence"]
    assert boundary["controlled_differential"] is True
    assert boundary["interpreter_boundary"] is True
    result = assess_rule_exit("injection", visible_evidence=boundary, rechecks=[boundary])
    assert result["status"] == "observable_success"
    assert boundary["database_touched"] is False


def test_sql_oracle_covers_error_blind_and_bounded_timing_channels_without_sleep():
    error = run_sql_ast_oracle("syntax_error").to_dict()["evidence"]
    assert error["modality"] == "syntax_error"
    assert error["syntax_error_observed"] is True
    assert error["real_sleep_performed"] is False

    blind = run_sql_ast_oracle("blind_boolean").to_dict()["evidence"]
    assert blind["modality"] == "blind_response"
    assert blind["blind_boolean_differential"] is True

    timed = run_sql_ast_oracle("time_delay").to_dict()["evidence"]
    assert timed["modality"] == "bounded_timing"
    assert timed["timeout_observed"] is True
    assert timed["simulated_latency_ms"] > timed["timeout_budget_ms"]

    for channel in (error, blind, timed):
        result = assess_rule_exit("injection", visible_evidence=channel, rechecks=[channel])
        assert result["status"] == "observable_success"

    local = run_sql_ast_oracle("local_side_channel").to_dict()["evidence"]
    assert local["modality"] == "local_side_channel"
    assert local["local_callback_observed"] is True
    assert local["local_callback_only"] is True
    assert local["external_network"] is False
    assert assess_rule_exit("injection", visible_evidence=local, rechecks=[local])["status"] == "observable_success"
