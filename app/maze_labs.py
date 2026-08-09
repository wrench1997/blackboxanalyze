"""Research registry for safe, local rule-maze laboratories.

Each entry points at a black-box scenario and names the *semantic* exit oracle
needed to claim a result.  The registry never embeds an exploit payload or
evaluator challenge key in an agent observation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MAZE_LABS: tuple[dict[str, Any], ...] = (
    {
        "id": "maze_truthiness_access",
        "scenario_id": "js_truthy_access",
        "family": "access_control",
        "state_model": "stateless",
        "exit_oracle": "protected_resource_transition",
        "candidate_signal": "boolean_policy_divergence",
        "stable_replay": "same_input_same_policy_result",
        "dead_ends": ["admin_only_baseline", "zero_value", "unchanged_decision"],
        "research_axis": "type coercion and boundary probing",
        "safety": "in_process_synthetic",
    },
    {
        "id": "maze_redirect_origin",
        "scenario_id": "js_substring_redirect",
        "family": "url_redirect",
        "state_model": "stateless",
        "exit_oracle": "location_origin_changed",
        "candidate_signal": "url_decision_divergence",
        "stable_replay": "same_url_same_origin_decision",
        "dead_ends": ["same_origin", "invalid_url", "no_location_change"],
        "research_axis": "structured URL parsing versus string matching",
        "safety": "in_process_synthetic_no_navigation",
    },
    {
        "id": "maze_business_boundary",
        "scenario_id": "js_boundary_coupon",
        "family": "logic",
        "state_model": "stateless",
        "exit_oracle": "invariant_violation",
        "candidate_signal": "boundary_output_differential",
        "stable_replay": "boundary_case_reproduces",
        "dead_ends": ["below_boundary", "non_member", "same_output"],
        "research_axis": "active search for narrow business-rule boundaries",
        "safety": "in_process_synthetic",
    },
    {
        "id": "maze_sequence_replay",
        "scenario_id": "js_sequence_replay",
        "family": "logic",
        "state_model": "history_dependent",
        "exit_oracle": "invariant_violation",
        "candidate_signal": "history_dependent_acceptance",
        "stable_replay": "fresh_episode_replay",
        "dead_ends": ["wrong_previous_action", "empty_history", "cancel_transition"],
        "research_axis": "long-context state reconstruction and replay isolation",
        "safety": "in_process_synthetic",
    },
    {
        "id": "maze_dom_sink",
        "scenario_id": "js_dom_sink_injection",
        "family": "xss",
        "state_model": "browser_observable",
        "exit_oracle": "browser_sink_observed+dom_change",
        "candidate_signal": "html_reflection_only",
        "stable_replay": "fresh_page_dom_replay",
        "dead_ends": ["escaped_text", "detached_no_sink", "no_dom_delta"],
        "research_axis": "source-transform-sink abstraction without script execution",
        "safety": "detached_dom_no_script",
    },
    {
        "id": "maze_dom_double_decode",
        "scenario_id": "js_dom_sink_injection",
        "family": "xss",
        "state_model": "browser_observable",
        "exit_oracle": "browser_sink_observed+dom_change+double_decode",
        "candidate_signal": "encoded_text_reflection",
        "stable_replay": "fresh_page_double_decode_replay",
        "dead_ends": ["single_decode_only", "escaped_text", "no_dom_delta"],
        "research_axis": "encoding depth and source-transform-sink composition",
        "safety": "detached_dom_no_script",
    },
    {
        "id": "maze_dom_template_sink",
        "scenario_id": "js_dom_sink_injection",
        "family": "xss",
        "state_model": "browser_observable",
        "exit_oracle": "template_sink+dom_change",
        "candidate_signal": "template_fragment_reflection",
        "stable_replay": "fresh_template_fragment_replay",
        "dead_ends": ["text_node_only", "escaped_text", "no_dom_delta"],
        "research_axis": "template fragment versus element sink semantics",
        "safety": "detached_dom_no_script",
    },
    {
        "id": "maze_sql_boundary",
        "scenario_id": "js_sql_structure_boundary",
        "family": "injection",
        "state_model": "interpreter_observable",
        "exit_oracle": "controlled_differential+interpreter_boundary",
        "candidate_signal": "query_shape_differential",
        "stable_replay": "same_fragment_class_same_ast_shape",
        "dead_ends": ["value_only_change", "parameterized_shape", "parse_shape_unchanged"],
        "research_axis": "language-independent representation of interpreter boundaries",
        "safety": "synthetic_no_database",
    },
    {
        "id": "maze_sql_error_channel",
        "scenario_id": "js_sql_structure_boundary",
        "family": "injection",
        "state_model": "error_channel_observable",
        "exit_oracle": "controlled_differential+syntax_error",
        "candidate_signal": "error_shape_changed",
        "stable_replay": "same_error_class_on_recheck",
        "dead_ends": ["normal_value", "generic_validation_error", "parse_shape_unchanged"],
        "research_axis": "error-channel evidence without treating every 4xx as injection",
        "safety": "synthetic_no_database",
    },
    {
        "id": "maze_sql_blind_channel",
        "scenario_id": "js_sql_structure_boundary",
        "family": "injection",
        "state_model": "blind_response_observable",
        "exit_oracle": "controlled_differential+blind_response",
        "candidate_signal": "boolean_or_row_shape_difference",
        "stable_replay": "same_shape_bit_on_recheck",
        "dead_ends": ["same_row_shape", "noise_only", "value_only_change"],
        "research_axis": "body-independent boolean and row-shape differentials",
        "safety": "synthetic_no_database",
    },
    {
        "id": "maze_sql_timing_channel",
        "scenario_id": "js_sql_structure_boundary",
        "family": "injection",
        "state_model": "bounded_timing_observable",
        "exit_oracle": "controlled_differential+bounded_timing",
        "candidate_signal": "latency_or_timeout_difference",
        "stable_replay": "bounded latency bucket repeats",
        "dead_ends": ["within_budget", "jitter_only", "unbounded_timeout"],
        "research_axis": "time-channel evidence with a hard local budget and no sleep",
        "safety": "synthetic_no_database_no_sleep",
    },
    {
        "id": "maze_sql_local_side_channel",
        "scenario_id": "js_sql_structure_boundary",
        "family": "injection",
        "state_model": "local_side_channel_observable",
        "exit_oracle": "controlled_differential+local_side_channel",
        "candidate_signal": "local_marker_observed",
        "stable_replay": "same_local_marker_on_recheck",
        "dead_ends": ["no_marker", "external_network_blocked", "noise_only"],
        "research_axis": "带外信号的本地代理，不触发真实反连",
        "safety": "synthetic_local_queue_no_network",
    },
)


def public_maze_labs() -> list[dict[str, Any]]:
    return deepcopy(list(MAZE_LABS))


def get_maze_lab(lab_id: str) -> dict[str, Any] | None:
    for lab in MAZE_LABS:
        if lab["id"] == lab_id:
            return deepcopy(lab)
    return None
