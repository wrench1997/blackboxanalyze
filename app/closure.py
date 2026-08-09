from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from .rule_ir import canonical, pretty, truthy_result
from .search import Candidate, build_histories_before, evaluation_cases, observation_envelope, score_expr

ClosureStatus = Literal[
    "insufficient_data",
    "open",
    "identified",
    "observationally_closed",
    "observationally_closed_low_coverage",
    "deadlocked",
    "suspected_deadlock",
    "context_incomplete_or_nondeterministic",
    "dsl_or_search_insufficient",
    "budget_or_domain_limited",
]


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def entropy_from_counts(counts: Counter[bool]) -> float:
    total = sum(counts.values())
    if total <= 1 or len(counts) <= 1:
        return 0.0
    result = 0.0
    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def _visible_key(obs: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    visible = observation_envelope(obs)
    if history is not None:
        visible["history"] = history
    elif isinstance(obs.get("history"), list):
        visible["history"] = obs["history"]
    return stable_json(visible)


def detect_conflicts(
    observations: list[dict[str, Any]],
    *,
    stateful: bool = False,
    history_depth: int = 1,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    histories = build_histories_before(observations) if stateful else [[] for _ in observations]
    for index, (obs, inferred_history) in enumerate(zip(observations, histories)):
        if isinstance(obs.get("history"), list):
            visible_history = obs["history"][-history_depth:] if history_depth > 0 else []
        elif stateful and history_depth > 0:
            visible_history = inferred_history[-history_depth:]
        else:
            visible_history = None
        groups[_visible_key(obs, visible_history)].append((index, obs))

    conflicts = []
    duplicate_groups = 0
    total_conflicting_rows = 0
    max_entropy = 0.0
    for key, rows in groups.items():
        if len(rows) > 1:
            duplicate_groups += 1
        counts = Counter(bool(row["output"]) for _, row in rows)
        if len(counts) > 1:
            ent = entropy_from_counts(counts)
            max_entropy = max(max_entropy, ent)
            total_conflicting_rows += len(rows)
            conflicts.append({
                "visible_envelope": json.loads(key),
                "outputs": {"true": counts.get(True, 0), "false": counts.get(False, 0)},
                "entropy": round(ent, 6),
                "observation_indexes": [index + 1 for index, _ in rows],
                "sources": sorted({str(row.get("source", "unknown")) for _, row in rows}),
            })

    return {
        "conflict_group_count": len(conflicts),
        "duplicate_group_count": duplicate_groups,
        "conflicting_observation_count": total_conflicting_rows,
        "conflict_rate": round(total_conflicting_rows / len(observations), 6) if observations else 0.0,
        "max_output_entropy": round(max_entropy, 6),
        "conflicts": conflicts[:30],
    }


def _delete_path(root: dict[str, Any], path: str) -> dict[str, Any]:
    clone = json.loads(stable_json(root))
    parts = path.split(".")
    current: Any = clone
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return clone
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)
    return clone


def analyze_field_necessity(fields: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for spec in fields:
        path = spec["path"]
        projected: dict[str, Counter[bool]] = defaultdict(Counter)
        for obs in observations:
            envelope = _delete_path(observation_envelope(obs), path)
            projected[stable_json(envelope)][bool(obs["output"])] += 1
        conflict_groups = [counts for counts in projected.values() if len(counts) > 1]
        conflicting_rows = sum(sum(counts.values()) for counts in conflict_groups)
        score = conflicting_rows / len(observations) if observations else 0.0
        results.append({
            "path": path,
            "label": spec.get("label", path),
            "necessity_score": round(score, 6),
            "projected_conflict_groups": len(conflict_groups),
            "classification": "necessary_for_disambiguation" if score > 0 else "not_proven_necessary",
        })
    results.sort(key=lambda item: (-item["necessity_score"], item["path"]))
    return {
        "fields": results,
        "required_fields": [item["path"] for item in results if item["necessity_score"] > 0],
        "not_proven_necessary": [item["path"] for item in results if item["necessity_score"] == 0],
    }


def _domain_size(fields: list[dict[str, Any]], prefix: str | None = None, cap: int = 10_000_000) -> int | None:
    size = 1
    matched = False
    for spec in fields:
        if prefix is not None and not spec.get("path", "").startswith(prefix):
            continue
        domain = spec.get("domain")
        if not isinstance(domain, list) or not domain:
            return None
        matched = True
        size *= len(domain)
        if size > cap:
            return size
    return size if matched else 1


def analyze_coverage(fields: list[dict[str, Any]], observations: list[dict[str, Any]], max_cases: int) -> dict[str, Any]:
    unique_envelopes = {_visible_key(obs) for obs in observations}
    total_domain = _domain_size(fields)
    if total_domain is None:
        envelope_coverage = None
    else:
        envelope_coverage = min(1.0, len(unique_envelopes) / max(1, total_domain))

    field_rows = []
    boundary_scores = []
    for spec in fields:
        path = spec["path"]
        domain = spec.get("domain", [])
        observed = []
        for obs in observations:
            current: Any = obs
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None and current not in observed:
                observed.append(current)
        if isinstance(domain, list) and domain:
            value_coverage = len([value for value in domain if value in observed]) / len(domain)
        else:
            value_coverage = None

        boundary_hit = None
        if spec.get("type") == "number" and domain:
            numeric = sorted(value for value in domain if isinstance(value, (int, float)) and not isinstance(value, bool))
            if numeric:
                hits = int(numeric[0] in observed) + int(numeric[-1] in observed)
                boundary_hit = hits / 2
                boundary_scores.append(boundary_hit)
        elif value_coverage is not None:
            boundary_scores.append(value_coverage)

        field_rows.append({
            "path": path,
            "observed_distinct": len(observed),
            "domain_size": len(domain) if isinstance(domain, list) and domain else None,
            "value_coverage": round(value_coverage, 6) if value_coverage is not None else None,
            "boundary_coverage": round(boundary_hit, 6) if boundary_hit is not None else None,
        })

    return {
        "unique_envelopes": len(unique_envelopes),
        "finite_domain_size": total_domain,
        "domain_truncated_by_max_cases": bool(total_domain and total_domain > max_cases),
        "envelope_coverage": round(envelope_coverage, 6) if envelope_coverage is not None else None,
        "boundary_coverage": round(sum(boundary_scores) / len(boundary_scores), 6) if boundary_scores else None,
        "fields": field_rows,
    }


def _candidate_pool(raw_candidates: list[dict[str, Any]], observations: list[dict[str, Any]], top_candidates: int) -> list[Candidate]:
    result = []
    seen = set()
    for item in raw_candidates:
        expr = item.get("expr")
        if not isinstance(expr, dict):
            continue
        key = canonical(expr)
        if key in seen:
            continue
        seen.add(key)
        result.append(score_expr(expr, observations))
    result.sort(key=lambda item: (-item.score, item.complexity, pretty(item.expr)))
    return result[:top_candidates]


def analyze_hypotheses(
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    validation_cases: list[dict[str, Any]] | None,
    history_depth: int,
    max_cases: int,
    top_candidates: int,
    accuracy_tolerance: float,
) -> dict[str, Any]:
    candidates = _candidate_pool(raw_candidates, observations, top_candidates)
    if not candidates:
        return {
            "candidate_count": 0,
            "consistent_candidate_count": 0,
            "best_accuracy": 0.0,
            "behavior_class_count": 0,
            "max_disagreement": None,
            "best_disagreement_case": None,
            "classes": [],
            "fingerprint": "none",
        }

    best_accuracy = candidates[0].accuracy
    eligible = [c for c in candidates if c.accuracy >= best_accuracy - accuracy_tolerance]
    consistent = [c for c in candidates if c.accuracy >= 1.0 - accuracy_tolerance]
    basis = consistent if consistent else eligible
    cases = evaluation_cases(
        fields,
        observations,
        history_depth=history_depth,
        max_cases=max_cases,
        validation_cases=validation_cases,
    )

    classes: dict[tuple[bool, ...], list[Candidate]] = defaultdict(list)
    for candidate in basis:
        vector = tuple(truthy_result(candidate.expr, case["envelope"], case["history"]) for case in cases)
        classes[vector].append(candidate)

    class_rows = []
    for vector, group in classes.items():
        representative = min(group, key=lambda c: (c.complexity, -c.score, pretty(c.expr)))
        vector_hash = hashlib.sha256(bytes(int(value) for value in vector)).hexdigest()[:16]
        class_rows.append({
            "behavior_hash": vector_hash,
            "rule_count": len(group),
            "representative": pretty(representative.expr),
            "representative_expr": representative.expr,
            "true_cases": sum(vector),
            "false_cases": len(vector) - sum(vector),
        })
    class_rows.sort(key=lambda item: (-item["rule_count"], item["representative"]))

    best_case = None
    best_entropy = -1.0
    if len(basis) >= 2:
        for case in cases:
            predictions = [truthy_result(candidate.expr, case["envelope"], case["history"]) for candidate in basis]
            yes = sum(predictions)
            if yes in {0, len(predictions)}:
                ent = 0.0
            else:
                p = yes / len(predictions)
                ent = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            if ent > best_entropy:
                best_entropy = ent
                best_case = {
                    "envelope": case["envelope"],
                    "history": case["history"],
                    "predicted_true": yes,
                    "predicted_false": len(predictions) - yes,
                    "candidate_count": len(predictions),
                    "entropy": round(ent, 6),
                }

    fingerprint_source = stable_json(sorted((row["behavior_hash"], row["rule_count"]) for row in class_rows))
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()[:20]
    return {
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "consistent_candidate_count": len(consistent),
        "best_accuracy": round(best_accuracy, 6),
        "behavior_class_count": len(classes),
        "evaluation_case_count": len(cases),
        "max_disagreement": round(max(0.0, best_entropy), 6) if best_case else None,
        "best_disagreement_case": best_case,
        "classes": class_rows[:20],
        "fingerprint": fingerprint,
    }


def _action_space(fields: list[dict[str, Any]], max_actions: int = 5000) -> list[dict[str, Any]] | None:
    specs = [spec for spec in fields if spec.get("path", "").startswith("input.")]
    if not specs:
        return [{}]
    domains = []
    for spec in specs:
        domain = spec.get("domain")
        if not isinstance(domain, list) or not domain:
            return None
        domains.append(domain)
    actions = []
    for values in itertools.product(*domains):
        action: dict[str, Any] = {}
        for spec, value in zip(specs, values):
            parts = spec["path"].split(".")[1:]
            current = action
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        actions.append(action)
        if len(actions) >= max_actions:
            break
    return actions


def _tarjan_scc(nodes: set[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in adjacency.get(node, set()):
            if nxt not in indexes:
                visit(nxt)
                lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
            elif nxt in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[nxt])
        if lowlinks[node] == indexes[node]:
            component = []
            while stack:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(component)

    for node in sorted(nodes):
        if node not in indexes:
            visit(node)
    return components


def analyze_state_graph(
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    stateful: bool,
    goal_mode: str,
) -> dict[str, Any]:
    if not stateful and not any(obs.get("state_after") is not None for obs in observations):
        return {
            "applicable": False,
            "reason": "场景未声明为有状态，且观测没有 state_after。",
            "state_count": 0,
            "transition_count": 0,
            "transition_coverage": None,
            "deadlock": None,
            "suspected_deadlock": None,
            "terminal_sccs": [],
        }

    action_space = _action_space(fields)
    expected_actions = len(action_space) if action_space is not None else None
    nodes: set[str] = set()
    state_values: dict[str, Any] = {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    tested_actions: dict[str, set[str]] = defaultdict(set)
    expected_action_sets: dict[str, set[str]] = defaultdict(set)
    goal_nodes: set[str] = set()
    explicit_terminal_nodes: set[str] = set()
    transitions = []

    by_episode: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, obs in enumerate(observations):
        by_episode[str(obs.get("episode_id", "default"))].append((index, obs))

    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda pair: (pair[1].get("step") is None, pair[1].get("step", pair[0]), pair[0]))
        for position, (index, obs) in enumerate(episode_rows):
            before = obs.get("state", {})
            after = obs.get("state_after")
            inferred = False
            if after is None and position + 1 < len(episode_rows):
                after = episode_rows[position + 1][1].get("state", {})
                inferred = True
            if after is None:
                continue
            before_key = stable_json(before)
            after_key = stable_json(after)
            nodes.update({before_key, after_key})
            state_values[before_key] = before
            state_values[after_key] = after
            adjacency[before_key].add(after_key)
            action_key = stable_json(obs.get("input", {}))
            tested_actions[before_key].add(action_key)
            if isinstance(obs.get("available_actions"), list):
                expected_action_sets[before_key].update(stable_json(action) for action in obs["available_actions"])
            is_goal = bool(obs.get("goal"))
            if goal_mode in {"output_true", "either"} and bool(obs.get("output")):
                is_goal = True
            if is_goal:
                goal_nodes.add(after_key)
            if bool(obs.get("terminal")):
                explicit_terminal_nodes.add(after_key)
            transitions.append({
                "from": before,
                "to": after,
                "action": obs.get("input", {}),
                "output": bool(obs.get("output")),
                "goal": is_goal,
                "inferred_state_after": inferred,
                "observation_index": index + 1,
            })

    if not nodes:
        return {
            "applicable": True,
            "reason": "没有足够的连续 state/state_after 数据来建立转移图。",
            "state_count": 0,
            "transition_count": 0,
            "transition_coverage": 0.0,
            "deadlock": None,
            "suspected_deadlock": None,
            "terminal_sccs": [],
        }

    per_node_expected: dict[str, int | None] = {}
    for node in nodes:
        if node in explicit_terminal_nodes:
            per_node_expected[node] = 0
        elif expected_action_sets.get(node):
            per_node_expected[node] = len(expected_action_sets[node])
        else:
            per_node_expected[node] = expected_actions
    possible_pairs = None if any(value is None for value in per_node_expected.values()) else sum(value or 0 for value in per_node_expected.values())
    tested_pair_count = sum(len(actions) for actions in tested_actions.values())
    transition_coverage = min(1.0, tested_pair_count / possible_pairs) if possible_pairs else (1.0 if possible_pairs == 0 else None)

    components = _tarjan_scc(nodes, adjacency)
    terminal_rows = []
    confirmed = []
    suspected = []
    for component in components:
        component_set = set(component)
        outgoing = {(src, dst) for src in component for dst in adjacency.get(src, set()) if dst not in component_set}
        if outgoing:
            continue
        has_goal = bool(component_set & goal_nodes)
        per_state = []
        complete = True
        for node in component:
            tested = len(tested_actions.get(node, set()))
            node_expected = per_node_expected.get(node)
            state_complete = node_expected is not None and tested >= node_expected
            complete = complete and state_complete
            per_state.append({
                "state": state_values.get(node, json.loads(node)),
                "tested_actions": tested,
                "expected_actions": node_expected,
                "complete": state_complete,
                "explicit_terminal": node in explicit_terminal_nodes,
            })
        explicitly_terminal = all(node in explicit_terminal_nodes for node in component)
        row = {
            "states": [state_values.get(node, json.loads(node)) for node in component],
            "size": len(component),
            "has_goal": has_goal,
            "action_coverage_complete": complete,
            "explicitly_terminal": explicitly_terminal,
            "per_state_coverage": per_state,
        }
        terminal_rows.append(row)
        if not has_goal:
            if complete or explicitly_terminal:
                confirmed.append(row)
            else:
                suspected.append(row)

    return {
        "applicable": True,
        "state_count": len(nodes),
        "transition_count": len(transitions),
        "tested_state_action_pairs": tested_pair_count,
        "expected_actions_per_state": expected_actions,
        "transition_coverage": round(transition_coverage, 6) if transition_coverage is not None else None,
        "goal_state_count": len(goal_nodes),
        "terminal_scc_count": len(terminal_rows),
        "terminal_sccs": terminal_rows[:20],
        "deadlock": confirmed[0] if confirmed else None,
        "suspected_deadlock": suspected[0] if suspected else None,
        "transitions": transitions[:200],
        "note": "默认把 goal=true 或 output=true 视作到达目标；可通过 goal_mode 调整。",
    }


def analyze_stability(history: list[dict[str, Any]], current_fingerprint: str, current_status: str) -> dict[str, Any]:
    fingerprints = [item.get("fingerprint") for item in history[-4:]] + [current_fingerprint]
    statuses = [item.get("status") for item in history[-4:]] + [current_status]
    if len(fingerprints) <= 1:
        return {"rounds_considered": 1, "fingerprint_stability": 0.5, "status_stability": 0.5, "stable_rounds": 1}
    fp_matches = sum(value == fingerprints[-1] for value in fingerprints)
    status_matches = sum(value == statuses[-1] for value in statuses)
    stable_rounds = 0
    for fp in reversed(fingerprints):
        if fp == fingerprints[-1]:
            stable_rounds += 1
        else:
            break
    return {
        "rounds_considered": len(fingerprints),
        "fingerprint_stability": round(fp_matches / len(fingerprints), 6),
        "status_stability": round(status_matches / len(statuses), 6),
        "stable_rounds": stable_rounds,
    }


def _average_known(*values: float | None, default: float = 0.0) -> float:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else default


def analyze_closure(
    *,
    scenario: dict[str, Any],
    observations: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
    previous_reports: list[dict[str, Any]] | None = None,
    max_cases: int = 5000,
    top_candidates: int = 48,
    accuracy_tolerance: float = 0.001,
    history_depth: int = 1,
    coverage_threshold: float = 0.9,
    goal_mode: str = "either",
) -> dict[str, Any]:
    previous_reports = previous_reports or []
    conflicts = detect_conflicts(
        observations,
        stateful=bool(scenario.get("stateful")),
        history_depth=history_depth,
    )
    necessity = analyze_field_necessity(scenario.get("fields", []), observations)
    coverage = analyze_coverage(scenario.get("fields", []), observations, max_cases)
    hypotheses = analyze_hypotheses(
        scenario.get("fields", []),
        observations,
        raw_candidates,
        scenario.get("validation_cases"),
        history_depth,
        max_cases,
        top_candidates,
        accuracy_tolerance,
    )
    graph = analyze_state_graph(
        scenario.get("fields", []),
        observations,
        bool(scenario.get("stateful")),
        goal_mode,
    )

    reasons: list[str] = []
    recommendations: list[str] = []
    status: ClosureStatus

    if not observations:
        status = "insufficient_data"
        reasons.append("还没有观测，无法判断一致性、可区分性或状态闭环。")
        recommendations.append("至少录入一组正例和一组反例，并覆盖输入域边界。")
    elif conflicts["conflict_group_count"] > 0:
        status = "context_incomplete_or_nondeterministic"
        reasons.append("相同可见上下文出现了不同输出，确定性规则无法同时解释这些观测。")
        recommendations.extend([
            "补录 episode_id、step、state、state_after、时间桶、随机种子或外部依赖版本。",
            "重复调用相同输入，判断是随机性、隐藏状态还是数据标注错误。",
        ])
    elif hypotheses["candidate_count"] == 0 or hypotheses["best_accuracy"] < 1.0 - accuracy_tolerance:
        status = "dsl_or_search_insufficient"
        reasons.append("当前候选规则无法完整解释无冲突数据。")
        recommendations.extend([
            "提高搜索深度和 Beam 宽度，或扩展 Rule IR 操作符。",
            "检查规则是否依赖时间、哈希、浮点误差、集合聚合、长历史或外部服务。",
        ])
    elif graph.get("deadlock") is not None:
        status = "deadlocked"
        reasons.append("发现无目标状态的终端强连通分量，而且其动作覆盖已完整。")
        recommendations.append("把报告中的终端 SCC 作为最小死路反例，检查是否应添加恢复转移。")
    elif graph.get("suspected_deadlock") is not None:
        status = "suspected_deadlock"
        reasons.append("发现无已知出口且不含目标的终端 SCC，但动作覆盖尚未完整。")
        recommendations.append("优先测试该 SCC 中每个状态尚未覆盖的动作，确认是否真的没有出口。")
    else:
        classes = hypotheses["behavior_class_count"]
        disagreement = hypotheses.get("max_disagreement")
        env_cov = coverage.get("envelope_coverage")
        trans_cov = graph.get("transition_coverage") if graph.get("applicable") else 1.0
        enough_coverage = (env_cov is not None and env_cov >= coverage_threshold) and (trans_cov is None or trans_cov >= coverage_threshold)
        if classes == 1 and (disagreement is None or disagreement <= 1e-12):
            if enough_coverage:
                if hypotheses["consistent_candidate_count"] == 1:
                    status = "identified"
                    reasons.append("只剩一个一致候选，并且有限输入域与状态转移覆盖达到阈值。")
                else:
                    status = "observationally_closed"
                    reasons.append("多个语法规则仍存在，但在已声明输入域内属于同一行为等价类。")
            else:
                status = "observationally_closed_low_coverage"
                reasons.append("候选在已枚举测试域内无分歧，但数据覆盖不足，不能证明真实黑盒已闭环。")
                recommendations.append("扩大字段 domain、补充边界值、异常值和状态动作转移。")
        elif disagreement is not None and disagreement > 1e-12:
            status = "open"
            reasons.append("仍能找到让高分候选产生不同预测的输入，逻辑尚未闭环。")
            recommendations.append("执行 best_disagreement_case；它是当前信息增益最大的实验。")
        else:
            status = "budget_or_domain_limited"
            reasons.append("没有找到可用分歧，但候选仍不唯一；测试域或搜索预算可能限制了判断。")
            recommendations.append("扩大数值范围、历史深度、状态域或 max_cases 后重新分析。")

    provisional_stability = analyze_stability(previous_reports, hypotheses["fingerprint"], status)
    consistency_score = 1.0 - conflicts["conflict_rate"]
    distinguishability_score = 1.0 - min(1.0, hypotheses.get("max_disagreement") or 0.0)
    envelope_score = coverage.get("envelope_coverage")
    boundary_score = coverage.get("boundary_coverage")
    transition_score = graph.get("transition_coverage") if graph.get("applicable") else 1.0
    stability_score = _average_known(
        provisional_stability["fingerprint_stability"],
        provisional_stability["status_stability"],
        default=0.5,
    )

    closure_score = (
        0.30 * consistency_score
        + 0.20 * distinguishability_score
        + 0.18 * (envelope_score if envelope_score is not None else 0.35)
        + 0.12 * (boundary_score if boundary_score is not None else 0.35)
        + 0.12 * (transition_score if transition_score is not None else 0.35)
        + 0.08 * stability_score
    )
    if status in {"context_incomplete_or_nondeterministic", "dsl_or_search_insufficient", "insufficient_data"}:
        closure_score = min(closure_score, 0.49)
    if status in {"open", "suspected_deadlock", "budget_or_domain_limited"}:
        closure_score = min(closure_score, 0.79)

    confidence = "high" if closure_score >= 0.9 else "medium" if closure_score >= 0.7 else "low"
    proof_conditions = {
        "no_visible_conflicts": conflicts["conflict_group_count"] == 0,
        "perfect_consistent_candidate_exists": hypotheses["consistent_candidate_count"] > 0,
        "single_behavior_class": hypotheses["behavior_class_count"] == 1,
        "no_candidate_disagreement": hypotheses.get("max_disagreement") in {None, 0.0},
        "envelope_coverage_reached": coverage.get("envelope_coverage") is not None and coverage["envelope_coverage"] >= coverage_threshold,
        "transition_coverage_reached": (not graph.get("applicable")) or (graph.get("transition_coverage") is not None and graph["transition_coverage"] >= coverage_threshold),
        "no_confirmed_deadlock": graph.get("deadlock") is None,
    }

    if not recommendations and status in {"identified", "observationally_closed"}:
        recommendations.append("保留一组独立隐藏测试，并在黑盒版本变化后重新运行闭环分析。")

    return {
        "closure_status": status,
        "closure_score": round(closure_score, 6),
        "confidence": confidence,
        "reasons": reasons,
        "recommendations": recommendations,
        "proof_conditions": proof_conditions,
        "conflicts": conflicts,
        "hypotheses": hypotheses,
        "coverage": coverage,
        "state_graph": graph,
        "field_necessity": necessity,
        "stability": provisional_stability,
        "fingerprint": hypotheses["fingerprint"],
        "limitations": [
            "闭环结论只对声明的有限 domain、历史深度和 Rule IR 搜索空间成立。",
            "未记录的时间、随机数、并发、网络响应和隐藏存储仍可能改变黑盒行为。",
            "低覆盖率下的‘无分歧’只能称为观测闭环，不能证明任意程序的数学等价。",
        ],
    }
