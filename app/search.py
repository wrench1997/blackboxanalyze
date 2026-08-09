from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .rule_ir import canonical, complexity, pretty, truthy_result


@dataclass
class Candidate:
    expr: dict[str, Any]
    accuracy: float
    correct: int
    total: int
    complexity: int
    score: float
    predictions: tuple[bool, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": self.expr,
            "pretty": pretty(self.expr),
            "accuracy": round(self.accuracy, 6),
            "correct": self.correct,
            "total": self.total,
            "complexity": self.complexity,
            "score": round(self.score, 6),
        }


def field(path: str) -> dict[str, Any]:
    return {"op": "field", "path": path}


def prev(path: str, offset: int = 1) -> dict[str, Any]:
    return {"op": "prev", "path": path, "offset": offset}


def const(value: Any) -> dict[str, Any]:
    return {"op": "const", "value": value}


def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "left": left, "right": right}


def logic(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "args": [left, right]}


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _get_path(root: dict[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def observation_envelope(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": obs.get("input", {}),
        "context": obs.get("context", {}),
        "state": obs.get("state", {}),
    }


def build_histories_before(observations: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return the visible history for each observation, isolated by episode.

    An observation may provide an explicit ``history`` array. Otherwise only
    earlier observations with the same episode_id are visible. This prevents
    independent runs from contaminating each other.
    """
    by_episode: dict[str, list[dict[str, Any]]] = {}
    result: list[list[dict[str, Any]]] = []
    for obs in observations:
        episode = str(obs.get("episode_id", "default"))
        explicit = obs.get("history")
        if isinstance(explicit, list):
            history = list(explicit)
        else:
            history = list(by_episode.get(episode, []))
        result.append(history)
        by_episode.setdefault(episode, []).append(obs)
    return result


def latest_episode_history(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not observations:
        return []
    episode = str(observations[-1].get("episode_id", "default"))
    return [obs for obs in observations if str(obs.get("episode_id", "default")) == episode]


def _observed_values(path: str, observations: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for obs in observations:
        value = _get_path(obs, path)
        if value is not None and value not in values:
            values.append(value)
    return values


def _string_tokens(values: list[Any]) -> list[str]:
    tokens: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        if 2 <= len(value) <= 16:
            tokens.add(value)
        for token in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,12}", value):
            tokens.add(token)
        if len(value) >= 3:
            tokens.add(value[:2])
            tokens.add(value[-2:])
    return sorted(tokens, key=lambda item: (len(item), item))[:24]


def generate_atoms(fields: list[dict[str, Any]], observations: list[dict[str, Any]], history_depth: int = 1) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for spec in fields:
        path = spec["path"]
        kind = spec.get("type", "string")
        domain = list(spec.get("domain", []))
        observed = _observed_values(path, observations)
        values: list[Any] = []
        for value in domain + observed:
            if value not in values:
                values.append(value)

        refs = [field(path)]
        if history_depth > 0:
            refs.extend(prev(path, offset) for offset in range(1, history_depth + 1))

        for ref in refs:
            if kind in {"enum", "bool"}:
                for value in values[:24]:
                    atoms.append(binary("eq", ref, const(value)))
            elif kind == "number":
                numeric = sorted({float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)})
                thresholds: set[float] = set(numeric)
                for a, b in zip(numeric, numeric[1:]):
                    thresholds.add((a + b) / 2)
                for threshold in sorted(thresholds)[:32]:
                    if threshold.is_integer():
                        threshold = int(threshold)
                    atoms.append(binary("ge", ref, const(threshold)))
                    atoms.append(binary("lt", ref, const(threshold)))
                for value in numeric[:16]:
                    if float(value).is_integer():
                        value = int(value)
                    atoms.append(binary("eq", ref, const(value)))
                if ref.get("op") == "field":
                    for modulus in range(2, 8):
                        for remainder in range(modulus):
                            atoms.append(binary("eq", binary("mod", ref, const(modulus)), const(remainder)))
            elif kind == "string":
                for value in values[:20]:
                    atoms.append(binary("eq", ref, const(value)))
                for token in _string_tokens(values):
                    atoms.append(binary("contains", ref, const(token)))
                lengths = sorted({len(v) for v in values if isinstance(v, str)})
                for length in lengths[:16]:
                    atoms.append(binary("ge", {"op": "length", "arg": ref}, const(length)))
                    atoms.append(binary("lt", {"op": "length", "arg": ref}, const(length)))
            elif kind == "array":
                for value in values[:16]:
                    if isinstance(value, list):
                        for item in value[:12]:
                            atoms.append(binary("contains", ref, const(item)))
                for length in range(0, 8):
                    atoms.append(binary("eq", {"op": "count", "arg": ref}, const(length)))
                    atoms.append(binary("ge", {"op": "count", "arg": ref}, const(length)))

    unique: dict[str, dict[str, Any]] = {}
    for atom in atoms:
        unique.setdefault(canonical(atom), atom)
    return list(unique.values())


def score_expr(expr: dict[str, Any], observations: list[dict[str, Any]], complexity_penalty: float = 0.65) -> Candidate:
    predictions: list[bool] = []
    correct = 0
    histories = build_histories_before(observations)
    for obs, history in zip(observations, histories):
        prediction = truthy_result(expr, observation_envelope(obs), history)
        predictions.append(prediction)
        if prediction == bool(obs["output"]):
            correct += 1
    total = len(observations)
    accuracy = correct / total if total else 0.0
    size = complexity(expr)
    score = accuracy * 100.0 - size * complexity_penalty
    return Candidate(expr, accuracy, correct, total, size, score, tuple(predictions))


def _retain_best(candidates: Iterable[Candidate], limit: int, per_observed_behavior: int = 8) -> list[Candidate]:
    """Keep several structurally different rules per observed behavior.

    Keeping just one representative would make closure analysis falsely claim
    uniqueness: many hypotheses can fit the same observations but disagree on
    unseen inputs.
    """
    by_structure: dict[str, Candidate] = {}
    for candidate in candidates:
        key = canonical(candidate.expr)
        old = by_structure.get(key)
        if old is None or candidate.score > old.score:
            by_structure[key] = candidate

    groups: dict[tuple[bool, ...], list[Candidate]] = {}
    for candidate in by_structure.values():
        groups.setdefault(candidate.predictions, []).append(candidate)

    retained: list[Candidate] = []
    for group in groups.values():
        group.sort(key=lambda item: (-item.score, item.complexity, pretty(item.expr)))
        retained.extend(group[:per_observed_behavior])
    return sorted(retained, key=lambda item: (-item.score, item.complexity, pretty(item.expr)))[:limit]


def search_rules(
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    max_depth: int = 3,
    beam_width: int = 120,
    history_depth: int = 1,
) -> list[Candidate]:
    if not observations:
        return []
    atoms = generate_atoms(fields, observations, history_depth=history_depth)
    atomic_scored = _retain_best((score_expr(atom, observations) for atom in atoms), max(beam_width, 100))
    candidates = list(atomic_scored)

    negated = [score_expr({"op": "not", "arg": item.expr}, observations) for item in atomic_scored[:60]]
    candidates = _retain_best(candidates + negated, beam_width)
    frontier = candidates
    atom_pool = atomic_scored[: min(56, len(atomic_scored))]

    for _depth in range(2, max(2, max_depth + 1)):
        expanded: list[Candidate] = []
        for left in frontier[:beam_width]:
            for right in atom_pool:
                if canonical(left.expr) == canonical(right.expr):
                    continue
                expanded.append(score_expr(logic("and", left.expr, right.expr), observations))
                expanded.append(score_expr(logic("or", left.expr, right.expr), observations))
        frontier = _retain_best(expanded, beam_width)
        candidates = _retain_best(candidates + frontier, beam_width)
        if not frontier:
            break
    return candidates


def enumerate_envelopes(fields: list[dict[str, Any]], max_cases: int = 5000) -> list[dict[str, Any]]:
    domains: list[list[Any]] = []
    paths: list[str] = []
    for spec in fields:
        domain = list(spec.get("domain", []))
        if not domain:
            kind = spec.get("type")
            domain = [False, True] if kind == "bool" else [0, 1] if kind == "number" else [""]
        domains.append(domain)
        paths.append(spec["path"])

    cases: list[dict[str, Any]] = []
    for values in itertools.product(*domains):
        envelope: dict[str, Any] = {"input": {}, "context": {}, "state": {}}
        for path, value in zip(paths, values):
            _set_path(envelope, path, value)
        cases.append(envelope)
        if len(cases) >= max_cases:
            break
    return cases


def enumerate_history_variants(
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    history_depth: int = 1,
    max_variants: int = 256,
) -> list[list[dict[str, Any]]]:
    variants: list[list[dict[str, Any]]] = [[]]
    seen = {"[]"}

    def add(history: list[dict[str, Any]]) -> None:
        key = json.dumps(history, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen and len(variants) < max_variants:
            seen.add(key)
            variants.append(history)

    histories = build_histories_before(observations)
    for history in histories:
        for depth in range(1, min(history_depth, len(history)) + 1):
            add(history[-depth:])

    if history_depth > 0:
        input_fields = [spec for spec in fields if spec.get("path", "").startswith("input.")]
        for envelope in enumerate_envelopes(input_fields, max_cases=min(max_variants, 64)):
            add([{**envelope, "output": False}])
    return variants


def evaluation_cases(
    fields: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    history_depth: int = 1,
    max_cases: int = 5000,
    validation_cases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if validation_cases:
        cases: list[dict[str, Any]] = []
        for case in validation_cases[:max_cases]:
            cases.append({
                "envelope": {
                    "input": case.get("input", {}),
                    "context": case.get("context", {}),
                    "state": case.get("state", {}),
                },
                "history": case.get("history", []),
            })
        return cases

    envelopes = enumerate_envelopes(fields, max_cases=max_cases)
    variants = enumerate_history_variants(fields, observations, history_depth=history_depth)
    cases = []
    for history in variants:
        for envelope in envelopes:
            cases.append({"envelope": envelope, "history": history})
            if len(cases) >= max_cases:
                return cases
    return cases


def suggest_query(
    fields: list[dict[str, Any]],
    candidates: list[Candidate],
    observations: list[dict[str, Any]],
    max_cases: int = 5000,
) -> dict[str, Any] | None:
    if len(candidates) < 2:
        return None
    best_accuracy = candidates[0].accuracy
    top = [candidate for candidate in candidates[:32] if candidate.accuracy >= best_accuracy - 0.02]
    if len(top) < 2:
        top = candidates[: min(16, len(candidates))]

    seen = {
        json.dumps(observation_envelope(obs), sort_keys=True, ensure_ascii=False)
        for obs in observations
    }
    history = latest_episode_history(observations)
    best: tuple[float, dict[str, Any], list[bool]] | None = None
    for envelope in enumerate_envelopes(fields, max_cases=max_cases):
        key = json.dumps(envelope, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        predictions = [truthy_result(candidate.expr, envelope, history) for candidate in top]
        yes = sum(predictions)
        count = len(predictions)
        if yes == 0 or yes == count:
            entropy = 0.0
        else:
            p = yes / count
            entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        balance = 1.0 - abs(0.5 - yes / count) * 2.0
        score = entropy + balance * 0.05
        if best is None or score > best[0]:
            best = (score, envelope, predictions)
    if best is None:
        return None
    score, envelope, predictions = best
    return {
        **envelope,
        "disagreement": round(score, 6),
        "predicted_true": sum(predictions),
        "predicted_false": len(predictions) - sum(predictions),
        "candidate_count": len(predictions),
    }
