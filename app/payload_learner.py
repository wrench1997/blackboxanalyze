"""Self-learning controller for safe detection payloads.

The learner searches a constrained probe grammar and updates a contextual
bandit from local oracle feedback.  It never invents or executes unrestricted
exploit strings; a future neural generator can plug into the same candidate
and feedback contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .detection_payload import build_detection_payload, payload_digest, validate_detection_payload
from .maze_engine import canonical_json, sha256_json, validate_evidence


LEARNER_SCHEMA = "sift-payload-learner-v1"
OUTCOME_STATUSES = frozenset({"rejected", "dead_end", "candidate", "observable_success", "evaluator_confirmed"})
SAFE_FAMILIES = frozenset({"xss", "injection", "access_control", "url_redirect", "logic"})


def _candidate_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def _candidate(
    *,
    family: str,
    path: str,
    marker: str,
    probe: str,
    probe_kind: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    payload = build_detection_payload(
        path=path,
        marker=marker,
        probe=probe,
        probe_kind=probe_kind,
        expected=expected,
    )
    candidate = {
        "candidate_id": _candidate_id(payload),
        "family": family,
        "grammar": probe_kind,
        "payload": payload,
    }
    return candidate


def generate_payload_candidates(family: str, *, path: str, marker: str = "sift-auto-probe") -> list[dict[str, Any]]:
    """Generate bounded, safe candidates for one semantic vulnerability family."""

    family = str(family).casefold()
    if family not in SAFE_FAMILIES:
        raise ValueError(f"unsupported self-learning family: {family}")
    if family == "xss":
        return [
            _candidate(
                family=family,
                path=path,
                marker=marker,
                probe=f'<span data-sift-marker="{marker}">{marker}</span>',
                probe_kind="inert_dom_markup",
                expected={"browser_sink_observed": True, "dom_change": True},
            ),
            _candidate(
                family=family,
                path=path,
                marker=marker,
                probe=f'&amp;lt;span data-sift-marker="{marker}"&amp;gt;{marker}&amp;lt;/span&amp;gt;',
                probe_kind="encoded_dom_markup",
                expected={"browser_sink_observed": True, "dom_change": True, "requires_decode_depth": 2},
            ),
        ]
    if family == "injection":
        return [
            _candidate(family=family, path=path, marker=marker, probe=fragment, probe_kind="sql_channel_class", expected={"channel": fragment, "requires_recheck": True})
            for fragment in ("operator_like", "subquery_like", "blind_boolean", "row_shape", "syntax_error", "time_delay", "local_side_channel")
        ]
    expected = {
        "access_control": {"protected_resource_transition": True, "requires_recheck": True},
        "url_redirect": {"location_origin_changed": True, "navigation_must_remain_false": True},
        "logic": {"invariant_violation": True, "state_replay": True},
    }[family]
    return [_candidate(family=family, path=path, marker=marker, probe=marker, probe_kind="http_canary", expected=expected)]


class PayloadLearner:
    """Deterministic UCB learner over validated detection payload candidates."""

    def __init__(self, *, seed: int = 20260802, exploration: float = 1.25) -> None:
        self.seed = int(seed)
        self.exploration = float(exploration)
        self.rng = random.Random(self.seed)
        self.stats: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self.total_attempts = 0

    def _ensure(self, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = validate_detection_payload(dict(candidate.get("payload") or {}))
        candidate_id = str(candidate.get("candidate_id") or _candidate_id(payload))
        row = self.stats.setdefault(candidate_id, {
            "candidate_id": candidate_id,
            "family": str(candidate.get("family", "unknown")),
            "grammar": str(candidate.get("grammar", payload.get("probe_kind", "unknown"))),
            "attempts": 0,
            "reward_sum": 0.0,
            "observable_successes": 0,
            "candidates": 0,
            "dead_ends": 0,
            "rejections": 0,
            "last_evidence_hash": None,
            "payload_sha256": payload_digest(payload),
        })
        return row

    def select(self, candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
        pool = [copy.deepcopy(candidate) for candidate in candidates]
        if not pool:
            raise ValueError("payload learner candidate pool must not be empty")
        scored: list[tuple[float, int, dict[str, Any]]] = []
        log_total = math.log(self.total_attempts + 2.0)
        for index, candidate in enumerate(pool):
            row = self._ensure(candidate)
            attempts = int(row["attempts"])
            mean = float(row["reward_sum"]) / attempts if attempts else 0.0
            bonus = self.exploration * math.sqrt(log_total / (attempts + 1.0))
            scored.append((mean + bonus, -index, candidate))
        score, _, chosen = max(scored, key=lambda item: (item[0], item[1]))
        chosen["selection_score"] = round(score, 6)
        return chosen

    def select_replay(self, candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Select a previously successful candidate for a replay episode.

        Standard UCB intentionally keeps exploring unseen arms.  That is
        useful during discovery but makes a short replay budget noisy.  This
        explicit memory gate preserves the original ``select`` behaviour and
        only changes replay selection after observable success has been
        recorded.  If no successful candidate exists yet, it falls back to
        ordinary UCB discovery.
        """

        pool = [copy.deepcopy(candidate) for candidate in candidates]
        if not pool:
            raise ValueError("payload learner replay candidate pool must not be empty")
        successful: list[tuple[float, int, dict[str, Any]]] = []
        for index, candidate in enumerate(pool):
            row = self._ensure(candidate)
            attempts = int(row["attempts"])
            successes = int(row["observable_successes"])
            if successes <= 0:
                continue
            mean = float(row["reward_sum"]) / attempts if attempts else 0.0
            successful.append((mean, -index, candidate))
        if not successful:
            return self.select(pool)
        score, _, chosen = max(successful, key=lambda item: (item[0], item[1]))
        chosen["selection_score"] = round(score, 6)
        chosen["selection_mode"] = "memory_replay"
        return chosen

    def observe(
        self,
        candidate: dict[str, Any],
        *,
        status: str,
        evidence: dict[str, Any] | None = None,
        evaluator_confirmed: bool = False,
    ) -> dict[str, Any]:
        status = str(status)
        if status not in OUTCOME_STATUSES:
            raise ValueError(f"unsupported payload learner outcome: {status}")
        row = self._ensure(candidate)
        checked_evidence = validate_evidence(evidence) if evidence is not None else None
        reward = {
            "rejected": -0.25,
            "dead_end": 0.0,
            "candidate": 0.25,
            "observable_success": 1.0,
            # Evaluator state is recorded for reporting but deliberately gets
            # the same policy reward as observable success.
            "evaluator_confirmed": 1.0,
        }[status]
        row["attempts"] += 1
        row["reward_sum"] = round(float(row["reward_sum"]) + reward, 6)
        if status in {"observable_success", "evaluator_confirmed"}:
            row["observable_successes"] += 1
        elif status == "candidate":
            row["candidates"] += 1
        elif status == "dead_end":
            row["dead_ends"] += 1
        elif status == "rejected":
            row["rejections"] += 1
        if checked_evidence:
            row["last_evidence_hash"] = checked_evidence["evidence_hash"]
        self.total_attempts += 1
        feedback = {
            "step": self.total_attempts,
            "candidate_id": row["candidate_id"],
            "status": status,
            "reward": reward,
            "evaluator_confirmed": bool(evaluator_confirmed),
            "policy_uses_evaluator": False,
            "evidence_hash": checked_evidence["evidence_hash"] if checked_evidence else None,
        }
        self.feedback.append(feedback)
        self.feedback = self.feedback[-1024:]
        return feedback

    def summary(self) -> dict[str, Any]:
        successes = sum(int(row["observable_successes"]) for row in self.stats.values())
        return {
            "schema_version": LEARNER_SCHEMA,
            "candidate_count": len(self.stats),
            "attempt_count": self.total_attempts,
            "observable_success_count": successes,
            "observable_success_rate": successes / self.total_attempts if self.total_attempts else 0.0,
            "policy_uses_evaluator": False,
            "families": sorted({str(row["family"]) for row in self.stats.values()}),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": LEARNER_SCHEMA,
            "seed": self.seed,
            "exploration": self.exploration,
            "total_attempts": self.total_attempts,
            "stats": copy.deepcopy(self.stats),
            "feedback": copy.deepcopy(self.feedback),
            "summary": self.summary(),
            "checkpoint_sha256": sha256_json({"seed": self.seed, "stats": self.stats, "feedback": self.feedback}),
        }

    def save(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint()
        path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return checkpoint

    @classmethod
    def load(cls, path: Path) -> "PayloadLearner":
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("schema_version") != LEARNER_SCHEMA:
            raise ValueError("unsupported payload learner checkpoint schema")
        learner = cls(seed=int(checkpoint["seed"]), exploration=float(checkpoint["exploration"]))
        learner.total_attempts = int(checkpoint.get("total_attempts", 0))
        learner.stats = dict(checkpoint.get("stats") or {})
        learner.feedback = list(checkpoint.get("feedback") or [])[-1024:]
        return learner
