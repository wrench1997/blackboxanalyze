"""Source-grounded structural memory for authorized detection probes."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .maze_engine import canonical_json, sha256_json, validate_evidence
from .memory_promotion_gate import DEFAULT_PROMOTION_POLICY, assess_memory_promotion
from .payload_catalog import structural_feature_key, validate_policy_candidate


GROUNDING_SCHEMA = "sift-source-grounded-memory-v1"
OUTCOME_STATUSES = frozenset({"rejected", "dead_end", "candidate", "observable_success"})


class SourceGroundedMemory:
    """Remember successful *structural* probe features with provenance.

    Family and evaluator labels are never read by this class.  A candidate is
    admitted only when its source attestation is valid; selection then uses a
    feature key derived from the validated payload (DOM encoding, SQL channel,
    or generic HTTP canary).  This is a small controller baseline for PG-02,
    not a neural generator.
    """

    def __init__(self, *, seed: int = 20260802, exploration: float = 1.0) -> None:
        self.seed = int(seed)
        self.exploration = float(exploration)
        self.rng = random.Random(self.seed)
        self.stats: dict[str, dict[str, Any]] = {}
        self.feedback: list[dict[str, Any]] = []
        self.total_attempts = 0

    def _ensure(self, candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
        normalized = validate_policy_candidate(candidate)
        feature = structural_feature_key(normalized["payload"])
        row = self.stats.setdefault(feature, {
            "feature_key": feature,
            "attempts": 0,
            "reward_sum": 0.0,
            "observable_successes": 0,
            "candidates": 0,
            "dead_ends": 0,
            "rejections": 0,
            "source_ids": [],
            "last_evidence_hash": None,
        })
        source_id = normalized["source_attestation"]["source_id"]
        if source_id not in row["source_ids"]:
            row["source_ids"].append(source_id)
            row["source_ids"] = sorted(row["source_ids"])
        return normalized, row, feature

    def supported_features(self) -> list[str]:
        return sorted(
            feature for feature, row in self.stats.items()
            if int(row.get("observable_successes", 0)) > 0
        )

    def select(
        self,
        candidates: Iterable[dict[str, Any]],
        *,
        require_supported: bool = True,
    ) -> dict[str, Any] | None:
        pool = [copy.deepcopy(candidate) for candidate in candidates]
        if not pool:
            raise ValueError("source-grounded candidate pool must not be empty")
        scored: list[tuple[float, int, dict[str, Any]]] = []
        log_total = math.log(self.total_attempts + 2.0)
        for index, candidate in enumerate(pool):
            normalized, row, feature = self._ensure(candidate)
            successes = int(row["observable_successes"])
            if require_supported and successes <= 0:
                continue
            attempts = int(row["attempts"])
            mean = float(row["reward_sum"]) / attempts if attempts else 0.0
            bonus = self.exploration * math.sqrt(log_total / (attempts + 1.0))
            normalized["selection_score"] = round(mean + bonus, 6)
            normalized["selection_mode"] = "source_grounded"
            normalized["structural_feature"] = feature
            scored.append((mean + bonus, -index, normalized))
        if not scored:
            return None
        _, _, chosen = max(scored, key=lambda item: (item[0], item[1]))
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
            raise ValueError(f"unsupported source-grounded outcome: {status}")
        normalized, row, feature = self._ensure(candidate)
        checked_evidence = validate_evidence(evidence) if evidence is not None else None
        reward = {
            "rejected": -0.25,
            "dead_end": 0.0,
            "candidate": 0.25,
            "observable_success": 1.0,
        }[status]
        row["attempts"] += 1
        row["reward_sum"] = round(float(row["reward_sum"]) + reward, 6)
        if status == "observable_success":
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
            "candidate_id": normalized["candidate_id"],
            "source_id": normalized["source_attestation"]["source_id"],
            "structural_feature": feature,
            "status": status,
            "reward": reward,
            "evaluator_confirmed": bool(evaluator_confirmed),
            "policy_uses_evaluator": False,
            "evidence_hash": checked_evidence["evidence_hash"] if checked_evidence else None,
        }
        self.feedback.append(feedback)
        self.feedback = self.feedback[-2048:]
        return feedback

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": GROUNDING_SCHEMA,
            "feature_count": len(self.stats),
            "supported_feature_count": len(self.supported_features()),
            "supported_features": self.supported_features(),
            "attempt_count": self.total_attempts,
            "policy_uses_evaluator": False,
            "long_term_promotion_requires_cross_dataset_replay": True,
            "promotion_policy": dict(DEFAULT_PROMOTION_POLICY),
            "source_ids": sorted({source_id for row in self.stats.values() for source_id in row["source_ids"]}),
        }

    def audit_promotion(
        self,
        rule_key: str,
        evaluations: Iterable[dict[str, Any]],
        *,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Audit durable-memory promotion without changing episode selection."""

        return assess_memory_promotion(rule_key, evaluations, policy=policy)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": GROUNDING_SCHEMA,
            "seed": self.seed,
            "exploration": self.exploration,
            "total_attempts": self.total_attempts,
            "stats": copy.deepcopy(self.stats),
            "feedback": copy.deepcopy(self.feedback),
            "summary": self.summary(),
            "checkpoint_sha256": sha256_json({
                "seed": self.seed,
                "stats": self.stats,
                "feedback": self.feedback,
            }),
        }

    def save(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint()
        path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return checkpoint
