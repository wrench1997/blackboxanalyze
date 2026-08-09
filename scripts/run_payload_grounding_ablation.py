"""Run the PG-01 payload-grounding controller ablation.

This is a local, synthetic experiment.  The policy receives a mixed pool of
validated detection manifests without the ``family`` or ``grammar`` labels.
The benchmark harness keeps the target contract private to the evaluator and
returns only bounded DOM/SQL oracle evidence plus a reward.  Nothing in this
script sends a request, executes JavaScript, sleeps, or touches a database.

The experiment has two phases:

* ``warmup`` measures discovery on a fresh candidate pool;
* ``replay`` reuses the learner state to test whether feedback makes the
  correct abstract probe easy to select on a later episode.

The UCB learner is a controller baseline, not a neural model and not a claim
that public penetration-testing payload corpora have been ingested.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make ``python scripts/run_payload_grounding_ablation.py`` work on a clean
# Windows checkout as well as an invocation with the repository on PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dom_oracle import run_dom_oracle
from app.payload_learner import PayloadLearner, generate_payload_candidates
from app.sql_ast_oracle import run_sql_ast_oracle


PROTOCOL_ID = "sift-payload-grounding-ablation-v1"
SEEDS = (20260811, 20260817, 20260823, 20260829, 20260831)
WARMUP_BUDGET = 10
REPLAY_BUDGET = 3
STRATEGIES = ("ucb", "ucb_memory", "random")


TARGETS: tuple[dict[str, Any], ...] = (
    {
        "target_id": "xss_double_decode",
        "family": "xss",
        "path": "/playground",
        "required_probe_kind": "encoded_dom_markup",
        "candidate_probe_kind": "inert_dom_markup",
        "description": "detached DOM sink requires two inert HTML-entity decodes",
    },
    {
        "target_id": "sql_time_channel",
        "family": "injection",
        "path": "/api/search",
        "required_probe_kind": "sql_channel_class",
        "required_probe": "time_delay",
        "candidate_modality": "bounded_timing",
        "description": "abstract SQL channel is a bounded timing differential",
    },
    {
        "target_id": "sql_error_channel",
        "family": "injection",
        "path": "/api/search",
        "required_probe_kind": "sql_channel_class",
        "required_probe": "syntax_error",
        "candidate_modality": "syntax_error",
        "description": "abstract SQL channel is a syntax-error differential",
    },
    {
        "target_id": "sql_blind_branch",
        "family": "injection",
        "path": "/api/search",
        "required_probe_kind": "sql_channel_class",
        "required_probe": "blind_boolean",
        "candidate_modality": "blind_response",
        "description": "abstract SQL channel is a blind response branch",
    },
)


def _fresh_marker(target_index: int, seed: int, *, fresh: bool = False) -> str:
    prefix = "pgfresh" if fresh else "pg"
    # The marker stays in the allow-listed identifier grammar and is unique per
    # target/seed without exposing a target label to the policy.
    return f"{prefix}-{target_index + 1}-{seed}"


def build_policy_pool(target: dict[str, Any], *, seed: int, fresh: bool = False) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build a shuffled mixed pool and keep evaluator metadata out of it."""

    target_index = next(index for index, item in enumerate(TARGETS) if item["target_id"] == target["target_id"])
    marker = _fresh_marker(target_index, seed, fresh=fresh)
    raw = [
        *generate_payload_candidates("xss", path="/playground", marker=marker),
        *generate_payload_candidates("injection", path="/api/search", marker=marker),
    ]
    random.Random(seed ^ (target_index + 1) * 7919 ^ (1 if fresh else 0)).shuffle(raw)

    # Only the candidate id and validated payload are visible to the policy.
    # Family/grammar and target metadata remain in this evaluator-side map.
    policy_pool = [
        {
            "candidate_id": item["candidate_id"],
            "payload": copy.deepcopy(item["payload"]),
        }
        for item in raw
    ]
    metadata = {
        item["candidate_id"]: {
            "family": item["family"],
            "grammar": item["grammar"],
            "probe_kind": item["payload"]["probe_kind"],
            "probe": item["payload"]["probe"],
        }
        for item in raw
    }
    return policy_pool, metadata


def evaluate_candidate(
    candidate: dict[str, Any],
    *,
    metadata: dict[str, Any],
    target: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run one local oracle and apply the hidden target contract.

    The returned ``observation`` is for the report only.  It is never passed
    to :meth:`PayloadLearner.observe`; the policy receives status/reward and
    the oracle's bounded evidence hash, not the hidden target match.
    """

    payload = candidate["payload"]
    family = str(metadata["family"])
    kind = str(payload["probe_kind"])
    probe = str(payload["probe"])
    marker = str(payload["marker"])

    if family == "xss":
        transforms = ["html_entity_decode", "html_entity_decode"] if kind == "encoded_dom_markup" else []
        evidence = run_dom_oracle(probe, transforms=transforms, marker=marker).to_dict()
        oracle_signal = bool(evidence.get("candidate_signal"))
        exact = (
            target["family"] == "xss"
            and kind == target.get("required_probe_kind")
            and len(transforms) == 2
            and int(evidence.get("marker_hits", 0)) > 0
            and bool(evidence.get("dom_change"))
        )
        near = (
            target["family"] == "xss"
            and kind == target.get("candidate_probe_kind")
            and oracle_signal
        )
        observed_surface = "dom_sink" if oracle_signal else "dom_no_change"
    elif family == "injection":
        evidence = run_sql_ast_oracle(probe).to_dict()["evidence"]
        oracle_signal = bool(evidence.get("candidate_signal"))
        exact = (
            target["family"] == "injection"
            and kind == target.get("required_probe_kind")
            and probe == target.get("required_probe")
            and oracle_signal
        )
        near = (
            target["family"] == "injection"
            and evidence.get("modality") == target.get("candidate_modality")
            and oracle_signal
            and not exact
        )
        observed_surface = str(evidence.get("modality", "unknown"))
    else:  # pragma: no cover - the pool is deliberately restricted above.
        raise ValueError(f"unsupported evaluator family: {family}")

    if exact:
        status = "observable_success"
        status_basis = "hidden_target_match"
    elif near:
        status = "candidate"
        status_basis = "same_surface_near_miss"
    else:
        status = "dead_end"
        status_basis = "no_target_surface_signal"

    observation = {
        "source_family": family,
        "observed_surface": observed_surface,
        "oracle_signal": oracle_signal,
        "hidden_target_match": exact,
        "status_basis": status_basis,
    }
    return status, evidence, observation


def _select(
    learner: PayloadLearner,
    pool: list[dict[str, Any]],
    *,
    strategy: str,
    rng: random.Random,
    phase: str,
) -> dict[str, Any]:
    if strategy == "ucb":
        return learner.select(pool)
    if strategy == "ucb_memory":
        return learner.select_replay(pool) if phase == "replay" else learner.select(pool)
    if strategy == "random":
        chosen = copy.deepcopy(rng.choice(pool))
        chosen["selection_score"] = None
        return chosen
    raise ValueError(f"unknown strategy: {strategy}")


def _run_episode(
    learner: PayloadLearner,
    pool: list[dict[str, Any]],
    metadata: dict[str, dict[str, Any]],
    target: dict[str, Any],
    *,
    strategy: str,
    rng: random.Random,
    phase: str,
    budget: int,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for step in range(1, budget + 1):
        chosen = _select(learner, pool, strategy=strategy, rng=rng, phase=phase)
        status, evidence, observation = evaluate_candidate(
            chosen,
            metadata=metadata[chosen["candidate_id"]],
            target=target,
        )
        feedback = learner.observe(
            chosen,
            status=status,
            evidence=evidence,
            evaluator_confirmed=False,
        )
        attempts.append({
            "phase": phase,
            "step": step,
            "candidate_id": chosen["candidate_id"],
            "probe_kind": chosen["payload"]["probe_kind"],
            "probe": chosen["payload"]["probe"],
            "selection_score": chosen.get("selection_score"),
            "selection_mode": chosen.get("selection_mode", "ucb_or_random"),
            "status": status,
            "reward": feedback["reward"],
            "evaluator_confirmed": feedback["evaluator_confirmed"],
            "policy_uses_evaluator": feedback["policy_uses_evaluator"],
            "evidence_hash": evidence.get("evidence_hash"),
            "observation": observation,
        })
    return attempts


def _first_success(attempts: list[dict[str, Any]]) -> int | None:
    for row in attempts:
        if row["status"] in {"observable_success", "evaluator_confirmed"}:
            return int(row["step"])
    return None


def _success_count(attempts: list[dict[str, Any]]) -> int:
    return sum(row["status"] in {"observable_success", "evaluator_confirmed"} for row in attempts)


def _stable_replay(
    candidate: dict[str, Any] | None,
    *,
    metadata: dict[str, dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    if candidate is None:
        return {"available": False, "stable": None, "evidence_hashes": []}
    hashes: list[str | None] = []
    statuses: list[str] = []
    for _ in range(2):
        status, evidence, _ = evaluate_candidate(
            candidate,
            metadata=metadata[candidate["candidate_id"]],
            target=target,
        )
        statuses.append(status)
        hashes.append(evidence.get("evidence_hash"))
    return {
        "available": True,
        "stable": statuses == ["observable_success", "observable_success"] and len(set(hashes)) == 1,
        "statuses": statuses,
        "evidence_hashes": hashes,
    }


def _fresh_target_replay(target: dict[str, Any], *, seed: int, successful_probe: str | None) -> dict[str, Any]:
    """Replay the learned grammar on a fresh marker/target instance."""

    if successful_probe is None:
        return {"available": False, "success": None}
    pool, metadata = build_policy_pool(target, seed=seed, fresh=True)
    matching = [item for item in pool if item["payload"]["probe"] == successful_probe]
    if not matching:
        # DOM payloads contain the marker, so match by the abstract kind for a
        # fresh marker; SQL classes are stable strings and use the exact probe.
        matching = [
            item for item in pool
            if item["payload"]["probe_kind"] == target.get("required_probe_kind")
            and (target["family"] != "injection" or item["payload"]["probe"] == target.get("required_probe"))
        ]
    if not matching:
        return {"available": True, "success": False}
    candidate = matching[0]
    status, evidence, observation = evaluate_candidate(
        candidate,
        metadata=metadata[candidate["candidate_id"]],
        target=target,
    )
    return {
        "available": True,
        "success": status == "observable_success",
        "status": status,
        "evidence_hash": evidence.get("evidence_hash"),
        "observed_surface": observation["observed_surface"],
    }


def run_one(target: dict[str, Any], *, seed: int, strategy: str) -> dict[str, Any]:
    pool, metadata = build_policy_pool(target, seed=seed)
    learner = PayloadLearner(seed=seed)
    rng = random.Random(seed ^ 0x5EED if strategy == "random" else seed)
    warmup = _run_episode(
        learner,
        pool,
        metadata,
        target,
        strategy=strategy,
        rng=rng,
        phase="warmup",
        budget=WARMUP_BUDGET,
    )
    replay = _run_episode(
        learner,
        pool,
        metadata,
        target,
        strategy=strategy,
        rng=rng,
        phase="replay",
        budget=REPLAY_BUDGET,
    )
    warmup_success_step = _first_success(warmup)
    replay_success_step = _first_success(replay)
    successful_attempt = next(
        (row for row in reversed(warmup + replay) if row["status"] == "observable_success"),
        None,
    )
    successful_candidate = None
    successful_probe = None
    if successful_attempt is not None:
        successful_candidate = next(
            (item for item in pool if item["candidate_id"] == successful_attempt["candidate_id"]),
            None,
        )
        successful_probe = successful_attempt["probe"]
    stable = _stable_replay(successful_candidate, metadata=metadata, target=target)
    fresh = _fresh_target_replay(target, seed=seed, successful_probe=successful_probe)

    checkpoint_dir = ROOT / "artifacts" / "payload-grounding" / PROTOCOL_ID
    checkpoint_path = checkpoint_dir / f"{target['target_id']}-{strategy}-seed{seed}.json"
    checkpoint = learner.save(checkpoint_path)
    attempts = warmup + replay
    return {
        "target_id": target["target_id"],
        "target_family_hidden_from_policy": target["family"],
        "seed": seed,
        "strategy": strategy,
        "policy_labels_visible": False,
        "candidate_count": len(pool),
        "candidate_generation_valid_rate": 1.0,
        "warmup_budget": WARMUP_BUDGET,
        "replay_budget": REPLAY_BUDGET,
        "warmup_first_success_step": warmup_success_step,
        "warmup_success_count": _success_count(warmup),
        "replay_first_success_step": replay_success_step,
        "replay_success_count": _success_count(replay),
        "replay_success_precision": _success_count(replay) / len(replay) if replay else 0.0,
        "stable_replay": stable,
        "fresh_target_replay": fresh,
        "evaluator_confirmation_count": sum(
            int(row.get("evaluator_confirmed", False)) for row in attempts
        ),
        "external_network": False,
        "database_touched": False,
        "script_execution": False,
        "real_sleep_performed": False,
        "attempts": attempts,
        "learner_summary": learner.summary(),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
    }


def _mean(values: list[int | float]) -> float | None:
    return round(sum(float(value) for value in values) / len(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_id"], row["strategy"])].append(row)
    result: list[dict[str, Any]] = []
    for (target_id, strategy), group in sorted(grouped.items()):
        warmup_hits = [row["warmup_first_success_step"] for row in group if row["warmup_first_success_step"] is not None]
        replay_hits = [row["replay_first_success_step"] for row in group if row["replay_first_success_step"] is not None]
        stable = [row["stable_replay"]["stable"] for row in group if row["stable_replay"]["available"]]
        fresh = [row["fresh_target_replay"]["success"] for row in group if row["fresh_target_replay"]["available"]]
        result.append({
            "target_id": target_id,
            "strategy": strategy,
            "seed_count": len(group),
            "warmup_success_rate": round(len(warmup_hits) / len(group), 4),
            "mean_warmup_first_success_step": _mean(warmup_hits),
            "replay_success_rate": round(len(replay_hits) / len(group), 4),
            "mean_replay_first_success_step": _mean(replay_hits),
            "mean_replay_success_precision": _mean([row["replay_success_precision"] for row in group]),
            "stable_replay_rate": round(sum(bool(value) for value in stable) / len(stable), 4) if stable else None,
            "fresh_target_replay_rate": round(sum(bool(value) for value in fresh) / len(fresh), 4) if fresh else None,
            "evaluator_confirmation_rate": 0.0,
        })
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PG-01 Payload Grounding Ablation",
        "",
        "本实验只运行本地合成 DOM/SQL oracle；候选是受限的安全 probe manifest，不发送网络请求、不执行脚本、不访问数据库。策略输入不包含 family/grammar/target 标签。",
        "",
        "| 目标 | 策略 | warmup 命中率 | warmup 首次命中 | replay 命中率 | replay 精度 | 稳定复放 | 新 marker 复放 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['target_id']} | {row['strategy']} | {row['warmup_success_rate']:.2f} | "
            f"{row['mean_warmup_first_success_step'] if row['mean_warmup_first_success_step'] is not None else '-'} | "
            f"{row['replay_success_rate']:.2f} | {row['mean_replay_success_precision']:.2f} | "
            f"{row['stable_replay_rate'] if row['stable_replay_rate'] is not None else '-'} | "
            f"{row['fresh_target_replay_rate'] if row['fresh_target_replay_rate'] is not None else '-'} |"
        )
    lines.extend([
        "",
        "结论边界：UCB 只证明‘在受限候选 grammar 上用 oracle 反馈做选择’的控制器能力；它不等于神经模型，也不表示已从公网渗透语料学习真实 payload。下一步若接入模型，替换候选生成器即可复用同一 evidence/replay 契约。",
        "",
        f"原始 JSON：`{report['report_path']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [
        run_one(target, seed=seed, strategy=strategy)
        for target in TARGETS
        for seed in SEEDS
        for strategy in STRATEGIES
    ]
    report_path = ROOT / "research" / "payload_grounding_ablation_v1.json"
    markdown_path = ROOT / "research" / "payload_grounding_ablation_v1.md"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "sift-payload-grounding-ablation-report-v1",
        "controls": {
            "policy_labels_visible": False,
            "public_payload_corpus_ingested": False,
            "evaluator_state_visible": False,
            "network_execution": False,
            "script_execution": False,
            "database_execution": False,
            "real_sleep_performed": False,
            "strategies": list(STRATEGIES),
            "seeds": list(SEEDS),
            "warmup_budget": WARMUP_BUDGET,
            "replay_budget": REPLAY_BUDGET,
        },
        "targets": [dict(target) for target in TARGETS],
        "aggregate": aggregate(rows),
        "runs": rows,
        "interpretation": [
            "warmup 衡量新候选池中的发现能力，replay 衡量是否能把反馈写入控制器记忆并优先选择成功 probe。",
            "hidden_target_match 只用于离线汇总；它没有进入 learner reward 之外的策略输入，evaluator confirmation 始终为 0。",
            "结果仅代表本地合成 oracle 与受限 grammar 的实验，不代表真实站点漏洞发现率。",
        ],
        "report_path": str(report_path.relative_to(ROOT)),
        "protocol_path": "research/payload_grounding_ablation_protocol_v1.json",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "aggregate": report["aggregate"],
        "report": str(report_path.relative_to(ROOT)),
        "markdown": str(markdown_path.relative_to(ROOT)),
        "run_count": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
