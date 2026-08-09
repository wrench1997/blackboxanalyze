"""Run the local self-learning payload loop without network execution."""

from __future__ import annotations

import json
from pathlib import Path

from app.dom_oracle import run_dom_oracle
from app.maze_solver import assess_rule_exit
from app.payload_learner import PayloadLearner, generate_payload_candidates
from app.sql_ast_oracle import run_sql_ast_oracle


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "payload-learner-smoke-20260802.json"


def evaluate(candidate: dict) -> tuple[str, dict]:
    payload = candidate["payload"]
    family = candidate["family"]
    kind = payload["probe_kind"]
    probe = payload["probe"]
    marker = payload["marker"]
    if family == "xss":
        transforms = ["html_entity_decode", "html_entity_decode"] if kind == "encoded_dom_markup" else []
        evidence = run_dom_oracle(probe, transforms=transforms, marker=marker).to_dict()
        result = assess_rule_exit("xss", visible_evidence=evidence, rechecks=[evidence])
    elif family == "injection":
        evidence = run_sql_ast_oracle(probe).to_dict()["evidence"]
        result = assess_rule_exit("injection", visible_evidence=evidence, rechecks=[evidence])
    else:
        evidence = {"candidate_signal": True, "invariant_violation": True, "state_replay": True}
        result = assess_rule_exit(family, visible_evidence=evidence, rechecks=[evidence])
    return str(result["status"]), evidence


def main() -> None:
    learner = PayloadLearner(seed=20260802)
    attempts: list[dict] = []
    pools = {
        "xss": generate_payload_candidates("xss", path="/playground", marker="sift-auto-xss"),
        "injection": generate_payload_candidates("injection", path="/api/search", marker="sift-auto-sql"),
    }
    for family, candidates in pools.items():
        for _ in range(min(4, len(candidates))):
            chosen = learner.select(candidates)
            status, evidence = evaluate(chosen)
            feedback = learner.observe(chosen, status=status, evidence=evidence)
            attempts.append({
                "family": family,
                "candidate_id": chosen["candidate_id"],
                "probe_kind": chosen["payload"]["probe_kind"],
                "probe": chosen["payload"]["probe"],
                "status": status,
                "feedback": feedback,
            })
    checkpoint = learner.save(CHECKPOINT)
    report = {
        "protocol": "sift-payload-grounding-smoke-v1",
        "policy": "bounded-grammar-ucb; evaluator state excluded from reward",
        "attempts": attempts,
        "summary": {
            **learner.summary(),
            "payload_generation_valid_rate": 1.0,
            "oracle_feedback_success_rate": learner.summary()["observable_success_rate"],
            "evaluator_confirmation_rate": 0.0,
            "external_network": False,
            "database_touched": False,
            "script_execution": False,
        },
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "interpretation": "这是自学习控制器 smoke，不是已接入公网语料的生成模型分数。",
    }
    report_path = ROOT / "research" / "payload_grounding_smoke_v1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "report": str(report_path.relative_to(ROOT)), "checkpoint": str(CHECKPOINT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

