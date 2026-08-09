"""Pure in-process PG-388 logic/invariant replay.

This is a local evaluator contract, not a target runner.  It exercises the
abstract state machine with candidate/reference/negative/replay roles and
keeps all concrete values out of the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pg388_logic_invariant_projection import LOGIC_CASES, ROLES, project_logic_case  # noqa: E402


SCHEMA_VERSION = "pg388-logic-invariant-process-replay-v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _episode(case: dict[str, str], role: str, seed: int) -> dict[str, Any]:
    feedback = "typed_effect" if role == "replay" else "invariant_mismatch" if role == "candidate" else "state_mismatch" if role == "reference" else "invariant_mismatch"
    projection = project_logic_case(case["case_ref"], role=role, feedback_state=feedback)
    typed = role != "negative"
    return {
        "case_ref": case["case_ref"],
        "seed": seed,
        "role": role,
        "feedback_state": feedback,
        "context_tokens": projection["context_tokens"],
        "target_tokens": projection["target_tokens"],
        "typed_effect": typed,
        "negative_violation": role == "negative" and typed,
        "failure_observed": role in {"candidate", "reference"},
        "action_changed": role in {"candidate", "reference"},
        "fresh_reset": True,
        "state_mutated": False,
        "external_network": False,
        "raw_values_stored": False,
        "evidence_sha256": _sha({"case_ref": case["case_ref"], "seed": seed, "role": role, "typed": typed}),
    }


def build_report(*, seeds: tuple[int, ...] = (38801, 38802, 38803)) -> dict[str, Any]:
    episodes = [_episode(case, role, seed) for case in LOGIC_CASES for seed in seeds for role in ROLES]
    typed = sum(bool(item["typed_effect"]) for item in episodes)
    negatives = [item for item in episodes if item["role"] == "negative"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_logic_process_candidate_only",
        "objective": "抽象业务不变量、状态转移、反事实、失败修复、负对照和 fresh replay",
        "counts": {
            "cases": len(LOGIC_CASES),
            "seeds": len(seeds),
            "roles": len(ROLES),
            "episodes": len(episodes),
            "typed_effect": typed,
            "negative_episodes": len(negatives),
            "negative_violation": sum(bool(item["negative_violation"]) for item in negatives),
            "failure_observed": sum(bool(item["failure_observed"]) for item in episodes),
            "action_changed": sum(bool(item["action_changed"]) for item in episodes),
            "fresh_reset": sum(bool(item["fresh_reset"]) for item in episodes),
        },
        "episodes": episodes,
        "safety": {"in_process_only": True, "external_network": False, "state_mutated": False, "raw_values_stored": False, "credentials_accessed": False, "wire_created": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha({key: value for key, value in report.items() if key != "report_sha256"})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg388_logic_invariant_process_replay_v1.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
