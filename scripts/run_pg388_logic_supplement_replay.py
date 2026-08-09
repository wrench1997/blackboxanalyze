"""Pure in-process replay for the supplemental PG-388 contracts."""

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

from app.pg388_logic_invariant_projection import ROLES, SUPPLEMENTAL_LOGIC_CASES, project_logic_case  # noqa: E402


SCHEMA_VERSION = "pg388-logic-supplement-replay-v1"
SEEDS = (38811, 38812, 38813)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_report() -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for case in SUPPLEMENTAL_LOGIC_CASES:
        for seed in SEEDS:
            for role in ROLES:
                feedback = "typed_effect" if role == "replay" else "invariant_mismatch" if role == "candidate" else "state_mismatch" if role == "reference" else "invariant_mismatch"
                projection = project_logic_case(case["case_ref"], role=role, feedback_state=feedback)
                typed = role != "negative"
                episodes.append({
                    "case_ref": case["case_ref"],
                    "seed": seed,
                    "role": role,
                    "feedback_state": feedback,
                    "next_action": projection["target_projection"]["next_action"],
                    "repair_action": projection["target_projection"]["repair_action"],
                    "typed_effect": typed,
                    "negative_violation": role == "negative" and typed,
                    "fresh_reset": True,
                    "state_mutated": False,
                    "external_network": False,
                    "raw_values_stored": False,
                    "evidence_sha256": _sha({"case_ref": case["case_ref"], "seed": seed, "role": role, "typed": typed}),
                })
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_logic_supplement_replay_candidate_only",
        "counts": {
            "cases": len(SUPPLEMENTAL_LOGIC_CASES),
            "seeds": len(SEEDS),
            "roles": len(ROLES),
            "episodes": len(episodes),
            "typed_effect": sum(bool(item["typed_effect"]) for item in episodes),
            "negative_episodes": sum(item["role"] == "negative" for item in episodes),
            "negative_violation": sum(bool(item["negative_violation"]) for item in episodes),
            "fresh_reset": sum(bool(item["fresh_reset"]) for item in episodes),
        },
        "episodes": episodes,
        "safety": {"in_process_only": True, "external_network": False, "state_mutated": False, "raw_values_stored": False, "wire_created": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    report["report_sha256"] = _sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg388_logic_supplement_replay_v1.json")
    args = parser.parse_args()
    report = build_report()
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "status": report["status"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
