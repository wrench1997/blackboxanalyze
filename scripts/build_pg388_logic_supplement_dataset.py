"""Build the PG-388 supplemental 4.13 logic-contract dataset."""

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

from app.pg388_logic_invariant_projection import (  # noqa: E402
    FEEDBACK_STATES,
    ROLES,
    SUPPLEMENTAL_LOGIC_CASES,
    project_logic_case,
)


SCHEMA_VERSION = "pg388-logic-supplement-dataset-v1"
IMPLEMENTATIONS = ("logic_supplement_fixture_a", "logic_supplement_fixture_b")
SEEDS = (38811, 38812, 38813, 38814)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        split = "train" if implementation_index == 0 else "implementation_holdout"
        for case in SUPPLEMENTAL_LOGIC_CASES:
            for seed in SEEDS:
                for feedback_state in FEEDBACK_STATES:
                    for role in ROLES:
                        projection = project_logic_case(case["case_ref"], role=role, feedback_state=feedback_state)
                        core = {
                            "record_ref_sha256": _sha({"implementation": implementation, "case_ref": case["case_ref"], "seed": seed, "feedback_state": feedback_state, "role": role}),
                            "split": split,
                            "implementation_ref": implementation,
                            "seed_bucket": f"seed_{seed % 2}",
                            "case_ref": case["case_ref"],
                            "feedback_state": feedback_state,
                            "role": role,
                            "context_tokens": projection["context_tokens"],
                            "target_tokens": projection["target_tokens"],
                            "logic_context": projection["logic_context"],
                            "target_projection": projection["target_projection"],
                            "raw_source_stored": False,
                            "raw_payload_stored": False,
                            "raw_response_body_stored": False,
                            "oracle_answer_in_context": False,
                            "fresh_reset": False,
                            "typed_evaluator_observed": False,
                            "training_eligible": False,
                        }
                        row = dict(core)
                        row["row_sha256"] = _sha(core)
                        rows.append(row)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg388_logic_supplement_dataset_v1",
        "status": "abstract_logic_supplement_candidate_only",
        "objective": "补齐 4.13 中 OAuth/激活/CSRF-2FA、CAPTCHA 细项和 Session 细项的抽象不变量上下文",
        "rows": rows,
        "counts": {
            "records": len(rows),
            "train": sum(row["split"] == "train" for row in rows),
            "implementation_holdout": sum(row["split"] == "implementation_holdout" for row in rows),
            "cases": len(SUPPLEMENTAL_LOGIC_CASES),
            "implementations": len(IMPLEMENTATIONS),
            "seeds": len(SEEDS),
            "feedback_states": len(FEEDBACK_STATES),
            "roles": len(ROLES),
        },
        "case_refs": [item["case_ref"] for item in SUPPLEMENTAL_LOGIC_CASES],
        "context_firewall": {"raw_source": False, "raw_payload": False, "raw_response": False, "evaluator_answer": False, "external_network": False},
        "source_contract": {"fresh_role_reset": False, "candidate_reference_negative_replay": False, "typed_evidence": False, "operator_reviewed": False, "live_rows_emitted": False},
        "training_eligible": 0,
        "promotion": {"training_allowed": False, "memory_promotion_allowed": False, "payload_catalog_promotion_allowed": False, "vulnerability_claim_allowed": False},
    }
    artifact["dataset_sha256"] = _sha({key: value for key, value in artifact.items() if key != "dataset_sha256"})
    return artifact


def write_dataset(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_dataset(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/pg388_logic_supplement_dataset_v1.json")
    args = parser.parse_args()
    output = write_dataset(args.output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": artifact["status"], "counts": artifact["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
