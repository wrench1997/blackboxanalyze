"""Build abstract trajectories from the three PG-388 local logic canaries.

This is a candidate-only trajectory set.  It preserves state-transition
shapes and Rule-IR targets but never copies evaluator answers into context and
never stores identifiers, prices, coupons, tokens, wire, or response values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pg388-logic-canary-trajectory-dataset-v1"
IMPLEMENTATIONS = ("logic_canary_fixture_a", "logic_canary_fixture_b")
SEEDS = (38821, 38822, 38823)
ROLES = ("candidate", "reference", "negative", "replay")
PHASES = ("baseline", "candidate", "reference", "negative", "replay")
CASES = {
    "nonce_replay": {"surface": "replay_protection", "invariant": "nonce_fresh_per_action", "repair": "replay"},
    "coupon_reuse_boundary": {"surface": "coupon_reuse", "invariant": "coupon_consumed_once", "repair": "replay"},
    "subject_resource_scope": {"surface": "horizontal_authorization", "invariant": "subject_matches_resource", "repair": "scope"},
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _state_before(case_ref: str, phase: str) -> str:
    if phase in {"baseline", "candidate", "reference", "negative"}:
        return "zero_effect_or_unknown_scope"
    if case_ref in {"nonce_replay", "coupon_reuse_boundary"}:
        return "prior_effect_observed"
    return "cross_subject_scope_tested"


def _outcome(case_ref: str, role: str, phase: str) -> dict[str, str | bool]:
    if phase == "baseline":
        return {"next_action": "select_probe_variant", "repair_action": "none", "effect_shape": "baseline_shape", "state_delta": "zero", "invariant_result": "holds", "vulnerable_effect": False}
    if role == "negative":
        return {"next_action": "abstain", "repair_action": "none", "effect_shape": "denied_shape", "state_delta": "zero", "invariant_result": "holds", "vulnerable_effect": False}
    if role == "reference":
        effect = "coupon_reuse_denied" if case_ref == "coupon_reuse_boundary" else "rejected_or_denied"
        return {"next_action": "confirm_reference", "repair_action": "none", "effect_shape": effect, "state_delta": "zero", "invariant_result": "holds", "vulnerable_effect": False}
    if case_ref == "nonce_replay":
        if phase == "candidate":
            return {"next_action": "replay", "repair_action": "none", "effect_shape": "accepted_once", "state_delta": "one_effect", "invariant_result": "holds", "vulnerable_effect": False}
        return {"next_action": "abstain", "repair_action": "replay", "effect_shape": "accepted_replay", "state_delta": "duplicate_effect", "invariant_result": "violated", "vulnerable_effect": True}
    if case_ref == "coupon_reuse_boundary":
        if phase == "candidate":
            return {"next_action": "replay", "repair_action": "none", "effect_shape": "benefit_applied", "state_delta": "discount_once", "invariant_result": "holds", "vulnerable_effect": False}
        return {"next_action": "abstain", "repair_action": "replay", "effect_shape": "benefit_applied_again", "state_delta": "discount_reused", "invariant_result": "violated", "vulnerable_effect": True}
    return {"next_action": "abstain", "repair_action": "scope", "effect_shape": "resource_visible", "state_delta": "read_cross_scope", "invariant_result": "violated", "vulnerable_effect": True}


def build_dataset() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for implementation_index, implementation in enumerate(IMPLEMENTATIONS):
        split = "train" if implementation_index == 0 else "implementation_holdout"
        for case_ref, case in CASES.items():
            for seed in SEEDS:
                for role, phase in (("candidate", "baseline"), ("candidate", "candidate"), ("reference", "reference"), ("negative", "negative"), ("replay", "replay")):
                    outcome = _outcome(case_ref, role, phase)
                    core = {
                        "record_ref_sha256": _sha({"implementation": implementation, "case_ref": case_ref, "seed": seed, "role": role, "phase": phase}),
                        "split": split,
                        "implementation_ref": implementation,
                        "seed_bucket": f"seed_{seed % 3}",
                        "case_ref": case_ref,
                        "role": role,
                        "phase": phase,
                        "context_tokens": [
                            f"canary_surface={case['surface']}",
                            f"logic_invariant={case['invariant']}",
                            f"phase={phase}",
                            f"role={role}",
                            f"state_before={_state_before(case_ref, phase)}",
                            "history=bounded_local_sequence",
                            "fresh_reset=required",
                            "oracle_mode=typed_shape_only",
                            "safe_to_send=false",
                        ],
                        "target_tokens": [
                            f"question={'ask' if phase in {'baseline', 'replay'} else 'observe'}",
                            f"ask_reason={'missing_state' if phase == 'baseline' else 'none'}",
                            f"logic_invariant_ref={case['invariant']}",
                            f"state_transition_ref={case['surface']}",
                            "precondition_ref=fresh_reset_required",
                            f"counterfactual_ref={phase}",
                            f"probe_variant_ref={role}_{phase}",
                            f"next_action={outcome['next_action']}",
                            f"repair_action={outcome['repair_action']}",
                            f"effect_shape={outcome['effect_shape']}",
                            f"state_delta={outcome['state_delta']}",
                            f"invariant_result={outcome['invariant_result']}",
                            "oracle_ref=typed_state_shape",
                            "safe_to_send=false",
                        ],
                        "evaluator_sidecar": {
                            "scope": "local_disposable_canary_only",
                            "vulnerable_effect": outcome["vulnerable_effect"],
                            "negative_control_clean": role == "negative" and not outcome["vulnerable_effect"],
                        },
                        "raw_source_stored": False,
                        "raw_payload_stored": False,
                        "raw_response_body_stored": False,
                        "oracle_answer_in_context": False,
                        "fresh_reset_attested": False,
                        "typed_evaluator_observed": False,
                        "training_eligible": False,
                    }
                    row = dict(core)
                    row["row_sha256"] = _sha(core)
                    rows.append(row)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "pg388_logic_canary_trajectory_dataset_v1",
        "status": "abstract_canary_trajectory_candidate_only",
        "objective": "让模型学习有限业务状态机的 baseline/candidate/reference/negative/replay 组合，而不是记忆攻击字符串",
        "rows": rows,
        "counts": {"records": len(rows), "train": 45, "implementation_holdout": 45, "cases": len(CASES), "implementations": len(IMPLEMENTATIONS), "seeds": len(SEEDS), "phases": len(PHASES), "roles": len(ROLES)},
        "case_refs": list(CASES),
        "context_firewall": {"raw_source": False, "raw_payload": False, "raw_response": False, "evaluator_answer": False, "external_network": False},
        "source_contract": {"fresh_role_reset": False, "candidate_reference_negative_replay": True, "typed_evidence": False, "operator_reviewed": False, "live_rows_emitted": False},
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
    parser.add_argument("--output", default="research/pg388_logic_canary_trajectory_dataset_v1.json")
    args = parser.parse_args()
    output = write_dataset(args.output)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": artifact["status"], "counts": artifact["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
