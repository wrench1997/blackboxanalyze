"""Immutable BSP v3 structure-contract adapter for Rule IR replay.

This module is deliberately not a model and it never loads or edits weights.
It translates a bounded, family-free Rule IR assembly into a structural replay
package that a BSP v3 Zig/CUDA smoke can inspect.  The package is an audit
artifact: it is non-executable, requires an evaluator-only oracle later, and
cannot be promoted to training or long-term memory.

The Mandarin foundation stage is kept as a separate lineage.  BSP v3 is
borrowed as an architecture contract only; an old checkpoint, optimizer state,
tokenizer, or resume path is never an input to this adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


REPLAY_SCHEMA_VERSION = "bsp-v3-rule-ir-replay-v1"
LINEAGE_SCHEMA_VERSION = "bsp-v3-fresh-lineage-v1"
SIGNATURE_SCHEMA_VERSION = "bsp-v3-structural-signature-v1"

_ALLOWED_UNITS = frozenset({"bsp_page", "bsp_node", "expert_slot", "fragment_head"})
_ALLOWED_ACTIONS = frozenset(
    {
        "hold_capacity",
        "wake_target_unit",
        "merge_then_ablate_low_contribution_units",
        "measure_speed_without_ablation",
        "hold_and_measure_tradeoff",
        "hold_and_repair_evidence",
        "hold_and_collect_cross_evidence",
    }
)

# This is the portable representation kept in blackboxanalyze.  It describes
# the BSP v3 Page/Node/Expert interfaces, not a checkpoint or a set of model
# weights.  The current project can therefore use the same structural ideas
# while the separate dog project remains free to pursue its Mandarin work.
STRUCTURE_CONTRACT = {
    "schema_version": "bsp-v3-structure-contract-v1",
    "pages": {
        "kind": "fixed_page_budget",
        "routing_signal": "bounded_page_mass",
        "mass_conservation_required": True,
    },
    "nodes": {
        "kind": "bounded_dynamic_node_pool",
        "topology": "runtime_split_merge_with_compact_plan",
        "capacity_is_fixed_budget": True,
    },
    "experts": {
        "kind": "routed_rank_factorized_expert_slots",
        "parameter_update": "outside_this_policy_adapter",
        "unit_kinds": sorted(_ALLOWED_UNITS),
    },
    "controller": {
        "typed_pressure_required": True,
        "wake_before_growth": True,
        "merge_before_ablation": True,
        "fresh_holdout_before_after": True,
        "rollback_on_regression": True,
    },
    "lineage": {
        "architecture_transfer_mode": "bsp_v3_structure_contract_only",
        "previous_weights_reuse_forbidden": True,
        "fresh_checkpoint_required": True,
        "mandarin_foundation_isolated": True,
    },
}
_FORBIDDEN_KEYS = frozenset(
    {
        "family",
        "family_name",
        "oracle",
        "oracle_id",
        "typed_oracle",
        "evaluator",
        "evaluator_label",
        "positive",
        "positive_authority",
        "raw_body",
        "raw_payload",
        "raw_request",
        "raw_response",
        "payload",
        "checkpoint_path",
        "parent_checkpoint",
        "parent_checkpoint_path",
        "resume_checkpoint",
        "weight_path",
        "optimizer_state",
        "resume",
    }
)
_HASH_KEYS = frozenset({"canonical_sha256", "observation_sha256", "contract_sha256", "lineage_id", "replay_sha256"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_KEYS or _contains_forbidden_key(child):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(child) for child in value)
    return False


def structure_contract() -> dict[str, Any]:
    """Return a copy of the portable BSP v3 contract."""

    return json.loads(json.dumps(STRUCTURE_CONTRACT, ensure_ascii=False))


def structure_contract_sha256() -> str:
    return sha256_value(STRUCTURE_CONTRACT)


def _require_hash(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 commitment")
    return text


def _bounded_text(value: Any, field: str, *, maximum: int = 128) -> str:
    text = str(value)
    if not text or len(text) > maximum or "\n" in text or "\r" in text:
        raise ValueError(f"{field} must be bounded single-line text")
    return text


def make_fresh_lineage(*, contract_sha256: str, foundation_stage: str = "mandarin_foundation") -> dict[str, Any]:
    """Create a deterministic fresh lineage without a checkpoint reference."""

    contract = _require_hash(contract_sha256, "contract_sha256")
    foundation = _bounded_text(foundation_stage, "foundation_stage")
    body = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "architecture_transfer_mode": "bsp_v3_structure_contract_only",
        "checkpoint_role": "fresh_structure_replay_smoke",
        "parent_checkpoint_reused": False,
        "weight_load_performed": False,
        "foundation_stage": foundation,
        "security_replay_stage": "post_training_security_trace",
        "dataset_merge": False,
        "contract_sha256": contract,
    }
    return {**body, "lineage_id": sha256_value(body)}


def validate_fresh_lineage(lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that lineage is architecture-only and cannot resume old weights."""

    if not isinstance(lineage, Mapping) or _contains_forbidden_key(lineage):
        raise ValueError("fresh lineage contains a legacy checkpoint, evaluator, or raw field")
    if str(lineage.get("schema_version")) != LINEAGE_SCHEMA_VERSION:
        raise ValueError("unsupported BSP v3 lineage schema")
    if str(lineage.get("architecture_transfer_mode")) != "bsp_v3_structure_contract_only":
        raise ValueError("BSP v3 lineage must transfer structure contract only")
    if lineage.get("parent_checkpoint_reused") is not False or lineage.get("weight_load_performed") is not False:
        raise ValueError("old weights/checkpoints are forbidden for this lineage")
    if lineage.get("dataset_merge") is not False:
        raise ValueError("Mandarin foundation and security replay datasets must remain isolated")
    _bounded_text(lineage.get("foundation_stage"), "foundation_stage")
    if str(lineage.get("security_replay_stage")) != "post_training_security_trace":
        raise ValueError("security replay must remain a separate post-training stage")
    contract = _require_hash(lineage.get("contract_sha256"), "contract_sha256")
    expected_body = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "architecture_transfer_mode": "bsp_v3_structure_contract_only",
        "checkpoint_role": "fresh_structure_replay_smoke",
        "parent_checkpoint_reused": False,
        "weight_load_performed": False,
        "foundation_stage": str(lineage["foundation_stage"]),
        "security_replay_stage": "post_training_security_trace",
        "dataset_merge": False,
        "contract_sha256": contract,
    }
    if _require_hash(lineage.get("lineage_id"), "lineage_id") != sha256_value(expected_body):
        raise ValueError("fresh lineage commitment does not match its fields")
    return dict(lineage)


def _validate_assembly(assembly: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(assembly, Mapping) or _contains_forbidden_key(assembly):
        raise ValueError("Rule IR assembly contains a forbidden evaluator/raw field")
    if str(assembly.get("schema_version")) != "generic-rule-assembly-v1":
        raise ValueError("BSP adapter accepts only generic Rule IR assemblies")
    if assembly.get("executable") is not False or assembly.get("promotion_eligible") is not False:
        raise ValueError("Rule IR replay must be non-executable and non-promotable")
    if assembly.get("typed_oracle_required") is not True:
        raise ValueError("Rule IR replay must defer confirmation to a typed oracle")
    if str(assembly.get("decision")) not in {"await_typed_oracle", "abstain"}:
        raise ValueError("only pending or abstaining Rule IR proposals can be replayed")
    slot = _bounded_text(assembly.get("slot"), "slot", maximum=64)
    methods = sorted({_bounded_text(method, "method", maximum=8).upper() for method in assembly.get("methods", [])})
    if methods != ["GET", "POST"]:
        raise ValueError("BSP Rule IR replay requires the bounded GET/POST pair")
    atoms = [str(atom) for atom in assembly.get("atoms", [])]
    if not atoms or len(atoms) > 16 or any(not atom or len(atom) > 64 for atom in atoms):
        raise ValueError("Rule IR atoms are outside the bounded replay contract")
    canonical = _require_hash(assembly.get("canonical_sha256"), "canonical_sha256")
    return {
        "schema_version": "generic-rule-assembly-v1",
        "decision": str(assembly["decision"]),
        "slot": slot,
        "methods": methods,
        "atoms": sorted(set(atoms)),
        "canonical_sha256": canonical,
    }


def _validate_capacity_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or _contains_forbidden_key(plan):
        raise ValueError("capacity plan contains a forbidden evaluator/raw field")
    if str(plan.get("schema_version")) != "bsp-capacity-pressure-v1":
        raise ValueError("unsupported BSP capacity plan schema")
    unit_kind = str(plan.get("unit_kind"))
    if unit_kind not in _ALLOWED_UNITS:
        raise ValueError("capacity plan has an unknown BSP unit kind")
    action = str(plan.get("action"))
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("capacity plan has an unknown non-bounded action")
    if plan.get("executable") is not False or plan.get("promotion_eligible") is not False:
        raise ValueError("capacity plan must remain non-executable and non-promotable")
    if plan.get("requires_rule_ir_replay") is not True or plan.get("requires_fresh_holdout") is not True:
        raise ValueError("capacity plan must require Rule IR replay and a fresh holdout")
    target_id = _bounded_text(plan.get("target_id"), "target_id")
    observation = _require_hash(plan.get("observation_sha256"), "observation_sha256")
    capacity_before = int(plan.get("capacity_before", -1))
    capacity_after = int(plan.get("capacity_after_proposed", -1))
    if capacity_before < 0 or capacity_after < 0:
        raise ValueError("capacity values must be non-negative")
    return {
        "schema_version": "bsp-capacity-pressure-v1",
        "target_id": target_id,
        "unit_kind": unit_kind,
        "action": action,
        "capacity_before": capacity_before,
        "capacity_after_proposed": capacity_after,
        "observation_sha256": observation,
    }


def build_replay_package(
    assembly: Mapping[str, Any],
    capacity_plan: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
    implementation: str = "bsp_v3_zig_cuda_structure_contract",
) -> dict[str, Any]:
    """Build an immutable, non-executable BSP Page/Node/Expert replay package."""

    normalized_assembly = _validate_assembly(assembly)
    normalized_plan = _validate_capacity_plan(capacity_plan)
    normalized_lineage = validate_fresh_lineage(lineage)
    implementation_name = _bounded_text(implementation, "implementation", maximum=96)
    target_ref = sha256_value({"target_id": normalized_plan["target_id"], "slot": normalized_assembly["slot"]})
    structural_event = {
        "unit_kind": normalized_plan["unit_kind"],
        "target_ref": target_ref,
        "action": normalized_plan["action"],
        "rule_ir_sha256": normalized_assembly["canonical_sha256"],
        "observation_sha256": normalized_plan["observation_sha256"],
    }
    package_body = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "implementation": implementation_name,
        "architecture_transfer_mode": "bsp_v3_structure_contract_only",
        "structure_contract_sha256": structure_contract_sha256(),
        "lineage_id": normalized_lineage["lineage_id"],
        "rule_ir": normalized_assembly,
        "structural_event": structural_event,
        "replay_sequence": ["rule_ir", "structural_event", "fresh_holdout_gate", "typed_oracle_handoff"],
        "executable": False,
        "mutates_weights": False,
        "mutates_topology": False,
        "typed_oracle_required": True,
        "training_eligible": False,
        "promotion_eligible": False,
        "long_term_memory_write": False,
    }
    return {**package_body, "replay_sha256": sha256_value(package_body)}


def validate_replay_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Verify package immutability commitments and safety flags."""

    if not isinstance(package, Mapping) or _contains_forbidden_key(package):
        raise ValueError("replay package contains forbidden legacy/evaluator/raw content")
    if str(package.get("schema_version")) != REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported BSP v3 replay package schema")
    if package.get("executable") is not False or package.get("mutates_weights") is not False or package.get("mutates_topology") is not False:
        raise ValueError("replay package must not execute or mutate model state")
    if package.get("typed_oracle_required") is not True or package.get("training_eligible") is not False or package.get("promotion_eligible") is not False:
        raise ValueError("replay package must defer oracle confirmation and promotion")
    _require_hash(package.get("lineage_id"), "lineage_id")
    _require_hash(package.get("replay_sha256"), "replay_sha256")
    body = {key: value for key, value in package.items() if key != "replay_sha256"}
    if sha256_value(body) != str(package["replay_sha256"]):
        raise ValueError("replay package commitment does not match its fields")
    return dict(package)


def structural_signature(package: Mapping[str, Any], *, backend: str) -> dict[str, Any]:
    """Return a backend-neutral signature; no forward pass or weight access."""

    normalized = validate_replay_package(package)
    backend_name = _bounded_text(backend, "backend", maximum=32)
    body = {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "backend": backend_name,
        "replay_sha256": normalized["replay_sha256"],
        "lineage_id": normalized["lineage_id"],
        "structural_event": normalized["structural_event"],
    }
    return {**body, "signature_sha256": sha256_value(body)}


def compare_structural_signatures(cpu: Mapping[str, Any], cuda: Mapping[str, Any]) -> dict[str, Any]:
    """Compare CPU/CUDA contract signatures, not numerical model outputs."""

    if _contains_forbidden_key(cpu) or _contains_forbidden_key(cuda):
        return {"schema_version": SIGNATURE_SCHEMA_VERSION, "parity": False, "reason": "forbidden_field"}
    fields = ("replay_sha256", "lineage_id", "structural_event")
    mismatches = [field for field in fields if cpu.get(field) != cuda.get(field)]
    return {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "parity": not mismatches,
        "mismatched_fields": mismatches,
        "comparison_scope": "structure_contract_only",
        "numerical_forward_parity_claimed": False,
    }


__all__ = [
    "LINEAGE_SCHEMA_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "SIGNATURE_SCHEMA_VERSION",
    "STRUCTURE_CONTRACT",
    "build_replay_package",
    "compare_structural_signatures",
    "make_fresh_lineage",
    "sha256_value",
    "structural_signature",
    "structure_contract",
    "structure_contract_sha256",
    "validate_fresh_lineage",
    "validate_replay_package",
]
