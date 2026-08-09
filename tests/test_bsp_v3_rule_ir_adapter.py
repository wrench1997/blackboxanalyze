import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_v3_rule_ir_adapter import (
    build_replay_package,
    compare_structural_signatures,
    make_fresh_lineage,
    structural_signature,
    structure_contract,
    structure_contract_sha256,
    validate_fresh_lineage,
    validate_replay_package,
)


def _assembly():
    return {
        "schema_version": "generic-rule-assembly-v1",
        "executable": False,
        "typed_oracle_required": True,
        "promotion_eligible": False,
        "decision": "await_typed_oracle",
        "slot": "p0",
        "methods": ["GET", "POST"],
        "atoms": ["effect_present", "get_post_repeat", "negative_control_clear", "probe_binding_valid"],
        "canonical_sha256": "a" * 64,
    }


def _plan():
    return {
        "schema_version": "bsp-capacity-pressure-v1",
        "target_id": "pg111-test-target",
        "unit_kind": "bsp_node",
        "action": "wake_target_unit",
        "capacity_before": 8,
        "capacity_after_proposed": 9,
        "observation_sha256": "b" * 64,
        "executable": False,
        "requires_rule_ir_replay": True,
        "requires_fresh_holdout": True,
        "promotion_eligible": False,
    }


def test_structure_contract_is_portable_and_does_not_include_weights_or_labels():
    contract = structure_contract()
    assert contract["schema_version"] == "bsp-v3-structure-contract-v1"
    assert set(contract["experts"]["unit_kinds"]) == {"bsp_page", "bsp_node", "expert_slot", "fragment_head"}
    assert contract["lineage"]["previous_weights_reuse_forbidden"] is True
    assert contract["lineage"]["mandarin_foundation_isolated"] is True
    assert "checkpoint_path" not in contract
    assert "family" not in contract
    assert len(structure_contract_sha256()) == 64


def test_fresh_lineage_rejects_old_checkpoint_reference_and_bad_commitment():
    lineage = make_fresh_lineage(contract_sha256=structure_contract_sha256())
    assert validate_fresh_lineage(lineage)["parent_checkpoint_reused"] is False
    legacy = dict(lineage, parent_checkpoint_path="checkpoints/old.bin")
    with pytest.raises(ValueError):
        validate_fresh_lineage(legacy)
    bad = dict(lineage, lineage_id="0" * 64)
    with pytest.raises(ValueError):
        validate_fresh_lineage(bad)


def test_replay_package_is_immutable_non_executable_and_backend_neutral():
    lineage = make_fresh_lineage(contract_sha256=structure_contract_sha256())
    package = build_replay_package(_assembly(), _plan(), lineage=lineage)
    assert validate_replay_package(package)["replay_sha256"] == package["replay_sha256"]
    assert package["executable"] is False
    assert package["mutates_weights"] is False
    assert package["mutates_topology"] is False
    assert package["training_eligible"] is False
    assert package["long_term_memory_write"] is False
    cpu = structural_signature(package, backend="cpu_contract")
    cuda = structural_signature(package, backend="cuda_contract")
    parity = compare_structural_signatures(cpu, cuda)
    assert parity["parity"] is True
    assert parity["numerical_forward_parity_claimed"] is False


def test_replay_package_rejects_tampering_and_evaluator_fields():
    lineage = make_fresh_lineage(contract_sha256=structure_contract_sha256())
    package = build_replay_package(_assembly(), _plan(), lineage=lineage)
    with pytest.raises(ValueError):
        validate_replay_package(dict(package, mutates_weights=True))
    with pytest.raises(ValueError):
        build_replay_package(dict(_assembly(), family="x"), _plan(), lineage=lineage)
