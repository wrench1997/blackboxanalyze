"""PG-111: Python BSP v3 Rule IR replay and structure smoke.

The experiment keeps a fast NumPy reference core in this project.  It does
not import the dog repository, load a checkpoint, mutate legacy weights, or
start Mandarin foundation training.  The adapter's CPU/CUDA names still mean
backend-neutral signatures; the Python reference core supplies the structural
split/merge/forward checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_v3_rule_ir_adapter import (  # noqa: E402
    build_replay_package,
    compare_structural_signatures,
    make_fresh_lineage,
    structural_signature,
    structure_contract,
    structure_contract_sha256,
    validate_fresh_lineage,
    validate_replay_package,
)
from app.bsp_v3_research_core import (  # noqa: E402
    BspV3Config,
    BspV3State,
    validate_fresh_manifest,
)


PROTOCOL_ID = "pg-pk-111-bsp-v3-adapter-smoke-v1"
PG109_REPORT_PATH = ROOT / "research" / "pg109_fragment_composition_report_v1.json"
PG110_REPORT_PATH = ROOT / "research" / "pg110_capacity_pressure_cycle_report_v1.json"
ADAPTER_PATH = ROOT / "app" / "bsp_v3_rule_ir_adapter.py"
CORE_PATH = ROOT / "app" / "bsp_v3_research_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_pg111_bsp_v3_adapter_smoke.py"
REPORT_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_report_v1.json"
PROPOSAL_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_proposal_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_protocol_v1.json"
DATASET_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_visible_dataset_v1.json"
TRACE_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg111_bsp_v3_adapter_smoke_report_v1.md"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    pg109 = _read(PG109_REPORT_PATH)
    pg110 = _read(PG110_REPORT_PATH)
    if pg109.get("status") != "passed_fragment_composition_diagnostic":
        raise ValueError("PG-111 requires the frozen PG-109 diagnostic source")
    if pg110.get("status") != "passed_capacity_pressure_diagnostic":
        raise ValueError("PG-111 requires the frozen PG-110 capacity source")
    cross_sample = pg109.get("cross_sample") or []
    if not cross_sample:
        raise ValueError("PG-109 has no bounded cross-sample Rule IR assembly")
    assembly = dict(cross_sample[0]["assembly"])
    decisions = pg110.get("capacity_decisions") or []
    speed = next((item for item in decisions if item.get("scenario_id") == "speed_redundancy"), None)
    if speed is None:
        raise ValueError("PG-110 has no fixed speed/redundancy capacity plan")
    plan = dict(speed["plan"])
    return pg109, pg110, assembly, plan


def _try_reject_old_lineage(lineage: dict[str, Any]) -> bool:
    legacy = dict(lineage)
    legacy["parent_checkpoint_path"] = "checkpoints/legacy.bin"
    try:
        validate_fresh_lineage(legacy)
    except ValueError:
        return True
    return False


def _try_reject_tamper(package: dict[str, Any]) -> bool:
    tampered = dict(package)
    tampered["mutates_weights"] = True
    try:
        validate_replay_package(tampered)
    except ValueError:
        return True
    return False


def run() -> dict[str, Any]:
    pg109, pg110, assembly, plan = _select_inputs()
    contract = structure_contract()
    contract_hash = structure_contract_sha256()
    lineage = make_fresh_lineage(contract_sha256=contract_hash)
    package = build_replay_package(assembly, plan, lineage=lineage)
    proposal = {
        "schema_version": "pg111-bsp-v3-adapter-proposal-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "structure_contract": contract,
        "structure_contract_sha256": contract_hash,
        "rule_ir_sha256": assembly["canonical_sha256"],
        "capacity_observation_sha256": plan["observation_sha256"],
        "lineage": lineage,
        "replay_package": package,
        "python_reference_core": {
            "schema_version": "bsp-v3-python-research-core-v1",
            "fresh_only": True,
            "old_weights_loaded": False,
            "training_started": False,
        },
        "decision": "await_typed_oracle",
        "executable": False,
        "promotion_eligible": False,
    }
    PROPOSAL_PATH.write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cpu_signature = structural_signature(package, backend="cpu_contract")
    cuda_signature = structural_signature(package, backend="cuda_contract")
    parity = compare_structural_signatures(cpu_signature, cuda_signature)
    python_core = BspV3State.fresh(
        BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2),
        seed=111,
    )
    contexts = np.asarray(
        [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]],
        dtype=np.float64,
    )
    page_mass = np.tile(np.asarray([[0.6, 0.4]], dtype=np.float64), (len(contexts), 1))
    before = python_core.forward(contexts, page_mass)
    split_event = python_core.split_leaf(0)
    after_split = python_core.forward(contexts, page_mass)
    merge_event = python_core.merge_internal(0)
    after_merge = python_core.forward(contexts, page_mass)
    fresh_manifest = validate_fresh_manifest(python_core.fresh_manifest())
    split_max_error = float(np.max(np.abs(before.expert_out - after_split.expert_out)))
    merge_max_error = float(np.max(np.abs(before.expert_out - after_merge.expert_out)))
    mass_max_error = float(np.max(np.abs(np.sum(after_split.leaf_mass_sum, axis=1) - 1.0)))
    core_plan = python_core.compile_plan()
    checks = {
        "source_pg109_frozen_diagnostic": pg109.get("capability_gate", {}).get("claim_allowed") is False,
        "source_pg110_frozen_diagnostic": pg110.get("capability_gate", {}).get("claim_allowed") is False,
        "portable_structure_contract_bounded": contract.get("schema_version") == "bsp-v3-structure-contract-v1" and len(contract) == 6,
        "fresh_lineage_valid": validate_fresh_lineage(lineage).get("parent_checkpoint_reused") is False,
        "old_checkpoint_reference_rejected": _try_reject_old_lineage(lineage),
        "replay_package_valid": validate_replay_package(package).get("replay_sha256") == package["replay_sha256"],
        "tamper_rejected": _try_reject_tamper(package),
        "structure_signature_parity": parity.get("parity") is True,
        "numerical_forward_parity_not_claimed": parity.get("numerical_forward_parity_claimed") is False,
        "family_and_evaluator_blind": all(key not in package for key in ("family", "oracle", "evaluator_label")),
        "raw_inputs_absent": all(key not in package for key in ("raw_payload", "raw_body", "raw_request", "raw_response")),
        "non_executable": package["executable"] is False and package["mutates_weights"] is False and package["mutates_topology"] is False,
        "mandarin_foundation_isolated": lineage["foundation_stage"] == "mandarin_foundation" and lineage["dataset_merge"] is False,
        "previous_weights_not_reused": lineage["weight_load_performed"] is False and lineage["parent_checkpoint_reused"] is False,
        "training_and_memory_blocked": package["training_eligible"] is False and package["long_term_memory_write"] is False,
        "python_reference_core_imported": python_core.config.max_nodes == 7 and python_core.config.max_pages == 2,
        "python_core_split_preserves_forward": split_max_error <= 1.0e-12,
        "python_core_merge_roundtrip": merge_max_error <= 1.0e-12,
        "python_core_page_mass_conserved": mass_max_error <= 1.0e-12,
        "python_core_fixed_budget": python_core.active_count + python_core.free_count == python_core.config.max_nodes,
        "python_core_fresh_manifest_valid": fresh_manifest["parent_checkpoint_reused"] is False and fresh_manifest["weight_load_performed"] is False,
    }
    blocked = [key for key, value in checks.items() if not value]
    status = "passed_bsp_v3_structure_adapter_smoke" if not blocked else "blocked"
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg111-bsp-v3-adapter-smoke-report-v1",
        "status": status,
        "scope": {
            "project": "blackboxanalyze",
            "migration": "python_bsp_v3_reference_core_plus_structure_contract",
            "dog_project_imported": False,
            "dog_weights_loaded": False,
            "mandarin_foundation_used": False,
            "full_training_started": False,
            "cuda_execution_started": False,
        },
        "source": {
            "pg109_report": str(PG109_REPORT_PATH.relative_to(ROOT)),
            "pg110_report": str(PG110_REPORT_PATH.relative_to(ROOT)),
            "source_hashes": {
                "pg109_report": _sha256_file(PG109_REPORT_PATH),
                "pg110_report": _sha256_file(PG110_REPORT_PATH),
                "adapter": _sha256_file(ADAPTER_PATH),
                "python_core": _sha256_file(CORE_PATH),
                "runner": _sha256_file(RUNNER_PATH),
            },
        },
        "contract": {
            "schema_version": contract["schema_version"],
            "sha256": contract_hash,
            "page_node_expert": True,
            "fixed_budget": True,
            "split_merge_compact_plan": True,
            "typed_pressure_and_rollback": True,
            "python_reference_core": True,
        },
        "lineage": lineage,
        "replay": {
            "replay_sha256": package["replay_sha256"],
            "rule_ir_sha256": package["rule_ir"]["canonical_sha256"],
            "capacity_observation_sha256": package["structural_event"]["observation_sha256"],
            "executable": package["executable"],
            "typed_oracle_required": package["typed_oracle_required"],
            "training_eligible": package["training_eligible"],
            "promotion_eligible": package["promotion_eligible"],
        },
        "backend_signatures": {
            "cpu": cpu_signature,
            "cuda": cuda_signature,
            "comparison": parity,
        },
        "python_reference_core": {
            "schema_version": "bsp-v3-python-research-core-v1",
            "config": python_core.config.__dict__,
            "seed": python_core.seed,
            "split_event": {"parent": split_event[0], "children": list(split_event[1:])},
            "merge_event": {"parent": merge_event[0], "reclaimed": list(merge_event[1:])},
            "split_forward_max_abs_error": split_max_error,
            "merge_roundtrip_max_abs_error": merge_max_error,
            "page_mass_max_abs_error": mass_max_error,
            "active_count": python_core.active_count,
            "free_count": python_core.free_count,
            "compiled_leaf_count": core_plan.leaf_count,
            "topology_version": core_plan.topology_version,
            "fresh_manifest_sha256": fresh_manifest["manifest_sha256"],
            "old_weights_loaded": False,
            "training_started": False,
        },
        "checks": checks,
        "capability_gate": {
            "status": status,
            "checks": checks,
            "blocking_reasons": blocked,
            "claim_allowed": False,
        },
        "promotion": {
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "status": "structure_adapter_evaluation_only",
            "reason": "Rule IR remains an evaluator-only handoff; BSP contract smoke is not model training or vulnerability confirmation",
        },
        "safety": {
            "loopback_only": True,
            "external_network": False,
            "dog_project_modified": False,
            "old_checkpoint_read": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
            "family_labels_in_model_input": False,
            "evaluator_labels_in_model_input": False,
            "typed_oracle_called": False,
            "fresh_lineage_required": True,
            "mandarin_foundation_isolated": True,
            "long_term_memory_write": False,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dataset = {
        "schema_version": "pg111-bsp-v3-adapter-visible-dataset-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "rows": [
            {
                "row_id": "pg111-structure-contract-replay-0001",
                "replay_sha256": package["replay_sha256"],
                "structure_contract_sha256": contract_hash,
                "lineage_id": lineage["lineage_id"],
                "backend_neutral_signature_sha256": cpu_signature["signature_sha256"],
                "python_core_state_sha256": python_core.state_sha256(),
                "python_core_split_forward_max_abs_error": split_max_error,
                "python_core_merge_roundtrip_max_abs_error": merge_max_error,
                "typed_oracle_required": True,
                "raw_payload_stored": False,
                "raw_response_body_stored": False,
                "family_label_stored": False,
                "training_eligible": False,
            }
        ],
    }
    DATASET_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = {
        "schema_version": "pg111-bsp-v3-adapter-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "steps": [
            {
                "step": 0,
                "event": "immutable_rule_ir_replay",
                "replay_sha256": package["replay_sha256"],
                "structural_event": package["structural_event"],
                "decision": "await_typed_oracle",
                "typed_oracle_called": False,
                "confirmed_positive": False,
                "fresh_lineage": True,
                "mutates_weights": False,
                "mutates_topology": False,
                "python_core": {
                    "split_forward_max_abs_error": split_max_error,
                    "merge_roundtrip_max_abs_error": merge_max_error,
                    "page_mass_max_abs_error": mass_max_error,
                    "topology_version_after_roundtrip": core_plan.topology_version,
                },
            }
        ],
    }
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "scope": "current blackboxanalyze project only",
        "architecture_transfer": "Python BSP v3 Page/Node/Expert reference core plus structure contract; no old weights",
        "foundation_isolation": "Mandarin foundation is an independent future stage and is not mixed into security replay",
        "required_gates": ["fresh lineage", "immutable package hash", "Python split/merge forward invariance", "page-mass conservation", "CPU/CUDA structural signature equality", "typed oracle handoff", "no promotion"],
        "out_of_scope": ["full training", "checkpoint migration", "numerical CUDA forward parity", "vulnerability confirmation"],
        "source_reports": [str(PG109_REPORT_PATH.relative_to(ROOT)), str(PG110_REPORT_PATH.relative_to(ROOT))],
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(
        "# PG-111 Python BSP v3 结构与回放 smoke\n\n"
        f"状态：`{status}`。本轮把 BSP v3 的 Page/Node/Expert 结构契约和 NumPy Python reference core 迁入 `blackboxanalyze`，不读取 dog 项目的权重，也不启动普通话基础训练。\n\n"
        f"- contract SHA-256: `{contract_hash}`\n"
        f"- replay SHA-256: `{package['replay_sha256']}`\n"
        f"- lineage: `{lineage['lineage_id']}`（fresh；旧 checkpoint 拒绝）\n"
        f"- Python split forward 最大误差：`{split_max_error:.3e}`；merge round-trip 最大误差：`{merge_max_error:.3e}`；page mass 最大误差：`{mass_max_error:.3e}`。\n"
        "- CPU/CUDA 只比较结构签名，不宣称数值 forward parity。\n"
        "- Rule IR 仍等待 evaluator-only typed oracle；不生成训练样本、不写长期记忆。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
