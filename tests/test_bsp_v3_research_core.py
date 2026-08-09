import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bsp_v3_research_core import (  # noqa: E402
    BspV3Config,
    BspV3State,
    validate_fresh_manifest,
)


def _inputs(samples: int = 4):
    contexts = np.arange(samples * 4, dtype=np.float64).reshape(samples, 4) / 10.0
    masses = np.tile(np.asarray([[0.6, 0.4]], dtype=np.float64), (samples, 1))
    return contexts, masses


def test_python_bsp_v3_split_and_merge_are_function_preserving_at_fresh_state():
    state = BspV3State.fresh(BspV3Config(max_pages=2, max_nodes=7, d_model=4, expert_rank=2), seed=11)
    contexts, masses = _inputs()
    before = state.forward(contexts, masses)
    split = state.split_leaf(0)
    after_split = state.forward(contexts, masses)
    assert split[0] == 0
    assert np.max(np.abs(before.expert_out - after_split.expert_out)) <= 1e-12
    assert np.max(np.abs(np.sum(after_split.leaf_mass_sum, axis=1) - 1.0)) <= 1e-12
    assert state.compile_plan().leaf_count == 3
    merge = state.merge_internal(0)
    after_merge = state.forward(contexts, masses)
    assert merge[0] == 0
    assert np.max(np.abs(before.expert_out - after_merge.expert_out)) <= 1e-12
    assert state.active_count == 2
    assert state.free_count == 5


def test_python_bsp_v3_capacity_and_action_policy_fail_closed_without_target():
    state = BspV3State.fresh(BspV3Config(max_pages=1, max_nodes=3, d_model=2, expert_rank=1), seed=12)
    held = state.apply_structural_action("hold_capacity")
    assert held["mutated"] is False
    abstain = state.apply_structural_action("wake_target_unit")
    assert abstain["status"] == "abstain"
    state.apply_structural_action("wake_target_unit", target_node_id=0)
    with pytest.raises(ValueError):
        state.split_leaf(0)
    state.assert_invariants()


def test_python_bsp_v3_fresh_manifest_rejects_old_checkpoint_and_tamper():
    state = BspV3State.fresh(BspV3Config(max_pages=1, max_nodes=3, d_model=2, expert_rank=1), seed=13)
    manifest = state.fresh_manifest()
    assert validate_fresh_manifest(manifest)["weight_load_performed"] is False
    with pytest.raises(ValueError):
        validate_fresh_manifest(dict(manifest, parent_checkpoint_path="old.bin"))
    with pytest.raises(ValueError):
        validate_fresh_manifest(dict(manifest, manifest_sha256="0" * 64))
