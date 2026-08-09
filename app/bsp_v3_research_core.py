"""Pure-Python BSP v3 reference core for fast research iteration.

The reference core is intentionally small and deterministic.  It models the
structural contract used by the Rule IR adapter: a fixed Page/Node/Expert
budget, runtime split/merge, compact execution plans, fresh initialization,
and a forward pass that exposes page-mass conservation.  It is a research
oracle/simulator, not a production trainer and never loads legacy weights.

The implementation uses NumPy because it keeps experiments readable while
leaving room for later replacement of a single kernel or a learned router.
No security payloads, evaluator labels, or raw web responses belong here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "bsp-v3-python-research-core-v1"
FRESH_MANIFEST_SCHEMA_VERSION = "bsp-v3-python-fresh-state-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BspV3Config:
    max_pages: int = 2
    max_nodes: int = 15
    d_model: int = 8
    expert_rank: int = 4
    route_temperature: float = 1.0

    def validate(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be positive")
        if self.max_nodes < self.max_pages:
            raise ValueError("max_nodes must cover one root per page")
        if self.d_model < 1 or self.expert_rank < 1:
            raise ValueError("model dimensions must be positive")
        if not math.isfinite(self.route_temperature) or self.route_temperature <= 0:
            raise ValueError("route_temperature must be finite and positive")


@dataclass
class BspNode:
    node_id: int
    page: int
    parent: int | None = None
    left: int | None = None
    right: int | None = None
    depth: int = 0
    usage: float = 0.0
    active: bool = True

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class ExecutionPlan:
    topology_version: int
    roots: tuple[int, ...]
    leaf_entries: tuple[dict[str, int | bool], ...]

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_entries)


@dataclass(frozen=True)
class ForwardOutput:
    expert_out: np.ndarray
    leaf_mass_sum: np.ndarray
    selected_leaf_ids: np.ndarray


class BspV3State:
    """Deterministic fixed-budget Page/Node/Expert state."""

    def __init__(self, config: BspV3Config, *, seed: int) -> None:
        config.validate()
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.topology_version = 0
        self.nodes: list[BspNode | None] = [None] * config.max_nodes
        self._free_ids: set[int] = set(range(config.max_pages, config.max_nodes))
        self.roots: list[int] = []
        self.route_w = self.rng.normal(0.0, 0.05, size=(config.max_nodes, config.d_model)).astype(np.float64)
        self.route_b = np.zeros(config.max_nodes, dtype=np.float64)
        self.expert_u = self.rng.normal(0.0, 0.05, size=(config.max_nodes, config.d_model, config.expert_rank)).astype(np.float64)
        self.expert_v = self.rng.normal(0.0, 0.05, size=(config.max_nodes, config.expert_rank, config.d_model)).astype(np.float64)
        self.expert_b = np.zeros((config.max_nodes, config.d_model), dtype=np.float64)
        for page in range(config.max_pages):
            node_id = page
            self.nodes[node_id] = BspNode(node_id=node_id, page=page)
            self.roots.append(node_id)
        self.assert_invariants()

    @classmethod
    def fresh(cls, config: BspV3Config | None = None, *, seed: int = 20260803) -> "BspV3State":
        """Create a fresh state; no checkpoint or optimizer state is accepted."""

        return cls(config or BspV3Config(), seed=seed)

    @property
    def active_count(self) -> int:
        return sum(int(node is not None and node.active) for node in self.nodes)

    @property
    def free_count(self) -> int:
        return len(self._free_ids)

    @property
    def active_budget_remaining(self) -> int:
        return self.config.max_nodes - self.active_count

    def _node(self, node_id: int) -> BspNode:
        if not isinstance(node_id, int) or node_id < 0 or node_id >= self.config.max_nodes:
            raise ValueError("node id is outside the fixed budget")
        node = self.nodes[node_id]
        if node is None or not node.active:
            raise ValueError("node is not active")
        return node

    def active_leaves(self) -> tuple[int, ...]:
        return tuple(node.node_id for node in self.nodes if node is not None and node.active and node.is_leaf)

    def active_nodes(self) -> tuple[int, ...]:
        return tuple(node.node_id for node in self.nodes if node is not None and node.active)

    def compile_plan(self) -> ExecutionPlan:
        entries: list[dict[str, int | bool]] = []
        for node_id in self.active_leaves():
            node = self._node(node_id)
            entries.append(
                {
                    "stable_id": node.node_id,
                    "page": node.page,
                    "parent": node.parent if node.parent is not None else -1,
                    "depth": node.depth,
                    "is_leaf": True,
                }
            )
        return ExecutionPlan(self.topology_version, tuple(self.roots), tuple(entries))

    def split_leaf(self, node_id: int) -> tuple[int, int, int]:
        parent = self._node(node_id)
        if not parent.is_leaf:
            raise ValueError("only a leaf can be split")
        if len(self._free_ids) < 2:
            raise ValueError("fixed BSP capacity exhausted")
        child_ids = tuple(sorted(self._free_ids))[:2]
        for child_id in child_ids:
            self._free_ids.remove(child_id)
            self.nodes[child_id] = BspNode(
                node_id=child_id,
                page=parent.page,
                parent=parent.node_id,
                depth=parent.depth + 1,
                usage=parent.usage,
            )
            self.route_w[child_id] = self.route_w[parent.node_id]
            self.route_b[child_id] = self.route_b[parent.node_id]
            self.expert_u[child_id] = self.expert_u[parent.node_id]
            self.expert_v[child_id] = self.expert_v[parent.node_id]
            self.expert_b[child_id] = self.expert_b[parent.node_id]
        parent.left, parent.right = child_ids
        self.topology_version += 1
        self.assert_invariants()
        return parent.node_id, child_ids[0], child_ids[1]

    def merge_internal(self, node_id: int) -> tuple[int, int, int]:
        parent = self._node(node_id)
        if parent.is_leaf or parent.left is None or parent.right is None:
            raise ValueError("only an internal node can be merged")
        left = self._node(parent.left)
        right = self._node(parent.right)
        if not left.is_leaf or not right.is_leaf:
            raise ValueError("reference merge only accepts leaf children")
        left_weight = max(float(left.usage), 0.0)
        right_weight = max(float(right.usage), 0.0)
        total = left_weight + right_weight
        alpha = 0.5 if total == 0 else left_weight / total
        self.route_w[parent.node_id] = alpha * self.route_w[left.node_id] + (1.0 - alpha) * self.route_w[right.node_id]
        self.route_b[parent.node_id] = alpha * self.route_b[left.node_id] + (1.0 - alpha) * self.route_b[right.node_id]
        self.expert_u[parent.node_id] = alpha * self.expert_u[left.node_id] + (1.0 - alpha) * self.expert_u[right.node_id]
        self.expert_v[parent.node_id] = alpha * self.expert_v[left.node_id] + (1.0 - alpha) * self.expert_v[right.node_id]
        self.expert_b[parent.node_id] = alpha * self.expert_b[left.node_id] + (1.0 - alpha) * self.expert_b[right.node_id]
        parent.left = None
        parent.right = None
        parent.usage = alpha * left.usage + (1.0 - alpha) * right.usage
        for child_id in (left.node_id, right.node_id):
            self.nodes[child_id] = None
            self._free_ids.add(child_id)
        self.topology_version += 1
        self.assert_invariants()
        return parent.node_id, left.node_id, right.node_id

    def apply_structural_action(self, action: str, *, target_node_id: int | None = None) -> dict[str, Any]:
        """Apply one explicit structural action; no implicit target guessing."""

        if action == "hold_capacity" or action.startswith("hold_and_") or action == "measure_speed_without_ablation":
            return {"action": action, "status": "held", "mutated": False, "topology_version": self.topology_version}
        if target_node_id is None:
            return {"action": action, "status": "abstain", "reason": "target_node_id_required", "mutated": False}
        if action == "wake_target_unit":
            parent, left, right = self.split_leaf(target_node_id)
            return {"action": action, "status": "applied", "mutated": True, "parent": parent, "children": [left, right], "topology_version": self.topology_version}
        if action == "merge_then_ablate_low_contribution_units":
            parent, left, right = self.merge_internal(target_node_id)
            return {"action": action, "status": "applied", "mutated": True, "parent": parent, "reclaimed": [left, right], "topology_version": self.topology_version}
        raise ValueError(f"unknown structural action: {action}")

    def forward(self, contexts: Sequence[Sequence[float]] | np.ndarray, page_mass: Sequence[Sequence[float]] | np.ndarray) -> ForwardOutput:
        context_array = np.asarray(contexts, dtype=np.float64)
        mass_array = np.asarray(page_mass, dtype=np.float64)
        if context_array.ndim != 2 or context_array.shape[1] != self.config.d_model:
            raise ValueError("contexts must have shape [samples, d_model]")
        if mass_array.shape != (context_array.shape[0], self.config.max_pages):
            raise ValueError("page_mass must have shape [samples, max_pages]")
        if not np.all(np.isfinite(context_array)) or not np.all(np.isfinite(mass_array)) or np.any(mass_array < 0):
            raise ValueError("contexts and page_mass must be finite and non-negative where applicable")
        if not np.allclose(np.sum(mass_array, axis=1), 1.0, atol=1e-12):
            raise ValueError("page_mass must conserve one unit of mass per sample")
        outputs = np.zeros((context_array.shape[0], self.config.d_model), dtype=np.float64)
        leaf_mass = np.zeros((context_array.shape[0], self.config.max_pages), dtype=np.float64)
        selected = np.full((context_array.shape[0], self.config.max_pages), -1, dtype=np.int64)
        for sample_index, context in enumerate(context_array):
            for page in range(self.config.max_pages):
                node_id = self._select_leaf(page, context)
                selected[sample_index, page] = node_id
                weight = mass_array[sample_index, page]
                leaf_mass[sample_index, page] += weight
                node = self._node(node_id)
                local = context @ self.expert_u[node_id] @ self.expert_v[node_id] + self.expert_b[node_id]
                outputs[sample_index] += weight * local
                node.usage = 0.99 * node.usage + 0.01 * float(weight)
        self.assert_invariants()
        return ForwardOutput(outputs, leaf_mass, selected)

    def _select_leaf(self, page: int, context: np.ndarray) -> int:
        node_id = self.roots[page]
        while True:
            node = self._node(node_id)
            if node.is_leaf:
                return node_id
            if node.left is None or node.right is None:
                raise RuntimeError("internal node has incomplete children")
            score = float(np.dot(context, self.route_w[node_id]) + self.route_b[node_id]) / self.config.route_temperature
            node_id = node.left if score < 0.0 else node.right

    def fresh_manifest(self) -> dict[str, Any]:
        body = {
            "schema_version": FRESH_MANIFEST_SCHEMA_VERSION,
            "core_schema_version": SCHEMA_VERSION,
            "config": self.config.__dict__,
            "seed": self.seed,
            "topology_version": self.topology_version,
            "architecture_transfer_mode": "bsp_v3_structure_contract_only",
            "parent_checkpoint_reused": False,
            "weight_load_performed": False,
            "foundation_stage": "mandarin_foundation",
            "security_replay_stage": "post_training_security_trace",
            "dataset_merge": False,
            "state_sha256": self.state_sha256(),
        }
        return {**body, "manifest_sha256": _sha256(body)}

    def state_sha256(self) -> str:
        metadata = [
            None
            if node is None
            else {
                "node_id": node.node_id,
                "page": node.page,
                "parent": node.parent,
                "left": node.left,
                "right": node.right,
                "depth": node.depth,
                "usage": round(float(node.usage), 12),
                "active": node.active,
            }
            for node in self.nodes
        ]
        digest = hashlib.sha256()
        digest.update(_canonical({"topology_version": self.topology_version, "roots": self.roots, "nodes": metadata}).encode("utf-8"))
        for array in (self.route_w, self.route_b, self.expert_u, self.expert_v, self.expert_b):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def parameter_sha256(self) -> str:
        """Hash only learnable arrays so replay can prove no weight update."""

        digest = hashlib.sha256()
        for array in (self.route_w, self.route_b, self.expert_u, self.expert_v, self.expert_b):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def assert_invariants(self) -> None:
        if len(self.roots) != self.config.max_pages:
            raise AssertionError("one root is required for every page")
        if self.active_count + self.free_count != self.config.max_nodes:
            raise AssertionError("fixed BSP capacity invariant failed")
        for page, root_id in enumerate(self.roots):
            root = self._node(root_id)
            if root.page != page or root.parent is not None:
                raise AssertionError("root metadata is inconsistent")
        for node in self.nodes:
            if node is None or not node.active:
                continue
            if node.is_leaf:
                continue
            if node.left is None or node.right is None:
                raise AssertionError("internal node must have two children")
            left = self._node(node.left)
            right = self._node(node.right)
            if left.parent != node.node_id or right.parent != node.node_id:
                raise AssertionError("child parent pointer is inconsistent")
            if left.page != node.page or right.page != node.page:
                raise AssertionError("child page differs from parent page")


def validate_fresh_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on legacy checkpoint/resume fields."""

    forbidden = {"checkpoint_path", "parent_checkpoint_path", "resume_checkpoint", "optimizer_state", "weight_path"}
    if any(key in manifest for key in forbidden):
        raise ValueError("Python BSP v3 reference core refuses legacy checkpoint references")
    if manifest.get("schema_version") != FRESH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported Python BSP v3 manifest schema")
    if manifest.get("parent_checkpoint_reused") is not False or manifest.get("weight_load_performed") is not False:
        raise ValueError("fresh Python BSP v3 manifest cannot load old weights")
    if manifest.get("dataset_merge") is not False:
        raise ValueError("Mandarin foundation and security replay datasets must remain isolated")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected = _sha256(body)
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("fresh Python BSP v3 manifest commitment mismatch")
    return dict(manifest)


__all__ = [
    "BspNode",
    "BspV3Config",
    "BspV3State",
    "ExecutionPlan",
    "ForwardOutput",
    "FRESH_MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "validate_fresh_manifest",
]
