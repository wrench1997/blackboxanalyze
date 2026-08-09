"""PG-63 independent target-zone implementation replay.

The PG-61 action-value checkpoint is frozen.  This runner constructs a new
fixture without importing PG-61's task generator: layouts, raw surface names,
raw transport names, seeds, and route labels all differ.  Two projections are
compared:

* canonicalized: the independent adapter maps raw observations to the shared
  abstract Rule-IR surface vocabulary;
* raw-shift: unknown raw transport names are fail-closed to ``neutral_lane``.

The second mode is expected to lose recall but must not create false accepts.
This makes a successful canonical replay evidence for the abstraction layer,
not evidence that raw strings transfer between implementations.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PG61_SCRIPT = ROOT / "scripts" / "run_pg61_target_zone_counterfactual.py"
PG61_CHECKPOINT = ROOT / "artifacts" / "pg61-target-zone-counterfactual" / "action_value.pt"
REPORT_PATH = ROOT / "research" / "pg63_independent_target_zone_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg63_independent_target_zone_protocol_v1.json"
TRACE_PATH = ROOT / "research" / "pg63_independent_target_zone_trace_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg63_independent_target_zone_report_v1.md"
SEED = 20630803
ACTION_ORDER = (("GET", "confirm"), ("POST", "confirm"))
RAW_CHANNELS = ("read_path", "write_path", "silent_path")
RAW_SURFACES = ("grid_panel", "detail_panel", "finder_panel")
RAW_SHAPES = ("flat", "map", "jump")
CANONICAL_CHANNEL = {"read_path": "query_lane", "write_path": "form_lane", "silent_path": "neutral_lane"}
CANONICAL_SURFACE = {"grid_panel": "collection", "detail_panel": "record", "finder_panel": "search"}
CANONICAL_SHAPE = {"flat": "compact", "map": "object", "jump": "redirect"}


def _load_pg61() -> Any:
    spec = importlib.util.spec_from_file_location("pg61_for_pg63", PG61_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG61 model contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _surface(raw_channel: str, raw_surface: str, raw_shape: str, index: int, seed: int, canonical: bool) -> dict[str, Any]:
    if canonical:
        channel = CANONICAL_CHANNEL[raw_channel]
        surface_class = CANONICAL_SURFACE[raw_surface]
        response_shape = CANONICAL_SHAPE[raw_shape]
    else:
        # A raw-shift adapter has no trusted vocabulary mapping.  It fails
        # closed instead of guessing that a new raw token means GET or POST.
        channel = "neutral_lane"
        surface_class = "search"
        response_shape = "compact"
    return {
        "base_state_key": f"pg63-base-{index % 7}-{raw_surface}-{raw_shape}",
        "surface_class": surface_class,
        "response_shape": response_shape,
        "channel_hint": channel,
        "route_depth": 1 + ((index + 3 * seed) % 3),
        "parameter_count_bucket": (2 * index + seed) % 3,
        "layout_role": "independent_fixture_surface",
        "pre_oracle": True,
        "raw_surface_class": raw_surface,
        "raw_response_shape": raw_shape,
        "raw_channel": raw_channel,
        "canonicalized": canonical,
    }


def _tasks(canonical: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    serial = 0
    for layout_index, layout in enumerate(("helios", "iris")):
        for local_index in range(96):
            seed = SEED + serial * 19 + layout_index * 211
            rng = random.Random(seed)
            raw_channel = "silent_path" if local_index % 4 == 0 else RAW_CHANNELS[(local_index + layout_index) % 2]
            negative = raw_channel == "silent_path"
            expected = "NONE" if negative else ("GET" if raw_channel == "read_path" else "POST")
            unknown = local_index % 6 == 0 and not negative
            raw_surface = RAW_SURFACES[(local_index + seed) % len(RAW_SURFACES)]
            raw_shape = RAW_SHAPES[(2 * local_index + seed) % len(RAW_SHAPES)]
            task_id = f"pg63-independent-{layout}-{serial:04d}"
            task = {
                "task_id": task_id,
                "dataset_role": "holdout",
                "source_id": f"pg63-independent-{layout}",
                "implementation": "independent_surface_adapter_v1",
                "layout": layout,
                "sampling_seed": seed,
                "surface_projection": _surface(raw_channel, raw_surface, raw_shape, local_index, seed, canonical),
                "expected_method": expected,
                "zone_type": ("unseen_query_zone" if expected == "GET" else "unseen_form_zone") if unknown else ("known_query_zone" if expected == "GET" else "known_form_zone"),
                "unknown_zone": unknown,
                "negative_control": negative,
            }
            task["candidate_order"] = [list(action) for action in ACTION_ORDER]
            rng.shuffle(task["candidate_order"])
            tasks.append(task)
            serial += 1
    return tasks


def _load_model(pg61: Any, device: torch.device) -> Any:
    tasks = _tasks(True)
    model = pg61.ActionValueModel(len(pg61._features(tasks[0], ACTION_ORDER[0]))).to(device)
    checkpoint = torch.load(PG61_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> int:
    pg61 = _load_pg61()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(pg61, device)
    canonical_tasks = _tasks(True)
    raw_shift_tasks = _tasks(False)
    threshold = 0.5
    canonical_metrics, canonical_steps = pg61._evaluate(model, canonical_tasks, device, threshold)
    raw_metrics, raw_steps = pg61._evaluate(model, raw_shift_tasks, device, threshold)
    canonical_metrics = {key: canonical_metrics[key] for key in ("task_count", "positive_task_count", "negative_control_count", "positive_action_accuracy", "target_success_rate", "known_target_success_rate", "unknown_positive_count", "unknown_strict_abstain", "unknown_abstain_count", "negative_false_accept_count", "negative_false_accept_rate", "selected_action_counts", "selected_action_entropy", "fresh_reset_count", "evidence_hash_count", "raw_probe_stored_count", "raw_response_stored_count")}
    raw_metrics = {key: raw_metrics[key] for key in ("task_count", "positive_task_count", "negative_control_count", "positive_action_accuracy", "target_success_rate", "known_target_success_rate", "unknown_positive_count", "unknown_strict_abstain", "unknown_abstain_count", "negative_false_accept_count", "negative_false_accept_rate", "selected_action_counts", "selected_action_entropy", "fresh_reset_count", "evidence_hash_count", "raw_probe_stored_count", "raw_response_stored_count")}
    canonical_gate = canonical_metrics["target_success_rate"] >= 0.90 and canonical_metrics["negative_false_accept_count"] == 0 and canonical_metrics["unknown_strict_abstain"] and canonical_metrics["selected_action_entropy"] >= 0.5
    raw_fail_closed = raw_metrics["negative_false_accept_count"] == 0 and raw_metrics["raw_probe_stored_count"] == 0 and raw_metrics["raw_response_stored_count"] == 0
    gate = {
        "schema_version": "pg63-independent-target-zone-hard-gate-v1",
        "status": "passed" if canonical_gate and raw_fail_closed else "blocked",
        "claim_allowed": False,
        "reasons": [] if canonical_gate and raw_fail_closed else ["independent_canonical_or_fail_closed_gate_failed"],
        "training_allowed": False,
        "memory_promotion_allowed": False,
    }
    trace = {
        "schema_version": "pg63-independent-target-zone-trace-v1",
        "evaluation_only": True,
        "training_eligible": False,
        "model_retrained_on_pg63": False,
        "raw_probe_strings_stored": False,
        "raw_response_bodies_stored": False,
        "canonicalized": {"steps": canonical_steps, "step_count": len(canonical_steps), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for step in canonical_steps])},
        "raw_shift_fail_closed": {"steps": raw_steps, "step_count": len(raw_steps), "trace_manifest_sha256": _sha256_json([step["evidence_hash"] for step in raw_steps])},
    }
    report = {
        "protocol_id": "pg-pk-63-independent-target-zone-v1",
        "schema_version": "pg63-independent-target-zone-report-v1",
        "status": "diagnostic_only",
        "source": {"implementation": "independent_surface_adapter_v1", "layouts": ["helios", "iris"], "task_count": len(canonical_tasks), "seed_formula": "base + serial*19 + layout_offset", "pg61_model_retrained": False, "pg61_task_generator_imported": False},
        "input_contract": {"pre_oracle_surface_only": True, "typed_oracle_after_action_only": True, "family_not_in_features": True, "expected_method_not_in_features": True, "raw_probe_response_not_stored": True, "canonical_mapping": {"raw_channel": CANONICAL_CHANNEL, "raw_surface": CANONICAL_SURFACE, "raw_shape": CANONICAL_SHAPE}},
        "canonicalized_holdout": canonical_metrics,
        "raw_shift_fail_closed": raw_metrics,
        "hard_gate": gate,
        "promotion": {"status": "quarantined_independent_replay", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False},
        "interpretation": "只有独立实现经过 canonical Rule-IR 表面映射后才允许比较动作能力；未映射原始表面必须 fail-closed，不能把原始字符串相似性当作泛化。",
    }
    protocol = {
        "protocol_id": "pg-pk-63-independent-target-zone-v1",
        "schema_version": "pg63-independent-target-zone-protocol-v1",
        "objective": "冻结 PG61 模型，在独立实现、布局、原始表面编码和通道命名上做族外复放。",
        "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "state_mutation": False, "raw_probe_persistence": False, "raw_response_body_persistence": False},
        "independence_contract": {"new_implementation": True, "new_layouts": ["helios", "iris"], "new_raw_surface_vocabulary": list(RAW_SURFACES), "new_raw_channel_vocabulary": list(RAW_CHANNELS), "pg61_model_retraining_forbidden": True, "pg61_task_generator_import_forbidden": True},
        "comparison": {"canonicalized_projection_required": True, "raw_shift_must_fail_closed": True, "typed_oracle_after_action_only": True, "fresh_reset_per_action": True, "matched_negative_controls": True, "evidence_hash_per_action": True, "unknown_zone_must_not_bind_known_family": True},
        "run_result": {"status": gate["status"], "canonicalized_target_success_rate": canonical_metrics["target_success_rate"], "raw_shift_negative_false_accept_count": raw_metrics["negative_false_accept_count"]},
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TRACE_PATH.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PG-63 独立目标区复放", "", "PG61 模型冻结；PG63 不重新训练、不导入 PG61 任务生成器。", "", "| 模式 | 目标成功率 | 动作准确率 | 阴性误报 | 未知弃权 | 动作熵 |", "|---|---:|---:|---:|---|---:|"]
    for name, metrics in (("canonicalized", canonical_metrics), ("raw-shift fail-closed", raw_metrics)):
        lines.append(f"| {name} | {metrics['target_success_rate']} | {metrics['positive_action_accuracy']} | {metrics['negative_false_accept_count']} | {metrics['unknown_strict_abstain']} | {metrics['selected_action_entropy']} |")
    lines.extend(["", f"硬门：`{gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""])
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "canonicalized_holdout": canonical_metrics, "raw_shift_fail_closed": raw_metrics, "hard_gate": gate, "report": str(REPORT_PATH.relative_to(ROOT)), "trace": str(TRACE_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
