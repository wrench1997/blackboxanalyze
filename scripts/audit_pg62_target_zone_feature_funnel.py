"""PG-62 feature funnel and permutation audit for PG-61.

PG-61 demonstrated a target-zone action policy on a controlled fixture.  PG-62
audits whether that policy depends on stable pre-oracle surface features rather
than layout IDs, labels, or accidental feature combinations.  It does not
retrain or promote the model.  Permutation is a diagnostic counterfactual: a
drop in action utility shows reliance, while the baseline cross-layout/seed
gate remains the safety criterion.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PG61_SCRIPT = ROOT / "scripts" / "run_pg61_target_zone_counterfactual.py"
CATALOG_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_catalog_v1.json"
PG61_REPORT_PATH = ROOT / "research" / "pg61_target_zone_counterfactual_report_v1.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "pg61-target-zone-counterfactual" / "action_value.pt"
REPORT_PATH = ROOT / "research" / "pg62_target_zone_feature_funnel_report_v1.json"
PROTOCOL_PATH = ROOT / "research" / "pg62_target_zone_feature_funnel_protocol_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg62_target_zone_feature_funnel_report_v1.md"
SEED = 20620803
SURFACE_FEATURES = ("surface_class", "response_shape", "channel_hint", "route_depth", "parameter_count_bucket")


def _load_pg61() -> Any:
    spec = importlib.util.spec_from_file_location("pg61_target_zone_fixture", PG61_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load PG61 fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_model(pg61: Any, device: Any) -> Any:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    first = catalog["tasks"][0]
    model = pg61.ActionValueModel(len(pg61._features(first, tuple(pg61.ACTION_ORDER[0])))).to(device)
    checkpoint = __import__("torch").load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _contract_audit() -> dict[str, Any]:
    tree = ast.parse(PG61_SCRIPT.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    feature_fn = functions.get("_features")
    forbidden = {"oracle_projection", "response_projection", "expected_method", "zone_type", "layout", "task_id", "raw_probe", "raw_response"}
    constants = {node.value for node in ast.walk(feature_fn) if isinstance(node, ast.Constant) and isinstance(node.value, str)} if feature_fn else set()
    return {
        "feature_function_present": feature_fn is not None,
        "forbidden_label_or_oracle_constants_present": sorted(constants & forbidden),
        "oracle_is_label_not_feature": not bool(constants & {"oracle_projection", "response_projection"}),
        "target_method_is_label_not_feature": "expected_method" not in constants,
        "layout_id_is_not_feature": "layout" not in constants,
        "raw_request_response_is_not_feature": not bool(constants & {"raw_probe", "raw_response"}),
    }


def _clone_with_feature(task: dict[str, Any], feature: str, value: Any) -> dict[str, Any]:
    clone = copy.deepcopy(task)
    clone["surface_projection"][feature] = value
    return clone


def _mode_value(tasks: list[dict[str, Any]], feature: str) -> Any:
    counts = Counter(task["surface_projection"][feature] for task in tasks)
    return counts.most_common(1)[0][0]


def _feature_values(tasks: list[dict[str, Any]], feature: str) -> list[Any]:
    return [task["surface_projection"][feature] for task in tasks]


def _permute_tasks(tasks: list[dict[str, Any]], feature: str, seed: int, strata: str = "global") -> list[dict[str, Any]]:
    """Shuffle one pre-oracle feature without changing IDs or labels."""

    output = [copy.deepcopy(task) for task in tasks]
    groups: dict[Any, list[int]] = {}
    for index, task in enumerate(output):
        if strata == "layout":
            key = task["layout"]
        elif strata == "seed_bucket":
            # PG-61 increments seeds by 17.  Modulo-4 would resonate with
            # the fixture's two-channel cycle and create single-valued bins;
            # modulo-3 keeps both channel hints in each stability stratum.
            key = int(task["sampling_seed"]) % 3
        else:
            key = "all"
        groups.setdefault(key, []).append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        values = [output[index]["surface_projection"][feature] for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values):
            output[index]["surface_projection"][feature] = value
    return output


def _drop_tasks(tasks: list[dict[str, Any]], feature: str) -> list[dict[str, Any]]:
    replacement = _mode_value(tasks, feature)
    return [_clone_with_feature(task, feature, replacement) for task in tasks]


def _metrics(pg61: Any, model: Any, tasks: list[dict[str, Any]], device: Any, threshold: float) -> dict[str, Any]:
    metrics, _ = pg61._evaluate(model, tasks, device, threshold)
    return {key: metrics[key] for key in ("task_count", "positive_action_accuracy", "target_success_rate", "negative_false_accept_count", "unknown_strict_abstain", "selected_action_entropy", "selected_action_counts")}


def _utility_drop(baseline: dict[str, Any], ablated: dict[str, Any]) -> float:
    return round(float(baseline["target_success_rate"] - ablated["target_success_rate"]), 6)


def main() -> int:
    import torch

    pg61 = _load_pg61()
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    pg61_report = json.loads(PG61_REPORT_PATH.read_text(encoding="utf-8"))
    tasks = catalog["tasks"]
    holdout = [task for task in tasks if task["dataset_role"] == "holdout"]
    threshold = float(pg61_report["dev"]["selected_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(pg61, device)
    baseline = _metrics(pg61, model, holdout, device, threshold)
    by_layout = {layout: _metrics(pg61, model, [task for task in holdout if task["layout"] == layout], device, threshold) for layout in sorted({task["layout"] for task in holdout})}
    by_seed_bucket = {str(bucket): _metrics(pg61, model, [task for task in holdout if int(task["sampling_seed"]) % 3 == bucket], device, threshold) for bucket in range(3)}

    permutation: dict[str, Any] = {}
    funnel_rows: list[dict[str, Any]] = []
    for offset, feature in enumerate(SURFACE_FEATURES):
        global_perm = _metrics(pg61, model, _permute_tasks(holdout, feature, SEED + offset), device, threshold)
        layout_perm = {}
        layout_drops = []
        for layout, layout_tasks in ((name, [task for task in holdout if task["layout"] == name]) for name in sorted({task["layout"] for task in holdout})):
            perm_tasks = _permute_tasks(layout_tasks, feature, SEED + offset + len(layout), strata="layout")
            metrics = _metrics(pg61, model, perm_tasks, device, threshold)
            layout_perm[layout] = metrics
            layout_drops.append(_utility_drop(by_layout[layout], metrics))
        seed_perm = {}
        seed_drops = []
        for bucket in range(3):
            bucket_tasks = [task for task in holdout if int(task["sampling_seed"]) % 3 == bucket]
            metrics = _metrics(pg61, model, _permute_tasks(bucket_tasks, feature, SEED + offset + bucket, strata="seed_bucket"), device, threshold)
            seed_perm[str(bucket)] = metrics
            seed_drops.append(_utility_drop(by_seed_bucket[str(bucket)], metrics))
        dropped = _metrics(pg61, model, _drop_tasks(holdout, feature), device, threshold)
        variation = len(set(_feature_values(tasks, feature)))
        mean_drop = round(sum(layout_drops) / max(len(layout_drops), 1), 6)
        stable_sign = all(drop >= 0.05 for drop in layout_drops) and all(drop >= 0.05 for drop in seed_drops)
        funnel_rows.append({"feature": feature, "observable_safe": True, "train_holdout_variation_cardinality": variation, "non_constant": variation > 1, "global_permutation": global_perm, "drop_to_mode": dropped, "layout_utility_drops": {layout: drop for layout, drop in zip(sorted(by_layout), layout_drops)}, "seed_bucket_utility_drops": {str(bucket): drop for bucket, drop in enumerate(seed_drops)}, "mean_layout_utility_drop": mean_drop, "stable_utility_across_layouts_and_seeds": stable_sign, "funnel_decision": "retain" if variation > 1 and stable_sign else "reject"})
        permutation[feature] = {"global": global_perm, "by_layout": layout_perm, "by_seed_bucket": seed_perm, "drop_to_mode": dropped}

    accepted = [row["feature"] for row in funnel_rows if row["funnel_decision"] == "retain"]
    contract = _contract_audit()
    baseline_gate = baseline["target_success_rate"] >= 0.90 and baseline["negative_false_accept_count"] == 0 and baseline["unknown_strict_abstain"] and baseline["selected_action_entropy"] >= 0.5
    gate = {
        "schema_version": "pg62-target-zone-feature-funnel-hard-gate-v1",
        "status": "passed" if baseline_gate and contract["oracle_is_label_not_feature"] and accepted else "blocked",
        "claim_allowed": False,
        "training_allowed": False,
        "memory_promotion_allowed": False,
        "reasons": [] if baseline_gate and contract["oracle_is_label_not_feature"] and accepted else ["feature_funnel_or_baseline_gate_failed"],
    }
    report = {
        "protocol_id": "pg-pk-62-target-zone-feature-funnel-v1",
        "schema_version": "pg62-target-zone-feature-funnel-report-v1",
        "status": "diagnostic_only",
        "source_report": str(PG61_REPORT_PATH.relative_to(ROOT)),
        "source_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "device": str(device),
        "threshold_from_dev": threshold,
        "input_contract_audit": contract,
        "candidate_features": list(SURFACE_FEATURES),
        "funnel": {"stages": ["observable_safe", "non_constant_across_train_holdout", "cross_layout_permutation_utility", "cross_seed_permutation_utility"], "rows": funnel_rows, "accepted_features": accepted},
        "baseline_holdout": baseline,
        "baseline_by_layout": by_layout,
        "baseline_by_seed_bucket": by_seed_bucket,
        "permutation_ablation": permutation,
        "hard_gate": gate,
        "promotion": {"status": "quarantined_feature_funnel_audit", "training_allowed": False, "memory_promotion_allowed": False, "formal_capability_claim_allowed": False},
        "interpretation": "PG62 审核的是输入特征稳定性，不是漏洞发现能力；即使硬门通过，也不把 PG61 fixture 标签或特征重要性写入长期记忆。",
    }
    protocol = {
        "protocol_id": "pg-pk-62-target-zone-feature-funnel-v1",
        "schema_version": "pg62-target-zone-feature-funnel-protocol-v1",
        "objective": "以特征漏斗和跨布局/种子置换消融，筛掉依赖布局、标签或偶然表面编码的目标区动作特征。",
        "input_contract": {"pre_oracle_only": True, "candidate_surface_features": list(SURFACE_FEATURES), "action_is_decision_input": True, "oracle_is_label_only": True, "family_is_not_feature": True, "raw_probe_response_forbidden": True},
        "required_stages": {"observable_safe": True, "non_constant_across_train_holdout": True, "permutation_by_layout": True, "permutation_by_seed_bucket": True, "seed_bucket_modulus": 3, "baseline_negative_false_accept_zero": True, "baseline_unknown_strict_abstain": True, "baseline_action_entropy_min": 0.5, "codex_review_required": True},
        "authorized_scope": {"target_host": "127.0.0.1", "external_network": False, "fixture_replay_only": True, "raw_probe_persistence": False, "raw_response_body_persistence": False, "state_mutation": False},
        "run_result": {"status": gate["status"], "accepted_features": accepted, "hard_gate": gate["status"]},
    }
    MARKDOWN_PATH.write_text("\n".join(["# PG-62 目标区特征漏斗", "", "PG61 模型的输入仅做 pre-oracle 特征审计；置换结果是诊断，不是新训练标签。", "", "| 特征 | 变体数 | 平均布局效用下降 | 跨布局/种子稳定 | 决策 |", "|---|---:|---:|---|---|"] + [f"| {row['feature']} | {row['train_holdout_variation_cardinality']} | {row['mean_layout_utility_drop']} | {row['stable_utility_across_layouts_and_seeds']} | {row['funnel_decision']} |" for row in funnel_rows] + ["", f"硬门：`{gate['status']}`；formal capability claim=false；训练/记忆不晋升。", ""]), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": report["protocol_id"], "baseline_holdout": baseline, "accepted_features": accepted, "funnel": funnel_rows, "hard_gate": gate, "report": str(REPORT_PATH.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
