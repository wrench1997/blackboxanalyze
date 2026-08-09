"""Run PG-34-E01 on the real PG-33 replay trace.

E01 measures a probe-order/stop controller upper bound.  It never sends a new
request and never trains a model: the source is the already captured loopback
HTTP trace.  The controller may observe a typed oracle *after* a probe only to
decide whether the episode can stop; the visible action ranking itself sees no
family, oracle, evidence or raw-body fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.active_probe import choose_active_probe  # noqa: E402
from app.belief_state import DECODER_FAMILIES, MultiStepBelief  # noqa: E402
from app.cross_lab_safe_catalog import sha256_json  # noqa: E402


PROTOCOL_ID = "sift-pg34-e01-probe-curriculum-v1"
CATALOG_PATH = ROOT / "research" / "pg_pk_33_get_post_typed_replay_catalog_v1.json"
REPORT_PATH = ROOT / "research" / "pg34_e01_probe_curriculum_report_v1.json"
MARKDOWN_PATH = ROOT / "research" / "pg34_e01_probe_curriculum_report_v1.md"


def _uniform_likelihood() -> dict[str, float]:
    return {family: 1.0 / len(DECODER_FAMILIES) for family in DECODER_FAMILIES}


def _visible_action(row: dict[str, Any], *, curriculum_prior: float) -> dict[str, Any]:
    """Return only action/shape fields available before typed oracle feedback."""

    manifest = row["payload_manifest"]
    response = row["response_projection"]
    shape = response.get("shape") or {}
    return {
        "method": manifest["method"],
        "placement": manifest["placement"],
        "route_shape": {"segment_count": 3, "query": manifest["method"] == "GET"},
        "probe_kind": manifest["probe_kind"],
        "encoding_depth": manifest["encoding_depth"],
        "response_shape": {
            "status_class": response["status_class"],
            "content_type_class": response["content_type_class"],
            "kind": shape.get("kind", "other"),
            "key_count_bucket": int(shape.get("key_count", 0)) // 4,
            "array_count": int(shape.get("array_count", 0)),
        },
        "model_score": float(curriculum_prior),
    }


def _method_rows(rows: list[dict[str, Any]], method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = [row for row in rows if row["method"] == method and row["sample_id"].endswith("-control")]
    candidates = [row for row in rows if row["method"] == method and row["sample_id"].endswith("-candidate")]
    if len(controls) != 1 or len(candidates) != 1:
        raise ValueError(f"PG-33 replay group must contain one {method} control/candidate pair")
    return controls[0], candidates[0]


def _fixed_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = []
    for method in ("GET", "POST"):
        control, candidate = _method_rows(rows, method)
        ordered.extend([control, candidate])
    positive = any(bool(row["oracle_projection"]["positive"]) for row in ordered)
    accepted = positive and all(
        bool((row.get("negative_control") or {}).get("same_source", False))
        for row in ordered
        if row["oracle_projection"]["positive"]
    )
    return {
        "query_count": len(ordered),
        "accepted_positive": bool(accepted),
        "false_positive": bool(not positive and accepted),
        "abstained": not bool(accepted),
        "method_order": ["GET", "POST"],
        "evidence_hashes": [row["evidence"]["evidence_hash"] for row in ordered],
    }


def _active_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    belief = MultiStepBelief()
    remaining = {"GET", "POST"}
    selected_methods: list[str] = []
    steps: list[dict[str, Any]] = []
    while remaining:
        candidates = []
        for method in sorted(remaining):
            control, _ = _method_rows(rows, method)
            # GET receives a preregistered curriculum prior because it is a
            # bounded, low-cost screen.  It is not a vulnerability/family
            # label and cannot authorize a positive.
            prior = 0.70 if method == "GET" else 0.50
            visible = _visible_action(control, curriculum_prior=prior)
            visible["rule_ir_decoder"] = {"probabilities": _uniform_likelihood(), "confidence": 0.20}
            visible["method"] = method
            candidates.append(visible)
        chosen = choose_active_probe(candidates)
        method = str(chosen["method"])
        selected_methods.append(method)
        control, candidate = _method_rows(rows, method)
        for stage, row in (("control", control), ("candidate", candidate)):
            visible = _visible_action(row, curriculum_prior=float(chosen["model_score"]))
            belief_step = belief.observe(
                f"{method}:{stage}",
                _uniform_likelihood(),
                evidence_hash=row["evidence"]["evidence_hash"],
            )
            steps.append({
                "stage": stage,
                "method": method,
                "sample_id": row["sample_id"],
                "visible_action": visible,
                "belief_step": belief_step,
                "typed_oracle_available_after_step": True,
                "evidence_sha256": row["evidence"]["evidence_hash"],
            })
        positive = bool(candidate["oracle_projection"]["positive"])
        pair_valid = bool((candidate.get("negative_control") or {}).get("same_source", False))
        if positive and pair_valid:
            return {
                "query_count": len(steps),
                "accepted_positive": True,
                "false_positive": False,
                "abstained": False,
                "method_order": selected_methods,
                "stop_reason": "typed_positive_with_matched_negative_control",
                "steps": steps,
                "belief": belief.snapshot(),
            }
        remaining.remove(method)
    return {
        "query_count": len(steps),
        "accepted_positive": False,
        "false_positive": False,
        "abstained": True,
        "method_order": selected_methods,
        "stop_reason": "no_typed_positive_after_get_post_budget",
        "steps": steps,
        "belief": belief.snapshot(),
    }


def _aggregate(results: list[dict[str, Any]], positive_episode_count: int) -> dict[str, Any]:
    accepted = sum(int(item["accepted_positive"]) for item in results)
    false_positive = sum(int(item["false_positive"]) for item in results)
    return {
        "episode_count": len(results),
        "positive_episode_count": positive_episode_count,
        "accepted_positive_count": accepted,
        "typed_recall": round(accepted / max(positive_episode_count, 1), 6),
        "exit_found_rate": round(accepted / max(len(results), 1), 6),
        "false_positive_count": false_positive,
        "false_positive_rate": round(false_positive / max(len(results) - positive_episode_count, 1), 6),
        "abstain_count": sum(int(item["abstained"]) for item in results),
        "median_queries": float(median(item["query_count"] for item in results)) if results else 0.0,
        "mean_queries": round(sum(item["query_count"] for item in results) / max(len(results), 1), 6),
    }


def _markdown(report: dict[str, Any]) -> str:
    fixed = report["fixed_policy"]
    active = report["active_stop_policy"]
    return "\n".join([
        "# PG-34-E01 探针课程与停止策略",
        "",
        "这是基于 PG-33 真实 loopback HTTP trace 的 controller 上界实验，不是训练结果。模型可见字段不含 family、oracle、证据哈希或原始响应；typed oracle 只在动作完成后用于确认停止。",
        "",
        "| policy | typed recall | exit found | FPR | median queries | mean queries |",
        "|---|---:|---:|---:|---:|---:|",
        f"| fixed GET+POST | {fixed['typed_recall']:.2f} | {fixed['exit_found_rate']:.2f} | {fixed['false_positive_rate']:.2f} | {fixed['median_queries']:.1f} | {fixed['mean_queries']:.2f} |",
        f"| active stop | {active['typed_recall']:.2f} | {active['exit_found_rate']:.2f} | {active['false_positive_rate']:.2f} | {active['median_queries']:.1f} | {active['mean_queries']:.2f} |",
        "",
        f"查询平均减少：{report['query_reduction']['mean']:.2f}；结果不能授权训练或长期记忆：`{report['promotion']['training_allowed']}` / `{report['promotion']['memory_promotion_allowed']}`。",
        "",
        "下一步：把 controller 的停止标签改成延迟反馈训练目标，并在独立实现上复测；不能把这个 oracle 上界当成模型泛化证明。",
    ]) + "\n"


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = list(catalog["samples"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant_id"]), int(row["sampling_seed"]))].append(row)
    fixed_results: list[dict[str, Any]] = []
    active_results: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for (variant_id, seed), episode_rows in sorted(grouped.items()):
        fixed = _fixed_episode(episode_rows)
        active = _active_episode(episode_rows)
        fixed_results.append(fixed)
        active_results.append(active)
        episodes.append({"variant_id": variant_id, "sampling_seed": seed, "fixed": fixed, "active": active})
    positive_episode_count = sum(
        int(any(bool(row["oracle_projection"]["positive"]) for row in episode_rows))
        for episode_rows in grouped.values()
    )
    fixed_metrics = _aggregate(fixed_results, positive_episode_count)
    active_metrics = _aggregate(active_results, positive_episode_count)
    report = {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "pg-pk-34-e01-probe-curriculum-report-v1",
        "source": {
            "catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
            "catalog_manifest_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
            "runtime_replay": True,
            "independent_target_implementation": False,
            "raw_probe_strings_stored": False,
            "raw_response_bodies_stored": False,
        },
        "controller": {
            "class": "oracle_verified_active_stop_controller",
            "visible_fields": ["method", "placement", "route_shape", "probe_kind", "encoding_depth", "response_shape"],
            "oracle_visible_before_probe": False,
            "typed_oracle_used_after_probe_for_stop_only": True,
            "rule_ir_positive_authority": False,
            "belief_update": "uniform_visible_likelihood_only",
        },
        "fixed_policy": fixed_metrics,
        "active_stop_policy": active_metrics,
        "query_reduction": {
            "mean": round(fixed_metrics["mean_queries"] - active_metrics["mean_queries"], 6),
            "median": round(fixed_metrics["median_queries"] - active_metrics["median_queries"], 6),
        },
        "episodes": episodes,
        "promotion": {
            "status": "diagnostic_only",
            "training_allowed": False,
            "memory_promotion_allowed": False,
            "reason": "controller_upper_bound_is_not_model_capability_and_independent_target_is_missing",
        },
        "manifest_sha256": sha256_json({"protocol_id": PROTOCOL_ID, "episodes": episodes}),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_PATH.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "fixed_policy": fixed_metrics,
        "active_stop_policy": active_metrics,
        "query_reduction": report["query_reduction"],
        "report": str(REPORT_PATH.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
