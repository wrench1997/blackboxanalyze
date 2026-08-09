#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = ROOT / "research/juice_shop_loop_12_baseline_runs.json"
OUT_JSON = ROOT / "research/juice_shop_loop_12_baseline_report.json"
OUT_MD = ROOT / "research/juice_shop_loop_12_baseline_report.md"


def main() -> None:
    artifact = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    rows = list(artifact["runs"].values())
    neural = [row for row in rows if row["policy"].startswith("frozen_neural")]
    c5 = next(row for row in rows if row["policy"] == "C5_executable_rule")
    random_row = next(row for row in rows if row["policy"] == "random")
    report: dict[str, Any] = {
        "schema_version": "sift-juice-shop-loop-12-baseline-report-v1",
        "status": "baseline_complete_descriptive_only",
        "scope": "four fixed policies, one fresh pinned local environment per policy, six GET actions each",
        "source": str(RUNS_PATH.relative_to(ROOT)),
        "policy_count": len(rows),
        "aggregate": {
            "random_episode_success_rate": mean([int(random_row["episode_success"])]),
            "frozen_neural_episode_success_rate": round(mean(int(row["episode_success"]) for row in neural), 6),
            "frozen_neural_mean_first_success_probe": round(
                mean(row["first_success_probe"] for row in neural if row["first_success_probe"] is not None), 6
            ) if any(row["first_success_probe"] is not None for row in neural) else None,
            "C5_episode_success_rate": int(c5["episode_success"]),
            "C5_first_success_probe": c5["first_success_probe"],
            "random_vs_frozen_neural_success_delta": round(
                int(random_row["episode_success"]) - mean(int(row["episode_success"]) for row in neural), 6
            ),
        },
        "diagnosis": {
            "environment_or_grader": "pass: every fresh run started with zero selected solved challenges and target stayed internal-only",
            "browser_or_action_grounding": "primary suspected bottleneck: neural policies ranked generic path strings without an observation-conditioned action head",
            "observation_representation": "primary suspected bottleneck: response body/status was not available to the frozen pre-action ranker",
            "episode_memory": "not sufficient by itself: pre-Juice-Shop synthetic URL memory moved the hit to rank 6 but did not provide target semantics",
            "rule_induction": "C5 produced one exact family match on the observed transition; neural paths abstained",
            "neural_capacity_or_objective": "not yet isolated; no retraining is counted in this baseline",
        },
        "intervention_gate": {
            "hidden_gain_over_frozen_neural": "not evaluated: this baseline has one selected hidden transition and C5 is non-neural",
            "old_synthetic_regression": "not evaluated in this artifact",
            "same_component_ablation": False,
            "fresh_environment_confirmation": True,
            "accepted": False,
            "reason": "descriptive baseline only; do not claim an intervention or full-catalog generalization",
        },
        "runs": rows,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table = "\n".join(
        f"| {row['policy']} | {'是' if row['episode_success'] else '否'} | {row['first_success_probe'] or '—'} | "
        f"{'是' if row['counterexample_top1'] else '否'} | {row['negative_control_false_positive_rate']:.1%} | "
        f"{row['rule_abstraction_output_coverage']:.1%} |"
        for row in rows
    )
    markdown = f"""# Juice Shop Loop 12：四组冻结基线

状态：**基线完成；仅作描述性结果，不构成干预接受结论。**

每个策略在独立的新容器中执行固定 6 次安全 GET。评估器只在后台读取本地挑战状态；策略没有看到挑战元数据。

| 策略 | 6 次内成功 | 首次成功探测 | Top-1 命中 | 负对照误报率 | Rule IR 输出覆盖 |
|---|---:|---:|---:|---:|---:|
{table}

## 结论

随机策略第 2 次触发了隐藏测试族 `observability` 的 `exposedMetricsChallenge`。无记忆神经策略 6 次均未触发；它把不存在的控制路径排在第一位。加入旧合成 URL 记忆后，神经策略在第 6 次触发同一挑战，但没有输出漏洞族或 Rule IR。独立 C5 规则在第 1 次触发，并给出正确的 `observability` 抽象。

因此当前最强证据指向动作接地/观察表示层：冻结模型接收的是合成 URL 规则提示，不是“请求—响应—下一动作”的交互表示。长期记忆单独不能修复这个接口错位。环境隔离、重置和证据链通过了本轮工程检查。

本报告没有宣称完整 24 题泛化，也没有把 C5 结果算作神经学习增益。下一步只能在保持参数预算和旧合成回归门槛的前提下，增加一个目标无关的 HTTP 响应语义投影与动作选择头，然后做同组件消融和新环境复验。
"""
    OUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"], "out": str(OUT_MD.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
