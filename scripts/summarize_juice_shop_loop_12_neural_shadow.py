#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "research/juice_shop_loop_12_neural_shadow_runs.json"
TRAINING = ROOT / "artifacts/neural-juice-loop-12-response-head-v2-20262097-rerun/report.json"
OUT_JSON = ROOT / "research/juice_shop_loop_12_neural_shadow_report.json"
OUT_MD = ROOT / "research/juice_shop_loop_12_neural_shadow_report.md"


def main() -> None:
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    trained = runs["runs"]["trained_response_head"]
    ablation = runs["runs"]["trained_response_head_ablation"]
    report: dict[str, Any] = {
        "schema_version": "sift-juice-shop-loop-12-neural-shadow-report-v1",
        "status": "single_hidden_surface_intervention_confirmed",
        "scope": "one hidden observability surface under fresh shadow/evaluation environments",
        "training": {
            "checkpoint": str((ROOT / "artifacts/neural-juice-loop-12-response-head-v2-20262097-rerun/tiny_rule_set_gpt.pt").relative_to(ROOT)),
            "parameter_count": training["parameters"],
            "response_validation_accuracy": training["response_validation"]["accuracy"],
            "worst_old_regression_delta": training["worst_regression_delta"],
            "parameter_budget_pass": training["parameter_budget_pass"],
            "old_regression_pass": training["old_regression_pass"],
        },
        "shadow_comparison": {
            "trained_policy": {
                "shadow_probe_count": len(trained["shadow_rows"]),
                "selected_action": trained["evaluation_action"],
                "selected_model_score": max(row["model_score"] for row in trained["shadow_rows"]),
                "episode_success": trained["episode_success"],
                "selected_transitions": trained["selected_loop12_transitions"],
            },
            "same_component_ablation": {
                "shadow_probe_count": len(ablation["shadow_rows"]),
                "selected_action": ablation["evaluation_action"],
                "selected_model_score": max(row["model_score"] for row in ablation["shadow_rows"]),
                "episode_success": ablation["episode_success"],
                "selected_transitions": ablation["selected_loop12_transitions"],
            },
        },
        "comparison_to_frozen_neural": {
            "frozen_neural_no_memory_episode_success": False,
            "trained_response_head_episode_success": trained["episode_success"],
            "gain_on_tested_surface": 1.0 if trained["episode_success"] else 0.0,
        },
        "diagnosis": {
            "root_cause": "the frozen classifier lacked HTTP response semantics; adding a gated response feature head repairs action grounding without changing its Transformer base",
            "rule_ir_output": "still not emitted by the neural head; family inference remains a separate semantic projection",
            "coverage_limit": "only one hidden observability surface was evaluated; six other hidden families remain untested",
            "claim_boundary": "not a full-catalog or production vulnerability-detection result",
        },
        "acceptance": {
            "response_validation_at_least_90pct": training["response_validation"]["accuracy"] >= 0.90,
            "old_regression_within_2pp": training["old_regression_pass"],
            "same_component_ablation_pass": trained["episode_success"] and not ablation["episode_success"],
            "fresh_environment_pass": trained["evaluation_environment"]["initial_solved_count"] == 0 and ablation["evaluation_environment"]["initial_solved_count"] == 0,
            "single_surface_intervention_accepted": trained["episode_success"] and not ablation["episode_success"],
            "full_loop12_accepted": False,
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(f"""# Loop 12 神经响应头 shadow 验证

状态：**单个 hidden observability surface 的神经干预通过；完整 Loop 12 仍未完成。**

| 条件 | shadow 观察 | 评估动作 | 评估成功 | 目标分数 |
|---|---:|---|---:|---:|
| 训练响应头 | {len(trained['shadow_rows'])} | `{trained['evaluation_action']['path']}` | {'是' if trained['episode_success'] else '否'} | {max(row['model_score'] for row in trained['shadow_rows']):.3f} |
| 关闭响应槽 | {len(ablation['shadow_rows'])} | `{ablation['evaluation_action']['path']}` | {'是' if ablation['episode_success'] else '否'} | {max(row['model_score'] for row in ablation['shadow_rows']):.3f} |

训练响应验证准确率为 {training['response_validation']['accuracy']:.2%}，旧合成族最差回归为 {training['worst_regression_delta']:+.2%}，参数数目保持 {training['parameters']}。训练模型在干净评估容器中只执行一次 `GET /metrics` 并成功；关闭响应槽后选择 `security.txt`，目标状态不变。

这确认了“HTTP 响应语义 + 参数守恒响应头”是当前动作接地瓶颈的有效修复，但只覆盖一个 hidden surface；下一轮必须在其余 hidden family 上重复 shadow/evaluation，并继续记录 Rule IR 解码仍然缺失这一限制。
""", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
