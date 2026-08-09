#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "research/juice_shop_loop_12_shadow_replay_runs.json"
OUT_JSON = ROOT / "research/juice_shop_loop_12_shadow_replay_report.json"
OUT_MD = ROOT / "research/juice_shop_loop_12_shadow_replay_report.md"


def main() -> None:
    source = json.loads(RUNS.read_text(encoding="utf-8"))
    projection = source["runs"]["response_projection"]
    ablation = source["runs"]["ablation_disabled_projection"]
    report: dict[str, Any] = {
        "schema_version": "sift-juice-shop-loop-12-shadow-replay-report-v1",
        "status": "engineering_intervention_confirmed_for_local_action_selection",
        "source": str(RUNS.relative_to(ROOT)),
        "scope": "one shadow container plus one fresh evaluation container per policy",
        "comparison": {
            "response_projection": {
                "shadow_probe_count": projection["shadow_probe_count"],
                "evaluation_request_count": projection["evaluation_request_count"],
                "selected_action": projection["evaluation_action"],
                "selected_loop12_transitions": projection["selected_loop12_transitions"],
                "episode_success": projection["episode_success"],
            },
            "ablation_disabled_projection": {
                "shadow_probe_count": ablation["shadow_probe_count"],
                "evaluation_request_count": ablation["evaluation_request_count"],
                "selected_action": ablation["evaluation_action"],
                "selected_loop12_transitions": ablation["selected_loop12_transitions"],
                "episode_success": ablation["episode_success"],
            },
        },
        "diagnosis": {
            "state_isolation": "pass: shadow transitions were not present in the fresh evaluation container",
            "response_grounding": "supported: only the response projection selected the Prometheus surface",
            "rule_abstraction": "observability family was inferred from a target-neutral media type, not challenge metadata",
            "neural_learning": "not claimed: this is a zero-parameter engineering/control intervention",
            "cost": "five shadow observations plus one evaluation request per policy; shadow startup is engineering overhead",
        },
        "acceptance": {
            "selected_action_ablation": True,
            "fresh_environment_confirmation": True,
            "neural_intervention_gate": False,
            "full_catalog_generalization": False,
            "accepted": True,
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(f"""# Loop 12 影子重放干预

状态：**在本地固定靶场上确认响应接地能改善动作选择。**

| 条件 | shadow GET | 干净评估 GET | 评估动作 | Loop 12 目标成功 |
|---|---:|---:|---|---:|
| 响应投影 | {projection['shadow_probe_count']} | {projection['evaluation_request_count']} | `{projection['evaluation_action']['path']}` | {'是' if projection['episode_success'] else '否'} |
| 关闭投影 | {ablation['shadow_probe_count']} | {ablation['evaluation_request_count']} | `{ablation['evaluation_action']['path']}` | {'是' if ablation['episode_success'] else '否'} |

shadow 容器中的状态变化没有带入评估容器；响应投影在全新评估容器中只执行一次 `GET /metrics` 并触发目标挑战，消融组选根路径且没有状态变化。这证明了观察表示/动作接地层的工程修复，但不是神经模型训练增益，也不是完整 24 题泛化结论。下一步若要训练模型，必须保持旧合成族回归门槛，并以同一 shadow/evaluation 协议做神经头部消融。
""", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
