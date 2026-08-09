#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "research/juice_shop_loop_12_response_intervention_runs_v3.json"
OUT_JSON = ROOT / "research/juice_shop_loop_12_response_intervention_report.json"
OUT_MD = ROOT / "research/juice_shop_loop_12_response_intervention_report.md"


def main() -> None:
    artifact = json.loads(RUNS.read_text(encoding="utf-8"))
    projection = artifact["runs"]["response_projection"]
    ablation = artifact["runs"]["ablation_disabled_projection"]
    projection_path = projection["chosen_get"]["action"]["path"]
    ablation_path = ablation["chosen_get"]["action"]["path"]
    report: dict[str, Any] = {
        "schema_version": "sift-juice-shop-loop-12-response-intervention-report-v1",
        "status": "engineering_intervention_inconclusive_for_episode_success",
        "source": str(RUNS.relative_to(ROOT)),
        "fresh_environments": [
            projection["environment"]["environment_seed"],
            ablation["environment"]["environment_seed"],
        ],
        "comparison": {
            "response_projection_final_get": projection_path,
            "disabled_projection_final_get": ablation_path,
            "response_projection_final_selection_correct": projection_path == "/metrics",
            "disabled_projection_final_selection_correct": ablation_path == "/metrics",
            "episode_success_with_projection": projection["episode_success"],
            "episode_success_ablation": ablation["episode_success"],
            "first_selected_loop12_transition_with_projection": projection["first_success_request"],
            "first_selected_loop12_transition_ablation": ablation["first_success_request"],
        },
        "diagnosis": {
            "primary": "HEAD is not side-effect-free in this target: both runs changed evaluator state before final GET",
            "selection_component": "supported: response projection selected the Prometheus surface while the ablation selected the SPA root",
            "episode_success_component": "not identifiable under this protocol because the probe itself can solve the challenge",
            "protocol_change_required": "use an explicitly verified non-mutating observation channel or score post-response identification separately",
        },
        "acceptance": {
            "neural_gain_claimed": False,
            "episode_success_gate": False,
            "same_component_selection_ablation": True,
            "fresh_environment_confirmation": True,
            "accepted": False,
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(f"""# Loop 12 响应投影干预

状态：**动作选择得到支持，但解题率干预不具可识别性。**

| 条件 | 最终 GET | 最终选择正确 | 6 次内 Loop 12 成功 | 首次目标状态变化 |
|---|---|---:|---:|---:|
| 响应投影 | `{projection_path}` | {'是' if projection_path == '/metrics' else '否'} | {'是' if projection['episode_success'] else '否'} | {projection['first_success_request'] or '—'} |
| 关闭投影 | `{ablation_path}` | {'是' if ablation_path == '/metrics' else '否'} | {'是' if ablation['episode_success'] else '否'} | {ablation['first_success_request'] or '—'} |

两组都在第 5 次 `HEAD /metrics` 时改变了目标状态，因此不能把成功率差异归因于响应投影。可以归因的结果是：投影依据通用 Prometheus 媒体类型选中了 `/metrics`，消融组选中了 SPA 根路径。下一轮必须使用已经验证为非突变的观测通道，或把指标改为“响应后的证据识别”而不是“动作导致的解题”。
""", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
