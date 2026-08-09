#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "research/juice_shop_loop_12_final_report.json"
OUT_MD = ROOT / "research/juice_shop_loop_12_final_report.md"


def read(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    baseline = read("research/juice_shop_loop_12_baseline_report.json")
    shadow = read("research/juice_shop_loop_12_shadow_replay_report.json")
    neural = read("research/juice_shop_loop_12_neural_shadow_report.json")
    seeds = read("research/juice_shop_loop_12_response_head_fresh_seeds.json")
    report = {
        "schema_version": "sift-juice-shop-loop-12-final-report-v1",
        "status": "partial_research_complete_high_variance_remains",
        "baseline": baseline["aggregate"],
        "engineering_intervention": shadow["acceptance"],
        "neural_single_surface": neural["acceptance"],
        "neural_fresh_seed": {
            "mean_accuracy": seeds["mean_accuracy"],
            "minimum_accuracy": seeds["minimum_accuracy"],
            "accepted": seeds["accepted"],
        },
        "confirmed_findings": [
            "the local target reset and internal-only network invariants pass",
            "the frozen neural classifier lacks HTTP response semantics and ranks generic SPA fallbacks poorly",
            "a shadow/evaluation split prevents probe-side challenge state from contaminating evaluation",
            "a parameter-preserving 8-slot response head selected the hidden metrics surface and passed same-component ablation on a fresh target",
        ],
        "unresolved_findings": [
            "fresh synthetic response seeds have a minimum of 88%, so robust response generalization is not confirmed",
            "only one hidden observability surface has been evaluated in the neural shadow protocol",
            "the neural head still does not emit canonical Rule IR; family abstraction remains a separate projection",
            "the full 24-task safe catalog and six remaining hidden families are not yet evaluated",
        ],
        "next_target": "register a multi-family shadow matrix with richer compositional response surface generation, preserve the -2pp old-family gate, and add a neural Rule IR decoder only after response-seed stability is restored",
        "claim_boundary": "local research evidence only; no production vulnerability rate and no authorization to test external systems",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(f"""# Juice Shop External-Validity Loop 12：阶段总结

状态：**局部研究完成；响应训练的跨种子稳定性仍不足，完整泛化未确认。**

## 已确认

- 本地 Docker 重置、内部网络和证据链通过。
- 冻结 Loop 11 模型在动作接地上失败；无记忆神经组 6 次零成功，旧合成记忆组只能在第 6 次命中。
- shadow/evaluation 分离修复了 HEAD 副作用污染问题。
- 参数守恒的 8 槽响应头在一个 hidden observability surface 上选择 `/metrics`，同组件消融失败；旧合成族回归为 0.00pp。

## 尚未确认

- 新合成响应种子准确率为 {seeds['mean_accuracy']:.2%}、最低 {seeds['minimum_accuracy']:.2%}，稳定性门槛未通过。
- 神经 shadow 只覆盖一个 hidden surface；其余 hidden family 和完整 24 题未跑。
- 神经头仍不输出 canonical Rule IR，抽象层尚未闭环。

下一目标：扩充组合式响应表面生成，注册多族 shadow 矩阵，在保持旧族 −2pp 门槛下恢复跨种子稳定性，再增加 Rule IR 解码实验。
""", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
