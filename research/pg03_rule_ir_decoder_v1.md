# PG-03 Catalog Rule IR Decoder

模型只接收安全 probe 和受限响应形状，不接收 family/source/evaluator 标签。`exit_found` 是 Rule IR 家族候选与靶场语义一致，不是漏洞 evaluator 确认。

| split | exit found | false positive | abstain | Rule IR emitted |
|---|---:|---:|---:|---:|
| source_split_same_family | 0.20 | 0.00 | 0.80 | 0.20 |
| family_holdout_structural_transfer | 0.00 | 0.00 | 1.00 | 0.00 |
| family_holdout_unseen_surface | 0.00 | 0.00 | 1.00 | 0.00 |

解释：族外结构相同但语义不可区分时，模型的误报会暴露数据/可观测性不足；完全未见的 surface 应 abstain。该模型是小型基线，不等于 GPT/MoE。
运行点：confidence ≥ 0.25 且 top-2 margin ≥ 0.08；novelty gate 独立生效。
边界：本轮是同一 in-repo 本地 ASGI adapter 内的来源/族/表面隔离，不把它宣称成独立第三方靶场泛化。

原始 JSON：`research\pg03_rule_ir_decoder_v1.json`
