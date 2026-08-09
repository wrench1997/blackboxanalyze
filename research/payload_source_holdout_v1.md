# PG-02 Source-Grounded Payload Holdout

本实验只读取带授权证明的本地 safe detection manifest。source split 检查换来源迁移，family holdout 检查族外结构迁移与无证据时的 fail-closed abstention。

| split | 策略 | exact success | structural transfer | abstention | unsupported selection | authorization valid |
|---|---|---:|---:|---:|---:|---:|
| family_holdout_structural_transfer | random_authorized | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| family_holdout_structural_transfer | source_grounded | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| family_holdout_unseen_surface | random_authorized | 0.80 | 1.00 | 0.00 | 1.00 | 1.00 |
| family_holdout_unseen_surface | source_grounded | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| source_split_same_family | random_authorized | 0.40 | 0.40 | 0.00 | 0.60 | 1.00 |
| source_split_same_family | source_grounded | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 |

边界：source_grounded 是结构记忆控制器，不是神经模型；族外 transfer 的 exact success 仍来自本地合成 oracle，不代表真实站点漏洞确认。公网语料和 evaluator 状态均未接入。

原始 JSON：`research\payload_source_holdout_v1.json`
