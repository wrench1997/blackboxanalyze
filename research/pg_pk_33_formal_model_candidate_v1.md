# PG-33 formal Rule IR candidate

本报告是候选训练与能力门结果，不是已晋升模型。输入仅包含脱敏可见投影；typed oracle、来源和证据哈希只用于验收与评估。

| cell | role | typed recall | precision | FPR | abstain precision |
|---|---|---:|---:|---:|---:|
| pg33-train-s331-v1 | train | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-train-s337-v1 | train | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-train-s347-v1 | train | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-dev-s331-v1 | dev | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-dev-s337-v1 | dev | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-dev-s347-v1 | dev | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-family_holdout-s331-v1 | family_holdout | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-family_holdout-s337-v1 | family_holdout | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-family_holdout-s347-v1 | family_holdout | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-ood_source-s331-v1 | ood_source | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-ood_source-s337-v1 | ood_source | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-ood_source-s347-v1 | ood_source | 0.00 | 1.00 | 0.00 | 0.50 |
| pg33-negative_control-s331-v1 | negative_control | 0.00 | 1.00 | 0.00 | 1.00 |
| pg33-negative_control-s337-v1 | negative_control | 0.00 | 1.00 | 0.00 | 1.00 |
| pg33-negative_control-s347-v1 | negative_control | 0.00 | 1.00 | 0.00 | 1.00 |

能力门状态：`no_proven_gain`。训练授权：`False`；长期记忆：`False`。

如果 family-holdout/OOD 召回没有超过 always-abstain 基线，结果只能说明数据和训练管线可复现，不能说明泛化能力提升。
