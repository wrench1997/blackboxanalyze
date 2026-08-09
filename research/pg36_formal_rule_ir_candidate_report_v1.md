# PG-36 formal Rule IR candidate

模型只读取 bounded visible projection；typed oracle 只作为监督标签，未进入输入。

| split | typed recall | effect recall | precision | FPR | abstain |
|---|---:|---:|---:|---:|---:|
| train | 0.50 | 0.50 | 1.00 | 0.00 | 0.94 |
| dev | 0.50 | 0.50 | 1.00 | 0.00 | 0.94 |
| family_holdout | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| ood_source | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| negative_control | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| source_holdout (south) | 0.50 | 0.50 | 1.00 | 0.00 | 0.94 |

状态：`blocked`；claim_allowed=`False`；训练晋升与长期记忆均为 `False`。
