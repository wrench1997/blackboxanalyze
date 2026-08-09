# PG-43 invariant effect candidate

只用 PG-37 训练；模型输入是 sign-binned shape delta 与 change bits，去除 envelope/body/header/status/method/phase 维度。

| split | effect recall | effect FPR | abstain |
|---|---:|---:|---:|
| pg37_train | 1.00 | 0.00 | 0.75 |
| pg37_seed_holdout | 1.00 | 0.00 | 0.75 |
| pg37_negative_control | 0.00 | 0.00 | 1.00 |
| pg42_train_role_diagnostic | 1.00 | 0.00 | 0.75 |
| pg42_dev_role | 1.00 | 0.00 | 0.75 |
| pg42_implementation_holdout | 1.00 | 0.00 | 0.75 |
| pg42_family_holdout | 1.00 | 0.00 | 0.75 |
| pg42_negative_control | 0.00 | 0.00 | 1.00 |

候选 effect 门禁：`passed`；完整能力 claim 仍关闭。
