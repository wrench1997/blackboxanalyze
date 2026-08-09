# PG-35 pair Rule IR candidate

模型只读取 bounded visible projection；identity/url_percent 只通过 pair consistency 约束对齐，不读取 typed oracle 或 family 标签。

| split | recall | precision | FPR | abstain | pair agreement |
|---|---:|---:|---:|---:|---:|
| train | 0.40 | 1.00 | 0.00 | 0.85 | 1.0 |
| dev | 0.00 | 0.00 | 0.00 | 1.00 | 1.0 |
| family_holdout | 0.00 | 0.00 | 0.50 | 0.50 | 1.0 |
| ood_source | 0.00 | 0.00 | 0.50 | 0.50 | 1.0 |
| negative_control | 0.00 | 0.00 | 0.00 | 1.00 | 1.0 |

状态：`no_proven_gain`；训练晋升：`False`；长期记忆：`False`。

族外和源外结果即使 abstain 正确，也不等于找到了漏洞；只有 typed recall 超过冻结基线且零误报才可晋升。
