# PG-42 independent semantic OOD

PG-39 effect head 只读取 bounded candidate/control coarse delta；语义未知时路由为 `unknown_surface + abstain`。

| split | effect recall | effect FPR | positives |
|---|---:|---:|---:|
| train | 1.00 | 0.00 | 64 |
| dev | 0.40 | 0.00 | 80 |
| implementation_holdout | 0.67 | 0.00 | 144 |
| family_holdout | 0.67 | 0.00 | 36 |
| negative_control | 0.00 | 0.00 | 0 |

安全门禁：`blocked`；claim_allowed=`False`；训练/记忆不晋升。
