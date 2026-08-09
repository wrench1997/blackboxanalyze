# PG-62 目标区特征漏斗

PG61 模型的输入仅做 pre-oracle 特征审计；置换结果是诊断，不是新训练标签。

| 特征 | 变体数 | 平均布局效用下降 | 跨布局/种子稳定 | 决策 |
|---|---:|---:|---|---|
| surface_class | 3 | 0.0 | False | reject |
| response_shape | 3 | 0.0 | False | reject |
| channel_hint | 3 | 0.633333 | True | retain |
| route_depth | 3 | 0.0 | False | reject |
| parameter_count_bucket | 3 | 0.0 | False | reject |

硬门：`passed`；formal capability claim=false；训练/记忆不晋升。
