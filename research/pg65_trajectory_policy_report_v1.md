# PG-65 轨迹策略头

输入为 pre-oracle surface + belief + candidate action；独立噪声/布局上冻结模型复放。

| split | accuracy/recall | 阴性误报 | 未知弃权 | multi-step | mean steps |
|---|---:|---:|---|---:|---:|
| train policy | 0.760234 | — | — | — | — |
| dev policy | 0.712644 | — | — | — | — |
| PG64 holdout policy | 0.721068 | — | — | — | — |
| independent noise | 1.0 | 0 | True | 0.666667 | 2.333333 |

安全门：`passed`；能力门：`blocked`；formal capability claim=false；训练/记忆不晋升。
