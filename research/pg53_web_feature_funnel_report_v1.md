# PG-53 网页特征漏斗审核

输入是 PG-53 的安全 response projection；原始请求、响应正文、URL token、oracle 值和来源/种子标识不进入模型特征。surface_observation 只作 evaluator 诊断；无字段名的 generic_effect_geometry 才有资格进入漏斗。

| stage | feature count |
|---|---:|
| candidate | 54 |
| observable_safe | 37 |
| quality | 30 |
| source_leakage | 10 |
| seed_stability | 10 |
| label_utility_audit | 8 |
| redundancy_pruned | 6 |

最终保留特征：`geometry_change_presence_control`, `geometry_true_boolean_delta_ratio_control`, `geometry_array_item_count`, `geometry_nonzero_numeric_count`, `geometry_numeric_count`, `geometry_array_count`

Codex 特征审核：`approved_for_downstream_ood_experiment`；审核证据哈希：`f6558ad12ad8aaa7fe79d6da5e27a560dc8436c76579844aa5a0dcc7c9c9b5fa`。

审核通过只允许进入下一轮独立 OOD 实验；不会自动训练晋升或写入长期记忆。
