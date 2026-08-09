# PG-67 独立 Rule IR / typed oracle 噪声

模型只在动作前看 surface + belief；Rule IR 只在动作后的 typed oracle 上绑定。

| 指标 | 值 |
|---|---:|
| effect_recall | 1.0 |
| known_family_recall | 0.844037 |
| unknown_misname_count | 0 |
| negative_false_accept_count | 0 |
| unknown_strict_abstain | True |
| rule_ir_bound_count | 92 |
| rule_ir_abstain_count | 36 |

硬门：`passed`；formal capability claim=false；训练/记忆不晋升。
