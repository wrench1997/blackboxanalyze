# PG-55 不变性特征漏斗

训练侧行数：`324`；来源数：`5`；族数：`9`。

| stage | count |
|---|---:|
| candidate | 54 |
| observable_safe | 37 |
| quality | 30 |
| source_leakage | 11 |
| seed_stability | 11 |
| label_utility_audit | 9 |
| redundancy_pruned | 7 |

保留特征：`geometry_change_presence_control`, `geometry_true_boolean_delta_ratio_control`, `geometry_nonzero_numeric_count`, `geometry_numeric_count`, `geometry_array_item_count`, `geometry_array_count`, `semantic_shape_changed_from_control`
Codex 审核：`approved_for_downstream_ood_experiment`；审核证据：`bbdf77ace434cebdead9616c1dd25129dab86886cf8efbdae953c6ac84b87a3c`。
PG-42 framed、seed 419 和 template_injection 不进入本漏斗，留作盲测。

训练晋升/长期记忆：`False/False`。
