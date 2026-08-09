# PG-55 不变性 Rule IR 候选

设备：`cuda`；训练/开发/盲测：`324/108/120`。
审核后的特征：`geometry_change_presence_control`, `geometry_true_boolean_delta_ratio_control`, `geometry_nonzero_numeric_count`, `geometry_numeric_count`, `geometry_array_item_count`, `geometry_array_count`, `semantic_shape_changed_from_control`
盲测 density gate 后 known recall：`0.000`；unknown misname：`0`；negative false accept：`0`；abstain：`1.000`。

该候选仍不训练晋升或写入长期记忆；密度门的安全 abstain 不等于泛化能力证明。
