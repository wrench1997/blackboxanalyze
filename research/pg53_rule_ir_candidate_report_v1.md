# PG-53 Rule IR 候选训练与族外实现留出

候选模型只在 PG-35 的可见 response-shape 投影上训练，PG-36 的不同实现/布局作为盲测。typed oracle 只做标签与复核，不进入特征；ordinary_response 被映射为 abstain。

设备：`cuda`；训练准确率：`0.556`。
漏斗审核后特征：`geometry_change_presence_control`, `geometry_true_boolean_delta_ratio_control`, `geometry_array_item_count`, `geometry_nonzero_numeric_count`, `geometry_numeric_count`, `geometry_array_count`。
PG-36 校准后 typed recall：`0.000`；precision：`1.000`；false accept：`0`；abstain：`1.000`。

结果仍是 quarantined candidate：需要更多独立实现、族外数据和多种子能力门后，才能考虑训练晋升或长期记忆。

训练晋升：`False`；长期记忆：`False`；正式能力声明：`False`。
