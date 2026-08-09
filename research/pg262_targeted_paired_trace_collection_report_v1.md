# PG-262 fresh paired local trace collection

records=20; gold=13; hard_negative=4; silver=3
sources={'pg260_pikachu_sql_paired': 8, 'pg260_pikachu_xss_paired': 6, 'pg260_pikachu_boolean_paired': 3, 'pg260_pikachu_widebyte_paired': 3}
八个独立 seed cell 分别映射到明确路由/表面；所有 wire 只在本地运行时 stdout 临时展示，数据集只保留抽象 token、响应投影和证据哈希。
训练晋级、长期记忆和公网能力声明均保持关闭，下一步由 PG-262 capacity training 独立验收。

本轮是针对混淆矩阵的 targeted fresh route schedule，不进入训练直到 PG-263 独立容量判官通过。
