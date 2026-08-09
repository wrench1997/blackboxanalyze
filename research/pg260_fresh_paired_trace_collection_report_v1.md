# PG-260 fresh paired local trace collection

records=32; gold=19; hard_negative=9; silver=4
sources={'pg260_pikachu_sql_paired': 8, 'pg260_pikachu_xss_paired': 8, 'pg260_pikachu_boolean_paired': 8, 'pg260_pikachu_widebyte_paired': 8}
八个独立 seed cell 分别映射到明确路由/表面；所有 wire 只在本地运行时 stdout 临时展示，数据集只保留抽象 token、响应投影和证据哈希。
训练晋级、长期记忆和公网能力声明均保持关闭，下一步由 PG-260 capacity training 独立验收。
