# PG-59 typed-oracle semantic router

设备：`cuda`；train/dev/holdout：`270/640/1728`；输入维度：`39`。
独立 PG-42 盲测（semantic gate）known recall：`1.000`；wrong family：`0`；unknown misname：`0`；negative false accept：`0`。
未知/负例门控后的弃权率：`0.917`。
该实验只验证“typed oracle 已提供语义后能否路由 Rule IR”，不等同于黑盒探测发现能力；仍不进入长期记忆。
