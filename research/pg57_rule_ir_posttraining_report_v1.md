# PG-57 typed-oracle Rule IR 后训练

设备：`cuda`；train/dev/holdout：`322/188/120`。
dev effect/modality：`0.936` / `0.872`。
盲测原始 confirmed recall：`1.000`；dev 安全门阈值：`0.9996`；门控后 recall：`0.000`；unknown confirmed attempts：`0`；negative false accepts：`0`。
结果仍保持隔离，不进入训练集或长期记忆；下一步必须增加独立 Rule IR 族外门和未知族密度/不确定性门。
