# PG-119 metadata transition slot 训练

新模型从零初始化，合并 PG-116 identity、PG-118 location delta 与 PG-119 第三编码顺序 metadata delta；PG-119 holdout、PG-117 gamma 和 PG-114 保持盲测。

- 参数/设备：`4996` / `cuda`；feature dim：`48`。
- PG-119 正例召回/decoy 误接受/未知弃权：`1.0` / `0` / `1.0`。
- PG-119 跨 seed 正例召回方差：`0.0`。
- metadata slot 消融后正例召回：`0.0`；预测变化：`True`。
- 开发准确率：`0.998106`；PG-119 逐步宏 F1：`0.96044`。
- 只有所有硬门通过才进入 Codex/人工复核；当前不进入长期记忆。
