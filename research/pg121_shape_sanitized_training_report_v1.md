# PG-121 shape-sanitized Rule IR 训练

PG-120 发现 shape hash shortcut 后，保持 48 维/4996 参数不变，将四个 hash bucket 在训练和评估中固定为零。

- PG-120 正例召回/decoy 误接受/未知弃权：`1.0` / `0` / `1.0`。
- PG-120 跨 seed 方差：`0.0`；旧模型未知弃权：`0.0`。
- 开发准确率：`0.992424`；容量未增加：`True`。
- 当前仍不进入长期记忆。
