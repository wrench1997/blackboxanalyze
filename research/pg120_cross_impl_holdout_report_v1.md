# PG-120 cross-implementation metadata holdout

冻结 PG-119 checkpoint，换用 PG-120 eta 独立实现，三档 matched decoy 强度和三个 seed 做 GET/POST fresh-reset 复放。

- 设备/feature dim：`cuda` / `48`；weights frozen：`True`。
- PG-120 正例召回/decoy 误接受/未知弃权：`1.0` / `0` / `0.0`。
- 跨 seed 正例召回方差：`0.0`；slot 消融后召回：`0.0`。
- 阴性对照强度档位：`[0, 1, 2]`；所有档位门：`True`。
- 本轮 evaluation-only，不增加训练行、不更新权重、不写长期记忆。
