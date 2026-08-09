# PG-58 effect confirmation / family naming 解耦

训练族分类样本：`306`；dev/holdout：`188/120`；设备：`cuda`。
盲测 raw known recall：`0.000`；wrong family：`0`；unknown misname：`0`。
盲测 calibrated known recall：`0.000`；unknown misname：`0`；negative false accept：`0`；abstain：`1.000`。
效果头只负责 confirmed/rejected；族命名头只在效果门通过后运行。所有结果仍在隔离区，未进入长期记忆。
