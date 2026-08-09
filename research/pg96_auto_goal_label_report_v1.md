# PG-96 自动目标/标签设计

状态：`blocked`；选择的无监督谓词：`any_delta_surface`。

提案只看安全观测差分，typed oracle 仅在提案之后用于盲测；不训练、不改 checkpoint、不写长期记忆。

seed holdout 召回：`1.0`；误报：`0`；未知族严格弃权：`False`。

阻塞项：unknown_family_strict_abstain。
