# PG-54 PG-42 独立实现 Rule IR 族外验证

只复用 PG-53 Codex 审核后的四个匿名 geometry 特征；不训练、不写长期记忆。PG-42 的 template_injection 不在模型类别中，必须安全 abstain。

样本：`360`；权威正例：`324`；阴性：`36`；GET/POST：`{'GET': 180, 'POST': 180}`。

| split | known recall | unknown misname | negative false accept | abstain |
|---|---:|---:|---:|---:|
| all | 0.000 | 0 | 0 | 1.000 |
| implementation_cobalt | 0.000 | 0 | 0 | 1.000 |
| implementation_quartz | 0.000 | 0 | 0 | 1.000 |
| variant_framed | 0.000 | 0 | 0 | 1.000 |
| unknown_template_family | 0.000 | 0 | 0 | 1.000 |
| negative_control | 0.000 | 0 | 0 | 1.000 |

PG-54 特征复审：`approved_for_downstream_ood_experiment`；PG-53 特征迁移门：`blocked`；审核证据哈希：`f6558ad12ad8aaa7fe79d6da5e27a560dc8436c76579844aa5a0dcc7c9c9b5fa`。
密度 abstain 门：`0.0`（只用 PG-53 dev 校准，PG-54 不参与阈值选择）。
未知族严格 abstain：`True`；训练/长期记忆晋升：`False/False`。
