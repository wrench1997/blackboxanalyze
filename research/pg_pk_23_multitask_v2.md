# PG-23 Pikachu 多任务硬负样本训练

模型同时学习漏洞族、表面角色和是否允许发射抽象 Rule IR。输入投影会抹掉原始 marker、路径族名、counterfactual 名称和 oracle 值；训练数据只来自授权 loopback Catalog，离线负样本不代表真实回放。

## 评估

| split | seed | total | positive recall | false accept (negative) | abstain | Rule IR emission |
|---|---:|---:|---:|---:|---:|---:|
| source_holdout | 20260802 | 24 | 0.50 | 0.00 | 0.96 | 0.04 |
| source_holdout | 20260803 | 24 | 0.00 | 0.00 | 1.00 | 0.00 |
| source_holdout | 20260804 | 24 | 0.50 | 0.00 | 0.96 | 0.04 |
| hard_negative_holdout | 20260802 | 56 | 0.00 | 0.00 | 1.00 | 0.00 |
| hard_negative_holdout | 20260803 | 56 | 0.00 | 0.00 | 1.00 | 0.00 |
| hard_negative_holdout | 20260804 | 56 | 0.00 | 0.00 | 1.00 | 0.00 |
| encoding_holdout | 20260802 | 39 | 0.00 | 0.00 | 0.92 | 0.08 |
| encoding_holdout | 20260803 | 39 | 0.00 | 0.00 | 0.92 | 0.08 |
| encoding_holdout | 20260804 | 39 | 0.00 | 0.00 | 0.92 | 0.08 |
| surface_holdout | 20260802 | 69 | 0.00 | 0.00 | 0.99 | 0.01 |
| surface_holdout | 20260803 | 69 | 0.00 | 0.00 | 0.99 | 0.01 |
| surface_holdout | 20260804 | 69 | 0.00 | 0.01 | 0.99 | 0.01 |
| joint_holdout | 20260802 | 21 | 0.00 | 0.00 | 0.95 | 0.05 |
| joint_holdout | 20260803 | 21 | 0.00 | 0.00 | 0.95 | 0.05 |
| joint_holdout | 20260804 | 21 | 0.00 | 0.00 | 0.95 | 0.05 |

## 当前结论

PG-23 的安全目标是先消除硬负样本误报，再逐步提高有 bounded evidence 的接受率。任何 transport failure、无 evidence 或族外表面都只能 abstain；Rule IR 是受限模板，不是自由代码生成。

完整 JSON：`research\pg_pk_23_multitask_v2.json`
