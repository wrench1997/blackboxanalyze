# PG-40 semantic Rule IR router

semantic probe_ref 不含 family 名称或原始 probe；该轮只宣称 source transfer。

| split | typed recall | effect recall | FPR | abstain |
|---|---:|---:|---:|---:|
| train | 1.00 | 1.00 | 0.00 | 0.80 |
| seed_holdout | 1.00 | 1.00 | 0.00 | 0.80 |
| source_holdout | 1.00 | 1.00 | 0.00 | 0.80 |
| negative_control | 0.00 | 0.00 | 0.00 | 1.00 |

状态：source-transfer diagnostic；不是 family-OOD capability claim；不晋升。
