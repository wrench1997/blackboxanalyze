# PG-35 source-transfer diagnostic

训练使用 alpha 的全部安全 family，beta/gamma 只作源外盲测。

| split | recall | precision | FPR | abstain | pair agreement |
|---|---:|---:|---:|---:|---:|
| train_alpha | 1.00 | 1.00 | 0.00 | 0.56 | 1.00 |
| beta_source_holdout | 1.00 | 1.00 | 0.00 | 0.56 | 1.00 |
| gamma_source_holdout | 1.00 | 1.00 | 0.00 | 0.56 | 1.00 |
| negative_control | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

状态：`diagnostic_only`；不允许训练晋升或长期记忆。
