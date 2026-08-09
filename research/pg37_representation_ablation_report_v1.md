# PG-37 representation ablation

typed oracle 是标签而不是输入；surface_variant 也不进入输入。

| ablation | surface-pair agreement | family holdout recall | source holdout recall | unknown FPR |
|---|---:|---:|---:|---:|
| surface_only | 0.62 | 0.00 | 0.25 | 0.00 |
| counterfactual_paired | 0.62 | 0.00 | 0.25 | 0.00 |
| phase_only | 1.00 | 0.00 | 0.00 | 0.00 |

状态：`diagnostic_only`；所有变体 training_allowed 和 memory_promotion_allowed 均为 `False`。
