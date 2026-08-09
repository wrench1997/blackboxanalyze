# PG-38 candidate-control effect pair

effect head 只看 bounded candidate-control delta；typed oracle 只是标签。

| split | typed recall | effect recall | typed FPR | effect FPR | abstain |
|---|---:|---:|---:|---:|---:|
| train | 0.38 | 1.00 | 0.00 | 0.00 | 0.75 |
| seed_holdout | 0.38 | 1.00 | 0.00 | 0.00 | 0.75 |
| surface_holdout | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| family_holdout | 0.00 | 0.50 | 0.00 | 0.00 | 0.88 |
| ood_source | 0.00 | 0.50 | 0.00 | 0.00 | 0.88 |
| source_holdout | 0.25 | 0.67 | 0.00 | 0.00 | 0.83 |
| negative_control | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

状态：`blocked`；claim_allowed=`False`；不晋升。
