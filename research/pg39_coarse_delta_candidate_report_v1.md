# PG-39 coarse candidate-control delta

effect head 只看 32 维 bounded shape/status delta；typed oracle 只作标签。

| split | typed recall | effect recall | typed FPR | effect FPR | abstain |
|---|---:|---:|---:|---:|---:|
| train | 0.38 | 1.00 | 0.00 | 0.00 | 0.75 |
| seed_holdout | 0.38 | 1.00 | 0.00 | 0.00 | 0.75 |
| surface_holdout | 0.00 | 1.00 | 0.00 | 0.00 | 0.75 |
| family_holdout | 0.00 | 1.00 | 0.00 | 0.00 | 0.75 |
| ood_source | 0.00 | 1.00 | 0.00 | 0.00 | 0.75 |
| source_holdout | 0.25 | 1.00 | 0.00 | 0.00 | 0.75 |
| negative_control | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

状态：`blocked`；claim_allowed=`False`；不晋升。
