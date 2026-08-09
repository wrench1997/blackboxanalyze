# PG-45 leave-one-family-out

训练中移除 injection/operator-context；被移除族必须 unknown + abstain。

| split | known recall | unknown effect recall | FPR | strict abstain |
|---|---:|---:|---:|---:|
| seed_holdout | 1.00 | 0.00 | 0.00 | True |
| retained_known | 1.00 | 0.00 | 0.00 | True |
| held_out_family | 0.00 | 1.00 | 0.00 | True |
| other_unknown | 0.00 | 1.00 | 0.00 | True |
| negative_control | 0.00 | 0.00 | 0.00 | True |

安全 leave-one-out gate：`passed`；formal capability claim=false。
