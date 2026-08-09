# PG-36 active belief diagnostic

source-transfer checkpoint 只对 control 投影评分，typed oracle 仅在 candidate replay 后用于停止。

| policy | typed recall | FPR | median queries | mean queries |
|---|---:|---:|---:|---:|
| fixed all phases | 1.00 | 0.00 | 16.0 | 16.00 |
| active belief | 1.00 | 0.00 | 4.0 | 7.20 |

状态：`diagnostic_only`；不允许训练晋升或长期记忆。
