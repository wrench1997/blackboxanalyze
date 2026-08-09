# PG-41 safe unknown router

效果确认与族命名分离；未见语义统一 `unknown_surface + abstain`。

| 指标 | 值 |
|---|---:|
| pair_count | 480 |
| known_positive_count | 48 |
| unknown_positive_count | 48 |
| effect_accepted_count | 96 |
| known_family_recall | 1.0 |
| unknown_effect_recall | 1.0 |
| negative_effect_false_accept_count | 0 |
| unknown_misname_count | 0 |

门禁：`passed`；claim_allowed=`True`；训练/长期记忆均不晋升。
