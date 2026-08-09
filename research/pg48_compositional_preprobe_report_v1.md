# PG-48 compositional pre-probe

发送前策略只看 semantic/channel slots、action 和 belief；response projection 只用于发送后 effect 确认。

| 指标 | 值 |
|---|---:|
| effect_success_rate | 1.0 |
| known_family_recall | 1.0 |
| unknown_strict_abstain | True |
| negative_false_accept_count | 0 |
| mean_queries | 3.125 |
| median_queries | 3.0 |
| mean_query_reduction_rate | 0.21875 |
| get_post_covered | True |
| accepted_trace_episode_count | 48 |

安全门禁：`passed`；formal capability claim=false；训练/记忆不晋升。
