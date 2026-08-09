# PG-46 active probe policy

effect-gated belief update；动作顺序 GET.screen → POST.screen → GET.confirm → POST.confirm。

| 指标 | 值 |
|---|---:|
| episode_count | 180 |
| effect_success_rate | 1.0 |
| known_family_recall | 1.0 |
| unknown_strict_abstain | True |
| negative_false_accept_count | 0 |
| median_queries | 3.0 |
| mean_queries | 3.1 |
| mean_query_reduction_rate | 0.225 |
| get_post_covered | True |
| accepted_trace_episode_count | 180 |

安全门禁：`passed`；formal capability claim=false；训练/记忆不晋升。
