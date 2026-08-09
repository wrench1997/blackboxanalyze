# PG-34 surface discriminator

该 head 只做 family-specific oracle 路由，不拥有 positive authority。输入为脱敏可见投影；未见族和普通响应必须通过 confidence/novelty abstain。

| split | route accuracy | accepted accuracy | false route | abstain |
|---|---:|---:|---:|---:|
| train | 1.00 | 1.0 | 0.00 | 0.00 |
| dev | 0.00 | — | 0.00 | 1.00 |
| family_holdout | 0.00 | — | 0.00 | 1.00 |
| ood_source | 0.00 | — | 0.00 | 1.00 |
| negative_control | 0.00 | — | 0.00 | 1.00 |

状态：`diagnostic_only`；positive authority：`False`；长期记忆：`False`。

族外路由失败只会触发 abstain 和指定 oracle 探测，不会直接生成漏洞结论。
