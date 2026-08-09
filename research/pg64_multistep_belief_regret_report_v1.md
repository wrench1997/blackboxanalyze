# PG-64 多步 belief / counterfactual regret

每步遵循 pre-oracle 状态 → 候选效用 → noisy observation → belief 后验 → typed oracle（仅动作后） → 下一动作。

| policy | recall | 阴性误报 | 未知弃权 | mean steps | entropy reduction | mean regret |
|---|---:|---:|---|---:|---:|---:|
| active belief | 1.0 | 0 | True | 2.368056 | 1.098612 | 0.0 |
| fixed order | 1.0 | 0 | True | 3.666667 | 1.098612 | 0.170003 |

硬门：`passed`；formal capability claim=false；训练/记忆不晋升。
