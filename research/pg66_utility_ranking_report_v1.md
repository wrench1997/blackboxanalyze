# PG-66 utility/ranking 轨迹策略头

每个 candidate 使用连续 pre-oracle utility + pairwise ranking；不使用硬选择标签。

| split | 全体 ranking | 明确优势 ranking | mean regret | MSE |
|---|---:|---:|---:|---:|
| train | 0.725146 | 1.0 | 0.0 | 0.000201 |
| dev | 0.735632 | 1.0 | 0.0 | 0.004891 |
| PG64 holdout | 0.721068 | 1.0 | 0.0 | 0.007039 |
| independent noise | — | — | — | — |

安全门：`passed`；能力门：`passed`；训练/记忆不晋升。
