# PG-53 特征漏斗反事实消融

固定 PG-35 训练、PG-36 实现留出，比较全部安全候选与 Codex 审核后的漏斗子集。

| feature set | count | typed recall | false accept | abstain |
|---|---:|---:|---:|---:|
| all safe candidates | 37 | 0.000 | 0 | 1.000 |
| reviewed funnel | 6 | 0.000 | 0 | 1.000 |

这是特征选择诊断，不是能力晋升；两条路径都必须经过独立实现、多种子和负对照门禁。
