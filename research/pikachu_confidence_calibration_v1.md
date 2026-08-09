# Pikachu PG-PK-04 置信度校准

冻结联合留出 checkpoint，只在 calibration subset 拟合一个温度，再把未见编码变体和反事实控制留作测试。Rule IR 只有在模型置信度与对应族的 bounded oracle 同时支持时才发射。

温度：`1.250`；class ECE 0.146 → 0.138；Brier 0.143 → 0.130。

| 测试集 | gate | exit | false positive | abstain |
|---|---|---:|---:|---:|
| `encoding_holdout_plus_negative` | `model_only` | 0.11 | 0.89 | 0.00 |
| `encoding_holdout_plus_negative` | `family_oracle` | 0.11 | 0.00 | 0.89 |

family_oracle gate 的 abstain 是有意的：没有族特异 bounded evidence 时，即使分类器猜中 family，也不发射 Rule IR。
该实验没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。

完整 JSON：`research\pikachu_confidence_calibration_v1.json`
