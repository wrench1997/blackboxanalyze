# PG-34 independent source generalization

PG-33 checkpoint 只在独立 Python HTTP fixture 上做盲测；权重没有更新。严格阈值来自 PG-33 dev 的零误报校准，未校准结果仅作诊断。

| mode | typed recall | precision | FPR | abstain precision |
|---|---:|---:|---:|---:|
| uncalibrated | 0.00 | 0.00 | 0.90 | 0.50 |
| calibrated | 0.00 | 1.00 | 0.00 | 0.56 |

能力门：`no_proven_gain`；训练晋升：`False`；记忆晋升：`False`。

source-holdout 同族结果不是能力门的族外证明；它只能说明源迁移诊断，最终仍需独立实现、族外、负对照全部过门。
