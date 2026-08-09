# PG-50 stability matrix

训练只使用 ember train split 的 semantic/channel/action/belief；frost 与 quartz 是实现外 holdout。

| implementation | effect success | known recall | unknown abstain | negative false accept | mean queries |
|---|---:|---:|---:|---:|---:|
| frost | 1.0 | 1.0 | True | 0 | 3.1 |
| quartz | 1.0 | 1.0 | True | 0 | 3.1 |

稳定性安全门：`passed`；formal capability claim=false；训练/记忆不晋升。
