# PG-101 主动 probe-bank signature

PG42 eval rows: 360；已知族召回: 1.0；未知族严格弃权: True；误报: 0。
PG35 third implementation recall: 1.0；误报: 0。
PG99 静态投影已知/未知重叠: 6；主动签名重叠: 0；顺序不变率: 1.0。

| split | rows | known recall | unknown abstain | false accepts |
|---|---:|---:|---|---:|
| `dev` | 16 | 1.0 | False | 0 |
| `pg42` | 360 | 1.0 | True | 0 |
| `pg35-third` | 162 | 1.0 | False | 0 |

这是 representation/baseline 实验：probe 值只在 loopback 运行时存在，模型输入只有 canonical ID 与无字段名几何差分。即使能力门通过，仍不训练/写长期记忆，下一步是神经 set decoder + 第三实现复放。

JSON: `research\pg101_active_probe_signature_report_v1.json`
协议: `research\pg101_active_probe_signature_protocol_v1.json`
