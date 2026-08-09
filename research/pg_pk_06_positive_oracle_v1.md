# PG-PK-06 跨应用正向 oracle 复核

本轮使用仓库内短生命周期 fixture；它只把 canary 放入 HTML 属性并进行 HTML 转义，不执行脚本。fixture 不进入训练集。严格 OOD 层保持 abstain，只有显式绑定的、带源码哈希和证据哈希的正向 oracle 才能复核成对样本。

样本：3；严格 OOD abstain：3；模型-only 接受：3；正向 oracle 复核接受样本：2（pair：1）。
长期记忆晋升试探：`quarantine`；原因：insufficient_distinct_datasets, insufficient_target_instances。

| pair | variant | candidate | OOD | model-only | strict action | oracle revalidated |
|---|---|---|---|---|---|---|
| `fixture-pair-01` | `plain` | `xss` | yes | accept | `abstain` | accept |
| `fixture-pair-01` | `url_percent` | `xss` | yes | accept | `abstain` | accept |
| `fixture-pair-02` | `plain` | `xss` | yes | accept | `abstain` | abstain |

正向复核不是把 OOD 阈值调低：它要求 pair 两个编码、同一 surface、模型族一致、fixture 源码哈希一致、每条证据 SHA-256 有效、属性 oracle 成立且没有脚本信号。plain-control 作为反事实负例必须 abstain。

完整 JSON：`research\pg_pk_06_positive_oracle_v1.json`
协议：`research\pg_pk_06_positive_oracle_protocol_v1.json`
