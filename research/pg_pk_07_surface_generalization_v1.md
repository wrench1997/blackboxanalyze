# PG-PK-07 多表面反事实与族外泛化

本轮在未见过的本地 fixture 上把同一 inert marker 放到 HTML 属性、HTML 文本、JSON、响应头和空白控制五类表面；只有属性表面属于本实验的正向 oracle。

样本：9；严格 OOD abstain：9；模型-only 接受：9；非属性表面模型-only 误报候选：7；oracle 复核样本：2。
辅助 surface discriminator 在非属性样本上拒绝：7；它只作诊断，不具备正向放行权。
同一 discriminator 对属性正例 abstain：2；因此当前结论是‘能筛掉一部分表面捷径，但尚不能做正向 gate’，需要后续做跨应用校准。
PG-PK-08 surface-role head 属性正例接受：0/2；非属性误接受：0；该 head 仍只作诊断。
surface-role promotion gate：`diagnostic_only`；原因：training_stability_below_gate, positive_recall_below_gate。

| surface role | samples | model-only accepts | oracle signal rows | revalidated pairs |
|---|---:|---:|---:|---:|
| `header_echo` | 2 | 2 | 0 | 0 |
| `json_echo` | 2 | 2 | 0 | 0 |
| `plain_control` | 1 | 1 | 0 | 0 |
| `reflected_attribute` | 2 | 2 | 2 | 1 |
| `reflected_text` | 2 | 2 | 0 | 0 |

单 fixture 晋升试探：`quarantine`；原因：insufficient_distinct_datasets, insufficient_target_instances。
属性 oracle 不是把 OOD 阈值放宽，而是要求同一 surface 的 plain/url-percent pair、模型族一致、源码哈希、证据哈希和属性信号全部通过；文本/JSON/响应头即使回显 marker 也保持负例。

完整 JSON：`research\pg_pk_07_surface_generalization_v1.json`
协议：`research\pg_pk_07_surface_generalization_protocol_v1.json`
