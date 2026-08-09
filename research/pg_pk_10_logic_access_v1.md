# PG-PK-10 逻辑/访问控制 typed oracle 与反事实回放

本轮使用三个未参与 head 训练的本地 fixture target（gamma/delta/epsilon）。场景覆盖 truthy 授权边界、业务边界偏移和 challenge 绑定缺失；每条正例都有 plain/url-percent pair，并配有相邻的正常 200/403 反事实。

样本：60；模型-only 接受：24；模型-only 反事实误接：0；typed oracle 复核样本：24；pair：12。

| target | model-only accepts | counterfactual candidates | revalidated pairs |
|---|---:|---:|---:|
| `gamma:8797` | 8 | 0 | 4 |
| `delta:8798` | 8 | 0 | 4 |
| `epsilon:8799` | 8 | 0 | 4 |

access_control 记忆门：`promote`；logic 记忆门：`promote`。
模型输出只产生候选 Rule IR；只有 typed boundary、同一 pair 双编码、fresh target、证据哈希和无状态副作用同时成立，才进入 oracle revalidation。

完整 JSON：`research\pg_pk_10_logic_access_v1.json`
协议：`research\pg_pk_10_logic_access_protocol_v1.json`
