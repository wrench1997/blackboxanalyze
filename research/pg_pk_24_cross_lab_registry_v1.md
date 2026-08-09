# PG-24 跨靶场数据注册表

该表只登记本地实验数据源的边界和质量，不复制原始 payload、响应正文、challenge key 或 evaluator 标签。Pikachu 是当前唯一满足 Catalog 训练契约的来源；其他靶场先作为隔离评估源，待补齐授权 probe、oracle、fresh reset 和 source hash 后才能进入训练。

| target | app family | samples/labs | instances | training role | eligible |
|---|---|---:|---:|---|---:|
| `pikachu` | `pikachu` | 44 | 3 | authorized_catalog_training_and_holdout | yes |
| `juice_shop_loop12` | `juice_shop` | 5 | 0 | evaluation_only_until_canonical_safe_catalog_is_collected | no |
| `sql_differential_fixture` | `synthetic_sql_fixture` | 15 | 0 | evaluation_only_until_source_catalog_is_attested | no |
| `logic_access_fixture` | `synthetic_logic_access_fixture` | 60 | 3 | evaluation_only_until_source_catalog_is_attested | no |
| `heterogeneous_surface_fixture` | `heterogeneous_surface_fixture` | 108 | 3 | evaluation_only_until_pair_catalog_is_attested | no |
| `rule_maze` | `abstract_rule_maze` | 12 | 0 | protocol_and_oracle_evaluation_only | no |

## 训练扩展条件

1. 每个靶场至少有独立 source hash、container/image digest、fresh reset 记录和 loopback 范围。
2. 每个样本保存 safe probe、编码、bounded oracle projection、Rule IR、evidence hash；不保存原始正文或凭据。
3. 按靶场实例和来源隔离 train/validation/test；同一模板不同标签不能算新来源。
4. 未满足条件的 Juice Shop、SQL、logic、maze 数据只能做 OOD/abstain 测试。

完整 JSON：`research\pg_pk_24_cross_lab_registry_v1.json`
