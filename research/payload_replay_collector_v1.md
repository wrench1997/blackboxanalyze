# PG-03 Local Replay Collector

采集器仅通过 `http://127.0.0.1:3100` 的只读 replay adapter 运行 GET probe；每条记录先 fresh reset，响应只保存 bounded projection、长度/哈希和 Rule IR，不保存原始 body。

样本数：20；来源数：10；Rule IR 一致性：1.00

| split | train 样本 | test 样本 | feature coverage | fail-closed abstention |
|---|---:|---:|---:|---:|
| source_split_same_family | 10 | 10 | 1.00 | 0.00 |
| family_holdout_structural_transfer | 8 | 2 | 1.00 | 0.00 |
| family_holdout_unseen_surface | 7 | 3 | 0.00 | 1.00 |

边界：这是本地 ASGI 应用的真实路由响应回放，不是公网数据，也不是 evaluator 确认。下一阶段可将同一 collector 接到用户明确授权的本地容器靶场。

原始 JSON：`research\payload_replay_collector_v1.json`
