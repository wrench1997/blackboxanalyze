# PG-01 Payload Grounding Ablation

本实验只运行本地合成 DOM/SQL oracle；候选是受限的安全 probe manifest，不发送网络请求、不执行脚本、不访问数据库。策略输入不包含 family/grammar/target 标签。

| 目标 | 策略 | warmup 命中率 | warmup 首次命中 | replay 命中率 | replay 精度 | 稳定复放 | 新 marker 复放 |
|---|---|---:|---:|---:|---:|---:|---:|
| sql_blind_branch | random | 0.80 | 3.75 | 0.20 | 0.07 | 1.0 | 1.0 |
| sql_blind_branch | ucb | 1.00 | 7.0 | 0.80 | 0.53 | 1.0 | 1.0 |
| sql_blind_branch | ucb_memory | 1.00 | 7.0 | 1.00 | 1.00 | 1.0 | 1.0 |
| sql_error_channel | random | 0.80 | 3.25 | 0.40 | 0.13 | 1.0 | 1.0 |
| sql_error_channel | ucb | 1.00 | 3.2 | 0.20 | 0.20 | 1.0 | 1.0 |
| sql_error_channel | ucb_memory | 1.00 | 3.2 | 1.00 | 1.00 | 1.0 | 1.0 |
| sql_time_channel | random | 0.80 | 4.5 | 0.00 | 0.00 | 1.0 | 1.0 |
| sql_time_channel | ucb | 1.00 | 5.8 | 0.60 | 0.47 | 1.0 | 1.0 |
| sql_time_channel | ucb_memory | 1.00 | 5.8 | 1.00 | 1.00 | 1.0 | 1.0 |
| xss_double_decode | random | 0.80 | 4.5 | 0.20 | 0.07 | 1.0 | 1.0 |
| xss_double_decode | ucb | 1.00 | 3.4 | 0.40 | 0.13 | 1.0 | 1.0 |
| xss_double_decode | ucb_memory | 1.00 | 3.4 | 1.00 | 1.00 | 1.0 | 1.0 |

结论边界：UCB 只证明‘在受限候选 grammar 上用 oracle 反馈做选择’的控制器能力；它不等于神经模型，也不表示已从公网渗透语料学习真实 payload。下一步若接入模型，替换候选生成器即可复用同一 evidence/replay 契约。

原始 JSON：`research\payload_grounding_ablation_v1.json`
