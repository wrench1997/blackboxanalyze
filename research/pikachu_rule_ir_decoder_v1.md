# Pikachu PG-PK-01 Rule IR 解码器小试验

训练只使用阶段 1 的安全 canary，阶段 2 的编码变体完全留作测试。模型看到的是 action/probe/response shape，不读取 family、evaluator 或漏洞确认标签；输出仍被限制为 grammar-checked Rule IR，低置信度或新表面必须 abstain。

设备：`cuda`；训练样本：7；测试样本：1。

| split | 找到出口 | 误报 | abstain | Rule IR 输出 |
|---|---:|---:|---:|---:|
| stage1_to_stage2_encoded_variant | 0.00 | 0.00 | 1.00 | 0.00 |

这里的“找到出口”仅指抽象族/规则模板迁移成功，不是漏洞 evaluator 确认。数据只有 7 条，结果用于检验训练管线和 fail-closed 行为，不能外推到 Pikachu 全部漏洞。

Checkpoint：`artifacts\pg-pk-01-rule-decoder\decoder.pt`
完整 JSON：`research\pikachu_rule_ir_decoder_v1.json`
