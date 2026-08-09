# PG-03 Catalog Rule IR Decoder V2

V2 使用 surface/context 双塔、噪声增强、监督式对比损失和最小族支持门；仍只输出 grammar-checked Rule IR。`exit_found` 不是漏洞 evaluator 确认。

| split | exit found | false positive | abstain | Rule IR emitted |
|---|---:|---:|---:|---:|
| source_split_same_family | 0.70 | 0.00 | 0.30 | 0.70 |
| family_holdout_structural_transfer | 0.00 | 0.00 | 1.00 | 0.00 |
| family_holdout_unseen_surface | 0.00 | 0.00 | 1.00 | 0.00 |

V2 的增强重点是稳定表示和 fail-closed 决策，而不是在不可辨识的同形响应上硬猜。若训练 split 中某族少于 2 条样本，强制 abstain。

边界：仍是同一 in-repo 本地 ASGI adapter 的来源/族/表面隔离；要证明独立靶场泛化，还需接入第二个授权本地实现。

原始 JSON：`research\pg03_rule_ir_decoder_v2.json`
