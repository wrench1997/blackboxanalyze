# Loop 12 v4：工程化完善结果

本轮完成三项工程增强，并重新跑了 active-policy 七族矩阵。

## 已完成

1. **置信度校准**：在 validation split 上以 95% precision gate 自动选择 abstention threshold，当前 checkpoint 持久化阈值为 **0.337907**，不再使用运行器硬编码的 0.55。
2. **证据 provenance**：每个 Rule IR binding 都带 canonical JSON 的 SHA-256；写入 runs 前校验 schema、slot 数量、bound 数量和 evidence hash，binding 明确标记 `executable: false`。
3. **Active probe planner**：用 `0.75 × decoder entropy + 0.20 × (1-confidence) + 0.05 × response score` 选下一探针；不读取 family label，仍是 GET-only、local-only、无 evaluator endpoint。

## Active matrix

协议：[hidden matrix v4](juice_shop_loop_12_hidden_matrix_protocol_v4.json)

| 策略 | evaluator hit | fully bound | bound slots | abstain |
|---|---:|---:|---:|---:|
| active response head | 0/7 | 4/7 | 8 | 3/7 |
| active response-disabled ablation | 0/7 | 4/7 | 8 | 3/7 |

Active entropy 确实改变了多个 family 的探测动作，binding coverage 从固定 v4 训练策略的 3/7 提升到 4/7；但 evaluator 命中仍为 0/7。这说明工程层的探针和证据链已可复现，核心缺口仍是 family-disambiguating interaction/action search，而不是记录格式或权限问题。

## 当前模型状态

Rule IR v4 保持编码深度改进：IID validation 100%，fresh holdout 平均 84.5833%，`dom_double_decode` 100%；训练使用 RTX 3060 CUDA。普通 API surface 仍可能被分类为 input-validation，低置信时会 abstain，不会伪造漏洞结论。

全量回归：**54 passed，1 warning**。

产物：[工程 v4 报告](D:/workspace/blackboxanalyze/research/juice_shop_loop_12_final_v4_report.json)、[active runs](D:/workspace/blackboxanalyze/research/juice_shop_loop_12_hidden_matrix_runs_v4.json)、[Rule IR checkpoint](D:/workspace/blackboxanalyze/artifacts/rule-ir-decoder-loop-12-20260899-v4/rule_ir_decoder.pt)。

下一步应训练 family-specific surface discriminator，并让 active probe 以多步反馈更新 beliefs；在此之前不宣称七族 evaluator 泛化已经解决。
