# Loop 12 v5：多步 belief + family-specific surface discriminator

这轮把“探测—判断—再探测”做成了真正的工程链：

`shadow GET → surface discriminator → belief posterior → information-gain probe → Rule IR binding → fresh target replay`

## 模型状态

- response head：跨种子平均 **99.9722%**，最低 **99.8889%**。
- Rule IR v4：IID **100%**，fresh holdout 平均 **84.5833%**，`dom_double_decode` **100%**。
- family-specific surface discriminator：合成 validation/fresh seeds 均 **100%**；真实 Juice Shop 普通 surface 置信度较低，这个 domain shift 已被单独记录。

## Belief v2 矩阵

协议：[hidden matrix v6](juice_shop_loop_12_hidden_matrix_protocol_v6.json)，每一步把 discriminator likelihood 经过 prior mixing 后更新，不再简单连乘；探针选择使用 Jensen–Shannon information gain。

| 策略 | evaluator hit | fully bound | 信息增益 | 最终 entropy |
|---|---:|---:|---:|---:|
| belief response | 0/7 | 6/7 | 7 步均非负 | 1.935418 |
| belief ablation | 0/7 | 7/7 | 7 步均非负 | 1.940252 |

修复前 belief 会被普通 `/` surface 推到 `url_redirect≈0.59`；修复后最终 posterior 保持在各类约 0.11–0.18，说明没有单个错误观察把系统锁死。

## 漏洞确认边界

本轮新的七族矩阵仍没有 fresh-target evaluator transition，因此没有新增已确认漏洞。之前单独 `/metrics` observability 实验的确认结果保留。Rule IR binding 只是结构化证据，不是漏洞证明。

全量回归：[57 passed，1 warning](D:/workspace/blackboxanalyze/tests)。

产物：[v5 总报告](D:/workspace/blackboxanalyze/research/juice_shop_loop_12_final_v5_report.json)、[belief runs](D:/workspace/blackboxanalyze/research/juice_shop_loop_12_hidden_matrix_runs_v6.json)、[surface discriminator](D:/workspace/blackboxanalyze/artifacts/surface-discriminator-loop-12-20260931/surface_discriminator.pt)。

下一步不是继续堆分类器，而是增加带安全 reset 的 within-target adaptive episode，让每个 probe 得到真实状态反馈，再校准 action-level likelihood。
