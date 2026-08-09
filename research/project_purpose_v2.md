# 项目最终目的（v2）

本项目研究的不是“背诵漏洞字符串”，也不是把一次靶场命中包装成模型变强。最终目标是验证：一个 AI 能否在**授权的本地黑盒靶场**中，依靠多步观察、主动探针、长期记忆和抽象 Rule IR，学习可迁移的漏洞行为规则，并在未见过的目标、来源、编码和漏洞族上稳定地：

1. 先建立基线，再选择安全、非破坏性的 GET/POST 等通道探针；
2. 从状态码、内容类型、结构变化、DOM/SQL AST/授权边界等**类型化 oracle**中更新 belief；
3. 区分“反射、语法错误、405、超时、状态差异”等弱信号与真正的漏洞效果；
4. 在证据不足时 abstain，而不是用模型置信度代替验证；
5. 输出人可读、跨语言可复用的 Rule IR、探针意图和证据哈希；
6. 在 fresh reset 的新目标上回放，统计 typed recall、precision、误报率、校准和查询成本。

## 数据的真实用途

数据必须分层，不能混成“训练集”：

- **Schema smoke test**：如 PG-31 当前合成 manifest。它只验证字段、隔离和 gate，`training_eligible=false`，不是能力数据。
- **Replay evaluation**：如 PG-28/29。它包含实际本地 GET/POST 回放的请求元数据、响应投影、reset 和 oracle 证据；没有类型化正例时，只能用于负例、abstain 和工程回归。
- **Training candidate**：只有实际回放、正负配对、类型化 oracle、fresh reset、来源授权、证据哈希和跨目标复验全部齐全的样本，才可进入训练候选；最终还要通过独立来源/族外能力门。

## 成功标准

成功不是“脚本跑通”或“反射被搜到”，而是候选模型相对冻结 baseline 在独立的 train/dev/family-holdout/ood-source/negative-control 矩阵上获得预注册的 typed recall 提升，同时 precision、误报率、abstain precision、ECE 和最差分桶不退化。任何缺少真实回放上下文、类型化正例或跨切分隔离的报告都只能是 `blocked` 或 `no_proven_gain`。

## 明确不做的事

- 不在授权范围外探测，不访问凭据，不执行脚本、破坏性 SQL 或外部网络操作；
- 不把合成 projection、单一靶场、HEAD-only、405、反射、错误文本或超时单独当作漏洞成功；
- 不把 unit test、训练 loss、普通 accuracy 或 UI 状态当作模型能力证据；
- 不为了“看起来变强”向长期记忆或训练集写入未经 oracle 验收的记录。

因此，工程优先级是：**真实本地 replay 上下文 → typed oracle 与配对证据 → 跨目标/跨族评估 → 再讨论模型架构、训练方法和长期记忆**。
