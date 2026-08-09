# Tiny Rule GPT 黑盒记忆先导实验

## 实验问题

在源码、Rule IR 和漏洞标签均不可见的前提下，一个小型因果 Transformer 能否仅凭黑盒示例轨迹，在完整未见的规则族上预测行为？轨迹记忆是否提供了公平、可测量的增益？

## 设计

- 模型：875,778 参数，4 层、`d=128`、4 heads、384 bytes 上下文；
- 硬件：RTX 3060 12GB；
- 数据：1,200 个自动生成并验真的程序；
- 训练输入：`<TRACE> 输入:输出 ... <QUERY> 待预测输入`；
- 训练规则族：`numeric_boundary`、`truthiness_gate`；
- 验证留出族：`substring_origin`；
- 最终测试留出族：`authorization_or`；
- 训练：10 epochs；
- 公平对照：相同模型、数据、epoch 和随机种子，从训练开始完全移除 TRACE。

## 结果

| 条件 | 未见 authorization_or 准确率 |
|---|---:|
| 有黑盒轨迹记忆 | **88.58%** |
| 无轨迹训练对照 | **50.00%** |
| 记忆净增益 | **+38.58 pp** |

有记忆模型训练耗时 28.7 秒。它在未见 `substring_origin` 上达到 85.75%，但在训练分布内部仅达到 61.67%。这意味着实验支持“轨迹记忆对族外规则恢复有用”，但不支持“当前模型已经普遍掌握规则归纳”。

## 根因诊断

训练分布的主要失败来自数值边界。当前 byte tokenizer 把 `169` 和 `170` 当作字符序列，没有数值大小、距离、排序和边界关系的归纳偏置。直接扩大参数量不是最小有效改动。

下一项预注册实验应比较：

1. 原始 byte prompt；
2. 匿名字段 + 数值 rank/delta/比较关系的 Common Semantic prompt；
3. 相同参数与训练预算。

预期：`numeric_boundary` IID 准确率至少提高 15 pp，`authorization_or` family-holdout 不得下降超过 2 pp，prompt token 数不得增加超过 20%。若失败，再考虑专用数值 head 或神经—符号混合结构。

## 证据

- 有记忆报告：`tiny_rule_gpt_experiment.json`
- 无记忆报告：`tiny_rule_gpt_no_memory_experiment.json`
- Checkpoint：`artifacts/rule-memory-pilot/tiny_rule_gpt.pt`
- 可复现训练脚本：`scripts/train_rule_memory_pilot.py`

结论范围仅限本次合成程序族和固定实验预算。
