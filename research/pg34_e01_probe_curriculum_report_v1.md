# PG-34-E01 探针课程与停止策略

这是基于 PG-33 真实 loopback HTTP trace 的 controller 上界实验，不是训练结果。模型可见字段不含 family、oracle、证据哈希或原始响应；typed oracle 只在动作完成后用于确认停止。

| policy | typed recall | exit found | FPR | median queries | mean queries |
|---|---:|---:|---:|---:|---:|
| fixed GET+POST | 1.00 | 0.86 | 0.00 | 4.0 | 4.00 |
| active stop | 1.00 | 0.86 | 0.00 | 2.0 | 2.29 |

查询平均减少：1.71；结果不能授权训练或长期记忆：`False` / `False`。

下一步：把 controller 的停止标签改成延迟反馈训练目标，并在独立实现上复测；不能把这个 oracle 上界当成模型泛化证明。
