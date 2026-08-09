# Loop 12 神经响应头 shadow 验证

状态：**单个 hidden observability surface 的神经干预通过；完整 Loop 12 仍未完成。**

| 条件 | shadow 观察 | 评估动作 | 评估成功 | 目标分数 |
|---|---:|---|---:|---:|
| 训练响应头 | 5 | `/metrics` | 是 | 0.999 |
| 关闭响应槽 | 5 | `/.well-known/security.txt` | 否 | 0.723 |

训练响应验证准确率为 90.17%，旧合成族最差回归为 +0.00%，参数数目保持 908546。训练模型在干净评估容器中只执行一次 `GET /metrics` 并成功；关闭响应槽后选择 `security.txt`，目标状态不变。

这确认了“HTTP 响应语义 + 参数守恒响应头”是当前动作接地瓶颈的有效修复，但只覆盖一个 hidden surface；下一轮必须在其余 hidden family 上重复 shadow/evaluation，并继续记录 Rule IR 解码仍然缺失这一限制。
