# PG-63 独立目标区复放

PG61 模型冻结；PG63 不重新训练、不导入 PG61 任务生成器。

| 模式 | 目标成功率 | 动作准确率 | 阴性误报 | 未知弃权 | 动作熵 |
|---|---:|---:|---:|---|---:|
| canonicalized | 1.0 | 1.0 | 0 | True | 0.954434 |
| raw-shift fail-closed | 0.0 | 0.5 | 0 | True | 0.0 |

硬门：`passed`；formal capability claim=false；训练/记忆不晋升。
