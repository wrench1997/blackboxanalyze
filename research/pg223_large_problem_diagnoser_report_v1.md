# PG-223 frozen XXL problem diagnoser

device=cuda; frozen parameters=101486878; train=90; holdout=250
selected adapter hidden=64; guarded holdout accuracy=1.0; guarded positive false accepts=0

这是容量实验：PG-191 的约 101M 参数过程体被冻结，只训练诊断 adapter。没有把 route、payload、响应正文或 evaluator target 喂给模型。

当前数据仍以 PG-222 的小规模真实投影和标记反事实为主，不能把完美 holdout 分数解释成通用渗透能力；下一步必须在未见实现的本地 GET/POST 靶场上收集真实失败 trace。
