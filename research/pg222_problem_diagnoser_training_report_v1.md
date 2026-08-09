# PG-222 Problem Diagnoser

device=cuda; rows=340 (train=90, holdout=250); counterfactual=306
selected hidden=64; guarded holdout accuracy=1.0; guarded positive false accepts=0

模型只判断过程问题：环境、绑定、oracle、候选无效、参考不一致、结果不匹配、模型自身决策错误或本地效果已被复放确认。它不生成 payload，也不把确认结果升级成任意网站漏洞结论。

PG-221 的真实修复被保留为过程教训：先前真假值构造多拼了结束引号，导致两条分支都无回显；修复后 fresh replay 变为 2/2。该轨迹说明诊断头需要检查绑定/候选构造，而不能直接责怪靶场。

反事实行已显式标记，只用于训练分类边界，不是新的靶场证据；raw payload/response body 未保存；memory promotion 和 vulnerability claim 均关闭。
