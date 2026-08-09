# PG-211 process learning audit

status=attached_but_not_learned; AI 实际发包=12; 路由/字段/哈希绑定=12; 独立参考一致=12
决策签名种类=1; evaluator 反馈回流=0; history 特征=0; online update=False

AI 的请求已经真实发出并且与路由/字段/哈希绑定；当前 checkpoint 仍是冻结决策头，反馈没有进入策略，不能声称模型从失败中学会了下一步。

本审计只保存计数、决策签名和哈希；不保存原始 payload 或响应正文。
