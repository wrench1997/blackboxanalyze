# PG-PK-11 共享匿名表示与族路由 head

共享 head 只做 XSS/SQL/访问控制/逻辑族路由；正向结论仍必须由族特异 typed oracle 复核。

状态：`accepted_for_diagnostic_routing`；训练样本：64；编码留出样本：31；新目标样本：23。

编码留出 accuracy：1.000；联合族外表面 accuracy：0.957；未知表面误路由：0.000；pair mean L2：0.000。

温度：0.250；校准后 ECE：0.039；abstain threshold：0.783；coverage：0.957。

共享 head 门禁：`pass`。共享 head 不具备正向放行权。
