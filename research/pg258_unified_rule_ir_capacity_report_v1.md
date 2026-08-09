# PG-258 unified SQL/XSS Rule-IR capacity

train=164; holdout=23; implementation OOD=21
selected_hidden=1024; holdout_rule=0.60869569; holdout_family=0.65217394; holdout_next_token=0.87983193
judge=blocked_insufficient_generalization; reasons=holdout_rule_accuracy_ge_0_80, holdout_family_accuracy_ge_0_80, implementation_ood_family_accuracy_ge_0_60, holdout_each_rule_class_support_ge_2
canary=True; state_unchanged=True; action_metrics_unchanged=True
旧发送/拒答策略保持冻结；新头只学习抽象 Rule-IR 与 surface family。PG-242/PG-257 oracle 目标不进入输入，结果不代表公网能力。
