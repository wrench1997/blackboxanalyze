# PG-259 active-belief Rule-IR capacity

train=174; fresh_train=10; holdout=31; fresh_holdout=8; implementation OOD=21
selected_hidden=4096; holdout_rule=0.64516127; holdout_family=0.7096774; fresh_rule=0.5; fresh_belief=1.0; holdout_next_token=0.8878249
judge=blocked_insufficient_generalization; reasons=holdout_rule_accuracy_ge_0_80, holdout_family_accuracy_ge_0_80, fresh_route_rule_accuracy_ge_0_70, implementation_ood_family_accuracy_ge_0_60
canary=True; state_unchanged=True; action_metrics_unchanged=True
新增 active-belief/probe 头只读取抽象轨迹；oracle 仅作监督，旧动作策略保持冻结。结果不代表公网能力。
