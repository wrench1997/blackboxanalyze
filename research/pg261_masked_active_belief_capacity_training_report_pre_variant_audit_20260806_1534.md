# PG-261 active-belief capacity training

records=258; train=191; holdout=46; fresh_holdout=23; implementation OOD=21
selected_hidden=4096; adapter_params=50913418; fresh_rule=0.9130434782608695; fresh_family=1.0; fresh_unknown_abstain=1.0; OOD_family=0.8571428571428571
judge=blocked_insufficient_generalization; reasons=holdout_rule_accuracy_ge_0_80; canary=True
PG-261 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。

mask-aware pooling=enabled; classification is invariant to batch padding width.
