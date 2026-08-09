# PG-260 active-belief capacity training

records=258; train=191; holdout=46; fresh_holdout=23; implementation OOD=21
selected_hidden=4096; adapter_params=50913418; fresh_rule=0.9130434782608695; fresh_family=0.9565217391304348; fresh_unknown_abstain=1.0; OOD_family=0.14285714285714285
judge=blocked_insufficient_generalization; reasons=holdout_rule_accuracy_ge_0_80, implementation_ood_family_accuracy_ge_0_60; canary=True
audit=pg260-holdout-support-recalculation-v1; weights_changed=False; artifact_unchanged=True
PG-260 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。
