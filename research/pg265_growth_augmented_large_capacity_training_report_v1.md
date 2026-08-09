# PG-261 active-belief capacity training

records=310; train=218; holdout=71; fresh_holdout=48; implementation OOD=21
selected_hidden=8192; adapter_params=168927370; fresh_rule=0.9791666666666666; fresh_family=1.0; fresh_unknown_abstain=1.0; OOD_family=0.8571428571428571
judge=candidate_eligible_for_next_replay; reasons=none; canary=True
PG-261 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。

mask-aware pooling=enabled; classification is invariant to batch padding width.
