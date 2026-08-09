# PG-261 active-belief capacity training

records=278; train=202; holdout=55; fresh_holdout=32; implementation OOD=21
selected_hidden=8192; adapter_params=168927370; fresh_rule=0.96875; fresh_family=1.0; fresh_unknown_abstain=1.0; OOD_family=0.8571428571428571
judge=candidate_eligible_for_next_replay; reasons=none; canary=True
PG-261 只训练抽象过程 token 与 unknown-family abstain 监督；oracle 不进入输入，真实公网能力不由本报告声明。

mask-aware pooling=enabled; classification is invariant to batch padding width.

PG-263 adds only the audited PG-262 abstract records; raw wire/response bodies remain excluded and all promotion gates remain blocked until independent review.

final_audit=pg263-final-report-audit-v1; decision=candidate_eligible_for_next_replay; reasons=none; capacity_variants=[2048, 4096, 8192]
