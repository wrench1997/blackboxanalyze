# PG-247 VulnerableApp capacity training

train=96; holdout=133; canary=34
selected hidden=256; holdout send=1.0; abstain=1.0; false_send=0
canary pass=True; final_judge=candidate_eligible_for_next_replay

训练集包含 PG-246 的抽象 DOM 过程，但所有 Pikachu 来源与 VulnerableApp seed 24603 留出；旧 SQL/XSS canary 只用于遗忘审计。最终判定来自独立硬门，不来自模型自报。
