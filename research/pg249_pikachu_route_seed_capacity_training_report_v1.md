# PG-249 Pikachu route/seed capacity training

train=176; Pikachu holdout=20; VulnerableApp OOD=54
Pikachu positive=1.0; abstain=1.0; false_send=0; missed=0
VulnerableApp positive=1.0; abstain=1.0; false_send=0; missed=0
canary=True; final_judge=candidate_eligible_for_next_replay

Pikachu 实用能力与跨实现 OOD 分开报告；通过前者不等于任意实现泛化。
