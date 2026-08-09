# PG-252 causal safe-probe gate capacity training

train=352; probe rows=250; Pikachu preprobe holdout=20; VulnerableApp preprobe OOD=54
Pikachu preprobe send=1.0; abstain=1.0; false_send=0; missed=0
VulnerableApp preprobe send=1.0; abstain=0.0; false_send=0; missed=0
canary=True; final_judge=candidate_eligible_for_next_preprobe_replay

PG-252 预测的是是否具备安全探针条件，不预测最终漏洞效果；最终效果仍由独立 oracle 与复放判定。
