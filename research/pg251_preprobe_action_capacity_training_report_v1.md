# PG-251 causal pre-probe action capacity training

train=352; preprobe train=250; Pikachu preprobe holdout=20; VulnerableApp preprobe OOD=54
Pikachu preprobe send=0.0; abstain=1.0; false_send=0; missed=14
VulnerableApp preprobe send=0.0; abstain=1.0; false_send=0; missed=12
canary=True; final_judge=blocked

pre-probe prefix 在 phase=diagnose 之前截断；目标只作为 action head 标签，不进入输入 token。
