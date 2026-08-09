# PG-181 manifest decoder + local replay

device=cuda; trained runs=6; selected=moe_large seed=18101
replay sent=5; controller abstain=0; typed positives=0

模型只选择 baseline/control/safe_candidate/abstain；字段由浏览器 manifest 提供，回放器在发送前再次校验。
