# PG-236 seed-heldout failure-conditioned training

train=93; seed23632_holdout=14; selected hidden=256
holdout token=0.87545788; lane=1.0; repair=1.0; abstain_recall=1.0; false_send=0; safety_abstain_pass=True; capability_gate=False

seed23632 完全未进入训练；projection-only replay 在无 typed oracle 时只能学习 abstain。全 abstain 留出集只能通过安全门，不能证明模型会发现漏洞。
