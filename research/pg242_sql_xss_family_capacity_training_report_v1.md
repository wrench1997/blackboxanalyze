# PG-237 non-trivial seed-heldout capacity training

train=128; holdout=23; holdout_actions={'send_candidate': 18, 'abstain': 5}
selected hidden=512; token=0.69181586; lane=0.9130435; repair=0.39130434; positive_recall=1.0; abstain_recall=1.0; false_send=0; missed_send=0
safety_abstain_gate=True; capability_gate=True

留出集同时包含 typed positive 和 abstain，避免全 abstain 自我安慰；正例仍是本地只读结果 fixture，不等于任意站点漏洞结论。
