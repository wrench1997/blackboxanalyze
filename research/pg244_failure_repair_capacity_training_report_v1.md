# PG-237 non-trivial seed-heldout capacity training

train=161; holdout=14; holdout_actions={'send_candidate': 10, 'abstain': 4}
selected hidden=2048; token=0.89264706; lane=1.0; repair=0.92857146; positive_recall=1.0; abstain_recall=1.0; false_send=0; missed_send=0
safety_abstain_gate=True; capability_gate=True

留出集同时包含 typed positive 和 abstain，避免全 abstain 自我安慰；正例仍是本地只读结果 fixture，不等于任意站点漏洞结论。
