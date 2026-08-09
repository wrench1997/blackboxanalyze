# PG-237 non-trivial seed-heldout capacity training

train=121; holdout=14; holdout_actions={'abstain': 4, 'send_candidate': 10}
selected hidden=1024; token=0.81302521; lane=1.0; repair=1.0; positive_recall=1.0; abstain_recall=1.0; false_send=0; missed_send=0
safety_abstain_gate=True; capability_gate=True

留出集同时包含 typed positive 和 abstain，避免全 abstain 自我安慰；正例仍是本地只读结果 fixture，不等于任意站点漏洞结论。
