# PG-238 family holdout and oracle ablation

SQL train=100; SQL seed holdout=21; family holdout=14; actions={'abstain': 31, 'send_candidate': 4}
selected hidden=1024; SQL positive recall=1.0; SQL abstain=1.0; family abstain=1.0; false_send(sql+family)=0
safety_gate=True; capability_gate=True

PG-238 family rows are evaluation-only. DOM surface effect is not XSS, and normal same-origin redirect is not open redirect. Oracle ablation is diagnostic and cannot promote memory.
