# PG-176 routed multi-seed/new OOD

- seeds: **(17601, 17602, 17603)**
- fourth generator rows: **1000**
- baseline existing aggregate: **2.51077335**
- gates: **[{'seed': 17601, 'existing_split_gate': True, 'existing_aggregate_gate': True, 'fourth_ood_gate': True, 'pass': True}, {'seed': 17602, 'existing_split_gate': True, 'existing_aggregate_gate': True, 'fourth_ood_gate': True, 'pass': True}, {'seed': 17603, 'existing_split_gate': True, 'existing_aggregate_gate': True, 'fourth_ood_gate': True, 'pass': True}]**
- all seeds pass: **True**

该轮仍只验证抽象 Rule-IR 训练泛化，不产生漏洞标签。

