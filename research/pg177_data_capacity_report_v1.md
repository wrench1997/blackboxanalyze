# PG-177 data and capacity sweep

- train rows: **7200**
- generators: **surface_permutation_v7, failure_recovery_v8, transport_matrix_v9, semantic_delta_v10**
- baseline existing/new-OOD PPL: **261.94270508 / 276.55533574**
- gates: **[{'seed': 17701, 'existing_split_gate': True, 'new_ood_gate': True, 'existing_aggregate': 3.34723866, 'new_ood_aggregate': 2.08915091, 'pass': True}, {'seed': 17702, 'existing_split_gate': True, 'new_ood_gate': True, 'existing_aggregate': 3.3421894, 'new_ood_aggregate': 2.08907346, 'pass': True}]**
- capacity comparison: **[{'seed': 17701, '160m_scratch_new_ood': 2.06274149, '200m_scratch_new_ood': 2.06510986, '200m_better': False}, {'seed': 17702, '160m_scratch_new_ood': 2.06300013, '200m_scratch_new_ood': 2.06379658, '200m_better': False}]**

该轮只验证抽象 Rule-IR 的 next-token 学习与跨生成器泛化，不生成漏洞标签或攻击 payload。
