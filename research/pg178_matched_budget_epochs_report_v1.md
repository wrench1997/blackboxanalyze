# PG-178 matched-budget epochs

- train rows: **7200**
- target epochs: **(2, 4)**
- capacity comparison: **{'2': [{'seed': 17701, 'epoch': 2, '160m_existing': 3.98053623, '200m_existing': 4.05943654, '160m_new_ood': 2.05703754, '200m_new_ood': 2.06000982, '200m_better_new_ood': False, '200m_better_existing': False}, {'seed': 17702, 'epoch': 2, '160m_existing': 3.86594506, '200m_existing': 3.85763363, '160m_new_ood': 2.05780078, '200m_new_ood': 2.05804279, '200m_better_new_ood': False, '200m_better_existing': True}], '4': [{'seed': 17701, 'epoch': 4, '160m_existing': 3.75906411, '200m_existing': 3.96085147, '160m_new_ood': 2.04111202, '200m_new_ood': 2.04236578, '200m_better_new_ood': False, '200m_better_existing': False}, {'seed': 17702, 'epoch': 4, '160m_existing': 3.88897951, '200m_existing': 3.64926508, '160m_new_ood': 2.03983835, '200m_new_ood': 2.04086757, '200m_better_new_ood': False, '200m_better_existing': True}]}**

该轮专门验证更大容量是否需要更多 token budget；optimizer 在 PG177 边界重建，因此不把它当作无条件连续训练。
