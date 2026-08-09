# PG-174 full holdout routing

- full eval counts: **{'base_holdout': 1210, 'typed_holdout': 203, 'pg168_ood': 1000, 'pg170_ood': 1000, 'pg172_ood': 1000}**
- baseline 160M aggregate PPL: **2.51077335**
- variant aggregate PPL: **{'uniform': 2.78323451, 'replay_weighted': 2.78190839, 'source_routed': 2.52150573}**
- best variant: **source_routed**
- beats baseline: **False**

本轮使用完整 holdout；若分支未超过未训练基线，不标记为模型增强。

