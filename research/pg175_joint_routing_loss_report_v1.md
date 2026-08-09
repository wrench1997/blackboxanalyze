# PG-175 joint routing/loss search

- baseline aggregate PPL: **2.51077335**
- variant aggregate PPL: **{'low_lr_replay': 2.43265515, 'balanced_low_lr': 2.43308846, 'old_first_weighted': 2.52729766, 'routed_low_lr': 2.43481149}**
- strict gate: **{'low_lr_replay': False, 'balanced_low_lr': False, 'old_first_weighted': False, 'routed_low_lr': True}**
- selected: **routed_low_lr**

完整 holdout 是硬门；没有超过基线的分支不会晋级。

