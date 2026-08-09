# PG-233 cross-family capacity

device=cuda; unique=115; train=59; source_holdout=13; family_holdout=5; double_holdout=4
selected hidden=256; strict_pass=False
source_holdout: token=0.75101215; lane=0.0; repair=0.0; self_error_recall=0.0
family_holdout: token=0.74736842; lane=0.2; repair=0.2; self_error_recall=0.0
double_holdout: token=0.73684211; lane=0.0; repair=0.0; self_error_recall=0.0
容量增大只有在 source+family 双重留出也通过时才有意义；本报告不自动晋级长期记忆。
