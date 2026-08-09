# PG-213 history policy training

device=cuda; train=52; holdout=52; counterfactual=26
train={'count': 52, 'accuracy': 0.96153849, 'abstain_recall': 0.95238095, 'retry_alternate_recall': 1.0, 'unsafe_allow_count': 0, 'behavioral_abstain_error_count': 2}; holdout={'count': 52, 'accuracy': 0.96153849, 'abstain_recall': 0.95238095, 'retry_alternate_recall': 1.0, 'unsafe_allow_count': 0, 'behavioral_abstain_error_count': 2}

该 head 只学习失败反馈与绑定失败后的动作选择；artifact 仍是诊断用，未接管真实发包，也未提升长期记忆。
