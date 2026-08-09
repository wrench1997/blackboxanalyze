# PG-257 wide-byte Rule-IR class capacity training

train=8; holdout=8; train_classes={'blind_boolean': 1, 'syntax_boundary': 4, 'widebyte_escape_boundary': 3}; holdout_classes={'blind_boolean': 1, 'syntax_boundary': 4, 'widebyte_escape_boundary': 3}
selected_hidden=2048; holdout_rule_accuracy=1.0; widebyte_recall=1.0; next_token=0.95955882

训练只使用抽象失败/复放过程 token；payload class 是独立 reference/evaluator 产生的监督标签，不进入模型输入。结果不能推出公网漏洞能力，晋级保持冻结。
