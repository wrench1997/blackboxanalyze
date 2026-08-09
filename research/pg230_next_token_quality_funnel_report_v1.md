# PG-230 next-token quality funnel

device=cuda; raw=176; unique=33; train=10; holdout=11; quarantine=12
lanes={'gold': 2, 'silver': 12, 'hard_negative': 7, 'quarantine': 12}; duplicates=143
selected hidden=128; holdout token accuracy=0.68899522; lane accuracy=0.27272728; repair accuracy=0.27272728; self-error recall=0.5

next-token loss 只作为表示学习指标；gold/hard-negative/silver/quarantine 分层决定数据能否进入对应训练头。冻结 XXL 主体哈希前后相同，未把本轮小数据误报成通用能力。
