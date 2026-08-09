# PG-216 real-trace capacity training

device=cuda; base train=600; PG-215 train=48; variants=4
capacity comparison=[{'seed': 17701, '160m_seed_holdout': 49.71626744, '200m_seed_holdout': 62.73875098, '160m_route_holdout': 48.65831073, '200m_route_holdout': 62.090472, '200m_better_seed_holdout': False, '200m_better_route_holdout': False}, {'seed': 17702, '160m_seed_holdout': 56.23541157, '200m_seed_holdout': 49.12897867, '160m_route_holdout': 56.05055733, '200m_route_holdout': 48.27655043, '200m_better_seed_holdout': True, '200m_better_route_holdout': True}]
200M better on both seed and route holdout across seeds=False

该轮是 next-token Rule-IR 训练，不是漏洞分类器。所有 checkpoint 仍为诊断用途；只有跨 seed/route OOD 与 typed oracle 同时通过，才允许接入发包策略。
