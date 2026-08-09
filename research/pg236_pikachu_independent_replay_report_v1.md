# PG-236 independent Pikachu replay

seeds=2; fresh=28; raw=28; unique_templates=14; GET=14; POST=14
families={'sql': 16, 'redirect': 4, 'dom': 8}
跨 seed 的相同 token 模板保留为 replicate group，不伪装成新表面；后续训练按 seed 留出。
