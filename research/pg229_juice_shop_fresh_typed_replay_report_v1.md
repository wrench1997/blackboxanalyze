# PG-229 fresh Juice Shop typed replay

fresh seeds=2; candidate episodes=28; negative controls=2; typed effect=2; reference agreement=28; model self-errors=2

AI 先从冻结的通用 GET 表面词表选择路径；每颗 seed 重新创建 pinned Juice Shop。evaluator 只在后台提供状态转移计数，agent 只能看到结构投影。路径表面不是 payload-grounded 记录，不提升长期记忆。typed evidence 与模型提议冲突时，记录为 model_self_error 并保留 gate correction。

- seed=22901 rank=1 GET /does-not-exist-sift-control: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=2 GET /: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=3 GET /swagger.json: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=4 GET /robots.txt: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=5 GET /sitemap.xml: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=6 GET /ftp/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=7 GET /logs/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=8 GET /actuator: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=9 GET /api-docs: status=3xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=10 GET /graphql: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=11 GET /backup/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=12 GET /debug/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=13 GET /version: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22901 rank=14 GET /metrics: status=2xx; delta=1; agreement=True; typed=True; diagnoser=binding_failure
- seed=22902 rank=1 GET /does-not-exist-sift-control: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=2 GET /: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=3 GET /swagger.json: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=4 GET /robots.txt: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=5 GET /sitemap.xml: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=6 GET /ftp/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=7 GET /logs/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=8 GET /actuator: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=9 GET /api-docs: status=3xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=10 GET /graphql: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=11 GET /backup/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=12 GET /debug/: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=13 GET /version: status=2xx; delta=0; agreement=True; typed=False; diagnoser=oracle_unavailable
- seed=22902 rank=14 GET /metrics: status=2xx; delta=1; agreement=True; typed=True; diagnoser=binding_failure
