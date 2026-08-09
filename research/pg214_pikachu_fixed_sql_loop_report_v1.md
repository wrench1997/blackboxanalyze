# PG-214 Pikachu fixed-runtime SQL loop

device=cuda:0; fresh containers=14; episodes=14; GET=10; POST=4
mysqli health gates=14; clean resets=14; AI sends=14; reference sends=14
typed response-shape evaluator available=14; AI/reference shape agreement=14; database unavailable=0

每个 route episode 都从派生镜像的全新无 volume 容器开始，并通过 mysqli(root/pikachu) 健康门；没有用 docker restart 伪装数据库 reset。结果只证明 GET/POST 能到达后端并产生响应形状，不证明 SQL AST、查询结果或漏洞 payload 成功。
