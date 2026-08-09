# PG-212 Pikachu SQL response-shape loop

device=cuda:0; fresh containers=14; episodes=14; GET=10; POST=4
AI sends=0; independent reference sends=14; database unavailable=14; abstain=14

每个 route episode 都使用 pinned digest 的全新无 volume 容器；docker restart 仅作为健康恢复手段，不作为干净数据库 reset。Pikachu 当前返回 database configuration failure，这是环境阻塞，不是 SQL 漏洞。
